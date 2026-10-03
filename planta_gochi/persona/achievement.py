"""
Persona.update()가 매 틱 만들어내는 센서별 결과를 관찰하다가, SupportedAchivements
(planta_gochi.persona.supported_achivements)에 정의된 것 중 조건을 만족하는 게 생기면
누적해서 기억해두는 트래커.

mood.py/status_effect.py와 달리 이 파일은 순수 함수가 아니다 — achievement는 "지금 이
순간의 state"가 아니라 "지금까지 한 번이라도 조건을 만족했는가"를 기억해야 하므로,
Persona가 세션 내내 하나의 AchievementTracker 인스턴스를 들고 있다가 매 update()마다
observe()를 호출해 내부 집합을 갱신한다.

휘발성(volatile)
-----------------
이 트래커가 들고 있는 "지금까지 달성한 achievement 집합"은 Persona.get_state()/
load_state()로 내보내거나 복원하지 않는다(사용자 확인, 2026-09-29 대화) — 즉 저장했다가
다시 불러오면 achievement는 전부 초기화된다. StateEngine.get_state()가 담는 "센서 값을
이어가기 위한 순수 상태"와는 다른 층위의 게임적(UI) 개념이라 의도적으로 분리했다.

감지 방법 정리 (SupportedAchivements 정의 순서와 동일)
--------------------------------------------------------
- STATUS_RECOVERY_TRIO: LinearEvents/DiseaseEvents로 "critical(또는 질병) 상태에서
  정상으로 돌아오는" 전이를 감지해, status_effect.py와 같은 매핑으로 StatusEffect
  종류를 얻고, 서로 다른 종류가 3개 모이면 달성. status_effect.py의 매핑 테이블을
  그대로 재사용한다(같은 "critical 구간 -> 상태이상 종류" 정의가 두 곳에서 어긋나지
  않도록).
- FUNGAL_CURE / BACTERIAL_CURE: disease의 raw_event가 FUNGAL_TO_HEALTHY /
  BACTERIAL_TO_HEALTHY면 즉시 달성.
- GROWTH_STAGE_FIVE: growth.stage == 5.
- GROWTH_STAGE_SKIP: growth.stage를 틱마다 기억해뒀다가, 직전 stage보다 2 이상
  올라가면 달성(GrowthStateMachine 자체는 건드리지 않고, 트래커가 직접 이전 값을
  들고 있는다 — stage는 원래 전이 이벤트가 없는 파생값이라서).

  알려진 한계: GrowthStateMachine.detect_event()는 stage를 canopy/leaf_size처럼
  TrendStateMachine으로 스무딩하지 않고, 그 프레임의 raw leaf_count/canopy_ratio/
  leaf_size_ratio로 classify_growth_stage()를 매번 새로 돌린 결과를 그대로 쓴다(반면
  canopy_machine/leaf_size_machine은 MAD 이상치 제거 + EMA + 회귀 slope를 거쳐야
  추세가 확정되므로 여러 프레임이 필요하다). 그래서 이 achievement는 실제로 며칠에
  걸쳐 급성장한 경우뿐 아니라, 연속된 두 사진의 세그멘테이션 노이즈(각도/조명 차이
  등)만으로도 stage가 한 틱 만에 2단계 이상 튀면서 터질 수 있다 — 사용자에게 보고 후
  "지금은 그대로 두고 알려진 한계로 문서화만 한다"로 확인 완료(2026-09-29 대화).
  더 엄격하게 만들려면 stage 자체에도 DiscreteCountMachine류의 confirm_streak
  디바운스를 추가해 "confirmed_stage"끼리 비교하는 방법이 있다.
- SUDDEN_ENVIRONMENT_SHIFT: temperature/humidity/soil_temp/soil_humidity 중 하나라도
  LinearEvents 상 밴드가 2단계 이상 점프(사용자 확인: "환경 센서 2단계 이상 점프").
- HIGH_HUMIDITY: humidity의 state가 VALUE_CRITICAL_HIGH(상추.json 기준 90 이상과
  같은 구간 — status_effect.py의 "정글같은습도"와 동일 조건).
- TOUCH_GRASS: growth.raw_metrics.canopy_ratio >= 0.5.
- NIGHT_OWL_FARMER: 00~04시(now_fn 기준 로컬 시각) 사이에 soil_humidity가 직전
  관측값보다 상승 — "물주기/관리" 자체를 감지할 센서가 없어 soil_humidity 상승으로
  추론하기로 사용자 확인(2026-09-29 대화). 테스트에서 시각을 주입할 수 있도록 now_fn을
  받는다.

"위기 대응 전문가"는 SupportedAchivements에도 없다 — 상세 이유는 그쪽 docstring 참고.
"""

from __future__ import annotations

import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from planta_gochi.persona.state_engine import DiseaseEvents, DiseaseState, LinearState
from planta_gochi.persona.status_effect import (
    StatusEffect,
    DISEASE_STATUS_EFFECTS,
    LINEAR_CRITICAL_STATUS_EFFECTS,
)
from planta_gochi.persona.supported_achivements import SupportedAchivements

_ENV_LINEAR_SENSORS = ("temperature", "humidity", "soil_temp", "soil_humidity")

_TOUCH_GRASS_CANOPY_RATIO = 0.5
_NIGHT_OWL_START_HOUR = 0
_NIGHT_OWL_END_HOUR = 4  # [0, 4) 시 사이 (자정 ~ 새벽 4시)


def _band_jump(raw_event) -> int:
    """LinearEvents(prev*10+curr 인코딩)에서 밴드 차이(절댓값)를 뽑는다. NO_EVENTS(0)면 0."""
    value = raw_event.value
    if value == 0:
        return 0
    prev_band, curr_band = divmod(value, 10)
    return abs(curr_band - prev_band)


def _recovered_from_band(raw_event) -> Optional[int]:
    """critical(1 또는 5) 밴드에서 정상 밴드(2/3/4)로 돌아오는 전이라면 그 critical
    밴드 번호(1 또는 5)를, 아니라면 None을 돌려준다."""
    value = raw_event.value
    if value == 0:
        return None
    prev_band, curr_band = divmod(value, 10)
    if prev_band in (1, 5) and curr_band in (2, 3, 4):
        return prev_band
    return None


class AchievementTracker:
    def __init__(self, now_fn: Callable[[], datetime.datetime] = datetime.datetime.now):
        self._now_fn = now_fn
        self._unlocked: Set[SupportedAchivements] = set()
        self._recovered_effect_types: Set[StatusEffect] = set()
        self._prev_growth_stage: Optional[int] = None
        self._prev_soil_humidity: Optional[float] = None

    def observe(self, sensor_values: Dict[str, Any], results: Dict[str, dict]) -> None:
        """Persona.update()가 매 틱 호출한다. 새 사실을 관찰할 때마다 내부 누적 집합에
        더하기만 할 뿐, 이미 달성한 걸 취소하지 않는다(게임 achievement는 한 번 달성하면
        계속 유지되는 게 자연스럽다는 판단 — 사용자에게 이 해석을 보고 완료)."""
        self._observe_recoveries(results)
        self._observe_disease_cures(results)
        self._observe_growth(results)
        self._observe_sudden_shift(results)
        self._observe_high_humidity(results)
        self._observe_night_owl(sensor_values)

    def unlocked(self) -> List[SupportedAchivements]:
        """상태를 바꾸지 않는 순수 read. SupportedAchivements 정의 순서로 고정해서,
        몇 번을 불러도 항상 같은 순서의 같은 리스트를 돌려준다."""
        return [a for a in SupportedAchivements if a in self._unlocked]

    # ---- 개별 감지 ----

    def _observe_recoveries(self, results: Dict[str, dict]) -> None:
        if SupportedAchivements.STATUS_RECOVERY_TRIO in self._unlocked:
            return  # 이미 달성했으면 더 볼 필요 없음

        for sensor_name, effect_table in LINEAR_CRITICAL_STATUS_EFFECTS.items():
            result = results.get(sensor_name)
            if result is None or "raw_event" not in result:
                continue
            recovered_band = _recovered_from_band(result["raw_event"])
            if recovered_band is None:
                continue
            effect = effect_table.get(LinearState(recovered_band))
            if effect is not None:
                self._recovered_effect_types.add(effect)

        disease_result = results.get("disease")
        if disease_result is not None and "raw_event" in disease_result:
            raw_event = disease_result["raw_event"]
            if raw_event in (DiseaseEvents.BACTERIAL_TO_HEALTHY, DiseaseEvents.FUNGAL_TO_HEALTHY):
                prev_state = DiseaseState(raw_event.value // 10)
                effect = DISEASE_STATUS_EFFECTS.get(prev_state)
                if effect is not None:
                    self._recovered_effect_types.add(effect)

        if len(self._recovered_effect_types) >= 3:
            self._unlocked.add(SupportedAchivements.STATUS_RECOVERY_TRIO)

    def _observe_disease_cures(self, results: Dict[str, dict]) -> None:
        disease_result = results.get("disease")
        if disease_result is None or "raw_event" not in disease_result:
            return
        raw_event = disease_result["raw_event"]
        if raw_event == DiseaseEvents.FUNGAL_TO_HEALTHY:
            self._unlocked.add(SupportedAchivements.FUNGAL_CURE)
        elif raw_event == DiseaseEvents.BACTERIAL_TO_HEALTHY:
            self._unlocked.add(SupportedAchivements.BACTERIAL_CURE)

    def _observe_growth(self, results: Dict[str, dict]) -> None:
        growth_result = results.get("growth")
        if growth_result is None or "stage" not in growth_result:
            return
        stage = growth_result["stage"]

        if stage == 5:
            self._unlocked.add(SupportedAchivements.GROWTH_STAGE_FIVE)

        if self._prev_growth_stage is not None and stage - self._prev_growth_stage >= 2:
            self._unlocked.add(SupportedAchivements.GROWTH_STAGE_SKIP)
        self._prev_growth_stage = stage

        canopy_ratio = growth_result.get("raw_metrics", {}).get("canopy_ratio")
        if canopy_ratio is not None and canopy_ratio >= _TOUCH_GRASS_CANOPY_RATIO:
            self._unlocked.add(SupportedAchivements.TOUCH_GRASS)

    def _observe_sudden_shift(self, results: Dict[str, dict]) -> None:
        for sensor_name in _ENV_LINEAR_SENSORS:
            result = results.get(sensor_name)
            if result is None or "raw_event" not in result:
                continue
            if _band_jump(result["raw_event"]) >= 2:
                self._unlocked.add(SupportedAchivements.SUDDEN_ENVIRONMENT_SHIFT)
                return

    def _observe_high_humidity(self, results: Dict[str, dict]) -> None:
        humidity_result = results.get("humidity")
        if humidity_result is None or "state" not in humidity_result:
            return
        if humidity_result["state"] == LinearState.VALUE_CRITICAL_HIGH:
            self._unlocked.add(SupportedAchivements.HIGH_HUMIDITY)

    def _observe_night_owl(self, sensor_values: Dict[str, Any]) -> None:
        curr = sensor_values.get("soil_humidity")
        prev = self._prev_soil_humidity
        if curr is not None:
            self._prev_soil_humidity = curr

        if curr is None or prev is None or curr <= prev:
            return

        hour = self._now_fn().hour
        if _NIGHT_OWL_START_HOUR <= hour < _NIGHT_OWL_END_HOUR:
            self._unlocked.add(SupportedAchivements.NIGHT_OWL_FARMER)
