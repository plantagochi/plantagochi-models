"""
Persona.update()의 결과(센서 이름 -> detect_event 결과 dict)를 보고, 게임의
"상태이상"처럼 지금 활성화된 상태이상 문구 목록을 뽑아내는 순수 함수. mood.py와
완전히 같은 위치의 해석 계층이다 — Persona/StateEngine은 "상태이상"이라는 개념을
전혀 모르고, PlantAI가 이 함수를 불러 쓴다.

mood.py와 마찬가지로, 상태이상 문구도 종(species)과 무관하게 앱 전체에서 통일된
UI 신호라(대사 문구와 달리) 종별 JSON이 아니라 이 파일에 고정값으로 둔다. — 이
결정은 사용자에게 보고 완료(2026-09-10 대화).

대상
----
- disease(DiseaseStateMachine): confirmed state가 BACTERIAL/FUNGAL일 때만
  ("박테리아"/"곰팡이"). HEALTHY는 상태이상이 아니다.
- temperature/humidity/soil_temp/soil_humidity(LinearStateMachine): state가
  VALUE_CRITICAL_LOW/VALUE_CRITICAL_HIGH일 때만. 완만한 VALUE_LOW/VALUE_HIGH는
  상태이상으로 치지 않는다("극단적으로"라는 표현에 맞춰 critical 구간만 반영).

mood와 마찬가지로 이벤트가 아니라 "현재 state" 기준이다 — 상태이상은 "지금 이
상태다"를 보여주는 것이지 "방금 바뀌었다"를 보여주는 게 아니기 때문이다.
"""

from enum import Enum
from typing import Dict, List

from planta_gochi.persona.state_engine import DiseaseState, LinearState


class StatusEffect(str, Enum):
    BACTERIAL_INFECTION = "박테리아"
    FUNGAL_INFECTION = "곰팡이"
    SCORCHING_TEMPERATURE = "타는듯한온도"
    FREEZING_TEMPERATURE = "어는듯한온도"
    JUNGLE_HUMIDITY = "정글같은습도"
    DESERT_HUMIDITY = "사막같은습도"
    ROOTS_BURNING = "타들어가는뿌리"
    ROOTS_FROZEN = "얼어붙은뿌리"
    ROOTS_ROTTING = "썩어가는뿌리"
    ROOTS_PARCHED = "메마른뿌리"


_DISEASE_STATUS_EFFECTS = {
    DiseaseState.BACTERIAL: StatusEffect.BACTERIAL_INFECTION,
    DiseaseState.FUNGAL: StatusEffect.FUNGAL_INFECTION,
    # HEALTHY는 의도적으로 매핑에 없음 -> 상태이상 없음
}

# 센서 이름마다 "critical 구간 state -> 상태이상"만 담는다. VALUE_LOW/VALUE_HIGH/
# VALUE_MID는 어디에도 없으므로 자동으로 상태이상 취급되지 않는다.
_LINEAR_CRITICAL_STATUS_EFFECTS = {
    "temperature": {
        LinearState.VALUE_CRITICAL_HIGH: StatusEffect.SCORCHING_TEMPERATURE,
        LinearState.VALUE_CRITICAL_LOW: StatusEffect.FREEZING_TEMPERATURE,
    },
    "humidity": {
        LinearState.VALUE_CRITICAL_HIGH: StatusEffect.JUNGLE_HUMIDITY,
        LinearState.VALUE_CRITICAL_LOW: StatusEffect.DESERT_HUMIDITY,
    },
    "soil_temp": {
        LinearState.VALUE_CRITICAL_HIGH: StatusEffect.ROOTS_BURNING,
        LinearState.VALUE_CRITICAL_LOW: StatusEffect.ROOTS_FROZEN,
    },
    "soil_humidity": {
        LinearState.VALUE_CRITICAL_HIGH: StatusEffect.ROOTS_ROTTING,
        LinearState.VALUE_CRITICAL_LOW: StatusEffect.ROOTS_PARCHED,
    },
}


def extract_status_effects(persona_result: Dict[str, dict]) -> List[StatusEffect]:
    """
    persona.update(sensor_values)가 돌려준 dict를 그대로 받아, 지금 활성화된 상태이상
    목록을 뽑는다. Persona/엔진 상태를 건드리지 않는 순수 함수라, 이미 계산된 update()
    결과에 대해 몇 번을 다시 호출해도 안전하다. 활성화된 상태이상이 없으면 빈 list.
    순서는 disease -> temperature -> humidity -> soil_temp -> soil_humidity 고정.
    """
    effects: List[StatusEffect] = []

    disease_result = persona_result.get("disease")
    if disease_result is not None and "state" in disease_result:
        effect = _DISEASE_STATUS_EFFECTS.get(disease_result["state"])
        if effect is not None:
            effects.append(effect)

    for sensor_name, state_effects in _LINEAR_CRITICAL_STATUS_EFFECTS.items():
        result = persona_result.get(sensor_name)
        if result is None or "state" not in result:
            continue
        effect = state_effects.get(result["state"])
        if effect is not None:
            effects.append(effect)

    return effects
