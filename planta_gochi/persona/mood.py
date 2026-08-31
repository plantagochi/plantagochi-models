"""
Persona.update()의 결과(센서 이름 -> detect_event 결과 dict)를 보고, UI 아바타가 지을
표정 하나(Expression)를 뽑아내는 순수 함수. Persona/StateEngine은 "mood"라는 개념을
전혀 모르고, 이 모듈이 그 결과를 소비하는 별개의 해석 계층이다 (PlantAI가 같은 결과를
소비해서 LLM 프롬프트를 만드는 것과 동일한 위치).

설계
----
1. (센서 이름, 현재 state) -> mood_key(서사적 이름, 예: "freezing") 로 먼저 좁히고,
   2. mood_key -> Expression 으로 다시 좁힌다.
   같은 LinearState.VALUE_CRITICAL_LOW라도 온도(freezing)와 습도(desperate_thirst)는
   서사적으로 다르기 때문에 mood_key는 반드시 "센서 이름"까지 같이 봐야 한다 — enum
   값 하나만으로는 결정할 수 없다.

이벤트가 아니라 상태(state) 기준
--------------------------------
이벤트(raw_event)는 대부분의 프레임에서 NO_EVENTS라 "지금 위험한 상태가 계속되고
있다"는 걸 표현하지 못한다. 그래서 온도/습도처럼 등급이 있는 센서(LinearStateMachine,
TrendStateMachine)는 항상 "현재 state"를 기준으로 mood_key를 정한다. 다만
leaf_count(DiscreteCountMachine)는 등급 개념이 없고(state가 항상 STABLE) 새 잎이
났다는 것 자체가 유일한 신호라서, 거기에만 예외적으로 이벤트(raw_event)를 쓴다.

mood_key -> Expression 매핑은 종(species)과 무관하게 앱 전체에서 통일된 UI 신호라
(대사 문구와 달리) 종별 JSON이 아니라 이 파일에 고정값으로 둔다.

growth의 특수 규칙
-------------------
growth는 leaf_count/canopy_trend/leaf_size_trend 세 개를 이미 합성한 결과라 mood
후보도 셋에서 나온다. 이 세 신호 중 "부정적인" 신호(잎 감소/캐노피 축소/개별 잎 축소)가
2개 이상이면 growth 전체를 DISTRESSED로 본다(팀 합의: 하나만 나빠지는 건 대세에
지장이 없는 노이즈로 보고, 두 개 이상 동시에 나빠질 때만 진짜 위험 신호로 취급).
그래서 "부정적" 개별 신호(leaf_loss, wilting) 자체는 mood_key -> Expression에서
NEUTRAL로 두고, DISTRESSED는 이 2개-이상 규칙에서만 나온다 — 안 그러면 신호 하나만
나빠져도 항상 DISTRESSED가 되어 팀 합의(2개 이상)가 무의미해진다.

여러 센서 후보 중 하나 고르기
-----------------------------
같은 tick에 센서마다 서로 다른 mood가 나올 수 있으므로, Expression에 심각도를 매겨
가장 심각한 것을 최종적으로 고른다: DISTRESSED > EXCITED > HAPPY > NEUTRAL.
아무 후보도 없으면(예: 아직 아무 센서도 안 들어온 update) NEUTRAL이 기본값이다.
"""

from enum import Enum
from typing import Dict, List

from planta_gochi.persona.state_engine import CountEvents, LinearState, TrendState


class Expression(str, Enum):
    HAPPY = "happy"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    DISTRESSED = "distressed"


_SEVERITY = {
    Expression.NEUTRAL: 0,
    Expression.HAPPY: 1,
    Expression.EXCITED: 2,
    Expression.DISTRESSED: 3,
}

# ---- 1단계: (센서, state/event) -> mood_key -----------------------------------

_TEMPERATURE_MOOD = {
    LinearState.VALUE_CRITICAL_LOW: "freezing",
    LinearState.VALUE_LOW: "chilly",
    LinearState.VALUE_MID: "sunbathing_happy",
    LinearState.VALUE_HIGH: "overheating",
    LinearState.VALUE_CRITICAL_HIGH: "scorching",
}

_HUMIDITY_MOOD = {
    LinearState.VALUE_CRITICAL_LOW: "desperate_thirst",
    LinearState.VALUE_LOW: "parched",
    LinearState.VALUE_MID: "refreshed",
    LinearState.VALUE_HIGH: "muggy",
    LinearState.VALUE_CRITICAL_HIGH: "waterlogged",
}

_SOIL_TEMP_MOOD = {
    LinearState.VALUE_CRITICAL_LOW: "frozen_roots",
    LinearState.VALUE_LOW: "cool_roots",
    LinearState.VALUE_MID: "warm_roots",
    LinearState.VALUE_HIGH: "hot_roots",
    LinearState.VALUE_CRITICAL_HIGH: "roots_scorching",
}

_SOIL_HUMIDITY_MOOD = {
    LinearState.VALUE_CRITICAL_LOW: "roots_thirsty",
    LinearState.VALUE_LOW: "soil_drying",
    LinearState.VALUE_MID: "well_watered",
    LinearState.VALUE_HIGH: "soil_soggy",
    LinearState.VALUE_CRITICAL_HIGH: "roots_drowning",
}

# 센서 이름(known_plant_builder JSON의 sensors 키)마다 어떤 mood 테이블을 쓸지.
# 여기 없는 센서 이름(known_plant_builder가 모르는 것 포함)은 mood 추출에서 무시된다.
_LINEAR_SENSOR_MOOD_TABLES = {
    "temperature": _TEMPERATURE_MOOD,
    "humidity": _HUMIDITY_MOOD,
    "soil_temp": _SOIL_TEMP_MOOD,
    "soil_humidity": _SOIL_HUMIDITY_MOOD,
}

# growth 내부 leaf_count(DiscreteCountMachine)는 state가 없고 event만 의미 있다.
_GROWTH_COUNT_MOOD = {
    CountEvents.NO_EVENTS: "neutral",
    CountEvents.INCREASED: "new_leaf_joy",
    CountEvents.DECREASED: "leaf_loss",
}

# growth 내부 canopy_trend/leaf_size_trend(둘 다 TrendStateMachine)는 공통으로 쓴다.
_GROWTH_TREND_MOOD = {
    TrendState.INSUFFICIENT_DATA: "neutral",
    TrendState.GROWING: "proud_growth",
    TrendState.STAGNANT: "steady_growth",
    TrendState.DECLINING: "wilting",
}

# ---- 2단계: mood_key -> Expression ---------------------------------------------
# 팀원이 준 4개(desperate_thirst/freezing/proud_growth/new_leaf_joy/sunbathing_happy/
# neutral)는 그대로 두고 나머지 센서/구간에 맞춰 확장했다.
# leaf_loss/wilting이 NEUTRAL인 이유는 위 docstring의 "growth의 특수 규칙" 참고.
MOOD_TO_EXPRESSION: Dict[str, Expression] = {
    # 공통
    "neutral": Expression.NEUTRAL,
    # 온도
    "freezing": Expression.DISTRESSED,
    "chilly": Expression.NEUTRAL,
    "sunbathing_happy": Expression.HAPPY,
    "overheating": Expression.NEUTRAL,
    "scorching": Expression.DISTRESSED,
    # 습도
    "desperate_thirst": Expression.DISTRESSED,
    "parched": Expression.NEUTRAL,
    "refreshed": Expression.HAPPY,
    "muggy": Expression.NEUTRAL,
    "waterlogged": Expression.DISTRESSED,
    # 토양 온도
    "frozen_roots": Expression.DISTRESSED,
    "cool_roots": Expression.NEUTRAL,
    "warm_roots": Expression.HAPPY,
    "hot_roots": Expression.NEUTRAL,
    "roots_scorching": Expression.DISTRESSED,
    # 토양 습도
    "roots_thirsty": Expression.DISTRESSED,
    "soil_drying": Expression.NEUTRAL,
    "well_watered": Expression.HAPPY,
    "soil_soggy": Expression.NEUTRAL,
    "roots_drowning": Expression.DISTRESSED,
    # growth
    "new_leaf_joy": Expression.EXCITED,
    "leaf_loss": Expression.NEUTRAL,  # 단독으로는 NEUTRAL. 2개 이상 겹치면 _growth_expression이 DISTRESSED로 override.
    "proud_growth": Expression.EXCITED,
    "steady_growth": Expression.NEUTRAL,
    "wilting": Expression.NEUTRAL,  # leaf_loss와 동일한 이유.
}

# growth의 부정적 신호로 셀 대상. 2개 이상이면 growth는 DISTRESSED로 확정된다.
_GROWTH_NEGATIVE_COUNT_EVENTS = {CountEvents.DECREASED}
_GROWTH_NEGATIVE_TREND_STATES = {TrendState.DECLINING}


def _severity(expression: Expression) -> int:
    return _SEVERITY[expression]


def _growth_expression(growth_result: dict) -> Expression:
    """growth 서브 신호 3개(leaf_count/canopy_trend/leaf_size_trend)를 하나의 Expression으로
    합친다. 부정적 신호가 2개 이상이면 무조건 DISTRESSED, 그 외에는 세 후보 중 가장
    심각한 Expression을 고른다."""
    count_event = growth_result["leaf_count"]["raw_event"]
    canopy_state = growth_result["canopy_trend"]["state"]
    leaf_size_state = growth_result["leaf_size_trend"]["state"]

    negative_count = (
        (count_event in _GROWTH_NEGATIVE_COUNT_EVENTS)
        + (canopy_state in _GROWTH_NEGATIVE_TREND_STATES)
        + (leaf_size_state in _GROWTH_NEGATIVE_TREND_STATES)
    )
    if negative_count >= 2:
        return Expression.DISTRESSED

    mood_keys = [
        _GROWTH_COUNT_MOOD.get(count_event, "neutral"),
        _GROWTH_TREND_MOOD.get(canopy_state, "neutral"),
        _GROWTH_TREND_MOOD.get(leaf_size_state, "neutral"),
    ]
    candidates = [MOOD_TO_EXPRESSION.get(key, Expression.NEUTRAL) for key in mood_keys]
    return max(candidates, key=_severity)


def extract_expression(persona_result: Dict[str, dict]) -> Expression:
    """
    persona.update(sensor_values)가 돌려준 dict를 그대로 받아 Expression 하나를 뽑는다.
    Persona/엔진 상태를 건드리지 않는 순수 함수라, 이미 계산된 update() 결과에 대해
    몇 번을 다시 호출해도 안전하다.
    """
    candidates: List[Expression] = []

    for sensor_name, mood_table in _LINEAR_SENSOR_MOOD_TABLES.items():
        result = persona_result.get(sensor_name)
        if result is None or "state" not in result:
            continue
        mood_key = mood_table.get(result["state"], "neutral")
        candidates.append(MOOD_TO_EXPRESSION.get(mood_key, Expression.NEUTRAL))

    growth_result = persona_result.get("growth")
    if growth_result is not None:
        candidates.append(_growth_expression(growth_result))

    if not candidates:
        return Expression.NEUTRAL

    return max(candidates, key=_severity)
