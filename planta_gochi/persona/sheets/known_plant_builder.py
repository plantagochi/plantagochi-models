"""
식물 종(species)별 persona를 JSON spec으로부터 자동으로 만들어주는 builder.

main.py에서는 LinearStateMachine 하나를 코드로 직접 조립했지만, 식물 종마다
온도/습도 등 센서별 프롬프트가 다르기 때문에 그 내용을 JSON으로 분리해두고,
이 파일이 JSON을 읽어 state machine들을 만든 뒤 Persona로 묶어준다.

JSON spec 형태 (known_plants/상추.json 참고):
{
    "name": "상추",
    "sensors": {
        "온도": {
            "type": "linear",
            "thresholds": [5, 15, 25, 35],
            "state_prompts": {"VALUE_MID": "...", ...},
            "event_prompts": {"NO_EVENTS": "...", "MID_TO_HIGH": "...", ...},
            "default_dialog": {"VALUE_MID": "...", "NO_EVENTS": "...", "MID_TO_HIGH": "...", ...}
        },
        "습도": { ... }
    }
}

state_prompts/event_prompts는 LLM에게 상황을 설명해주기 위한 프롬프트 조각이고,
default_dialog는 LLM에 접속할 수 없을 때 대신 사용자에게 그대로 보여줄, 식물
페르소나가 실제로 할 법한 짧은 대사다. state 이름(VALUE_*)과 event 이름(NO_EVENTS,
MID_TO_HIGH 등)을 같은 dict 안에 함께 적으면 되고, 두 종류 이름은 서로 겹치지
않으므로 이름만 보고 LinearState/LinearEvents 중 어디에 속하는지 자동으로 구분한다.
default_dialog는 state_prompts/event_prompts에 있는 모든 state/event를 빠짐없이
커버해야 하며, 하나라도 빠지면 builder가 조립 시점에 바로 에러를 낸다.

engine "type"별 조립 방법은 _ENGINE_BUILDERS에 등록되어 있다. 지금은 "linear"만
있지만, 이후 boolean state machine이 추가되면 build_boolean_state_machine 같은
함수를 만들어 @register_engine_builder("boolean")으로 등록하기만 하면 되고,
이 파일의 나머지 파싱/조립 로직이나 Persona는 전혀 건드릴 필요가 없다.
"""

import json
from pathlib import Path
from typing import Callable, Dict, Union

from planta_gochi.persona.persona import Persona
from planta_gochi.persona.state_engine import (
    LinearEvents,
    LinearState,
    LinearStateMachine,
    StateEngine,
)

KNOWN_PLANTS_DIR = Path(__file__).parent / "known_plants"

_ENGINE_BUILDERS: Dict[str, Callable[[dict], StateEngine]] = {}


def register_engine_builder(engine_type: str):
    """spec["type"] == engine_type인 sensor spec을 StateEngine으로 조립하는 함수를 등록한다."""

    def decorator(builder_fn: Callable[[dict], StateEngine]):
        _ENGINE_BUILDERS[engine_type] = builder_fn
        return builder_fn

    return decorator


def _parse_state_or_event_dict(raw: dict):
    """key가 LinearState 이름이든 LinearEvents 이름이든 알아서 구분해 enum으로 변환한다."""
    parsed = {}
    for name, value in raw.items():
        if name in LinearState.__members__:
            parsed[LinearState[name]] = value
        elif name in LinearEvents.__members__:
            parsed[LinearEvents[name]] = value
        else:
            raise ValueError(
                f"알 수 없는 state/event 이름: {name!r} "
                f"(LinearState: {sorted(LinearState.__members__)}, "
                f"LinearEvents: {sorted(LinearEvents.__members__)})"
            )
    return parsed


@register_engine_builder("linear")
def build_linear_state_machine(spec: dict) -> LinearStateMachine:
    thresholds = spec["thresholds"]
    state_prompts = {
        LinearState[name]: prompt for name, prompt in spec["state_prompts"].items()
    }
    event_prompts = {
        LinearEvents[name]: prompt for name, prompt in spec["event_prompts"].items()
    }
    default_dialog = _parse_state_or_event_dict(spec["default_dialog"])

    missing_states = set(state_prompts) - set(default_dialog)
    missing_events = set(event_prompts) - set(default_dialog)
    missing = missing_states | missing_events
    if missing:
        raise ValueError(
            "default_dialog에 다음 state/event의 대사가 없습니다: "
            f"{sorted(m.name for m in missing)}"
        )

    return LinearStateMachine(
        thresholds=thresholds,
        state_prompts=state_prompts,
        event_prompts=event_prompts,
        default_dialog=default_dialog,
    )


def build_engine(sensor_spec: dict) -> StateEngine:
    engine_type = sensor_spec.get("type", "linear")
    try:
        builder_fn = _ENGINE_BUILDERS[engine_type]
    except KeyError:
        raise ValueError(
            f"알 수 없는 state machine 타입: {engine_type!r} "
            f"(등록된 타입: {sorted(_ENGINE_BUILDERS)})"
        )
    return builder_fn(sensor_spec)


def build_persona_from_spec(plant_spec: dict) -> Persona:
    """{"name": ..., "sensors": {...}} 형태의 dict로부터 Persona를 만든다."""
    engines = {
        sensor_name: build_engine(sensor_spec)
        for sensor_name, sensor_spec in plant_spec["sensors"].items()
    }
    return Persona(name=plant_spec["name"], engines=engines)


def build_persona_from_json(json_path: Union[str, Path]) -> Persona:
    """JSON 파일 경로로부터 Persona를 만든다."""
    with Path(json_path).open(encoding="utf-8") as f:
        plant_spec = json.load(f)
    return build_persona_from_spec(plant_spec)


def build_known_persona(plant_name: str) -> Persona:
    """known_plants/ 디렉토리에 미리 등록된 종 이름(예: "상추")으로부터 Persona를 만든다."""
    json_path = KNOWN_PLANTS_DIR / f"{plant_name}.json"
    if not json_path.exists():
        known = sorted(p.stem for p in KNOWN_PLANTS_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"등록되지 않은 식물 종: {plant_name!r} (등록된 종: {known})"
        )
    return build_persona_from_json(json_path)


def build_lettuce_persona() -> Persona:
    """온도/습도 linear state machine을 가진 "상추" persona 예시."""
    return build_known_persona("상추")


if __name__ == "__main__":
    lettuce = build_lettuce_persona()

    print(lettuce.prompts({"온도": 20, "습도": 60}))
    print(lettuce.prompts({"온도": 40, "습도": 60}))
    print(lettuce.prompts({"온도": 10, "습도": 20}))
    print(lettuce.prompts({"온도": 0, "습도": 95}))

    # LLM에 접속할 수 없을 때는 default_dialog를 그대로 사용자에게 보여준다.
    print(lettuce.default_dialogs({"온도": 40, "습도": 60}))
