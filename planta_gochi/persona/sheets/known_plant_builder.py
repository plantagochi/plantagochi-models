"""
식물 종(species)별 persona를 JSON spec으로부터 자동으로 만들어주는 builder.

main.py에서는 LinearStateMachine 하나를 코드로 직접 조립했지만, 식물 종마다
온도/습도 등 센서별 프롬프트가 다르기 때문에 그 내용을 JSON으로 분리해두고,
이 파일이 JSON을 읽어 state machine들을 만든 뒤 Persona로 묶어준다.

JSON spec 형태 (known_plants/상추.json 참고):
{
    "name": "상추",
    "sensors": {
        "temperature": {
            "type": "linear",
            "thresholds": [5, 15, 25, 35],
            "state_prompts": {"VALUE_MID": "...", ...},
            "event_prompts": {"NO_EVENTS": "...", "MID_TO_HIGH": "...", ...},
            "default_dialog": {"VALUE_MID": "...", "NO_EVENTS": "...", "MID_TO_HIGH": "...", ...}
        },
        "humidity": { ... },
        "growth": {
            "type": "growth",
            "leaf_analyzer": {"backend": "onnx"},
            "count_machine": {
                "confirm_streak": 3,
                "state_prompts": {"STABLE": "..."},
                "event_prompts": {"NO_EVENTS": "...", "INCREASED": "...", "DECREASED": "..."},
                "default_dialog": {"STABLE": "...", "NO_EVENTS": "...", "INCREASED": "...", "DECREASED": "..."}
            },
            "trend_machine": {
                "window": 7, "mad_k": 3.5, "growth_eps": 0.001, "decline_eps": 0.001,
                "state_prompts": {"GROWING": "...", ...},
                "event_prompts": {"NO_EVENTS": "...", "STAGNANT_TO_GROWING": "...", ...},
                "default_dialog": {...}
            }
        },
        "disease": {
            "type": "disease",
            "disease_analyzer": {"backend": "onnx"},
            "confirm_streak": 5,
            "state_prompts": {"HEALTHY": "...", "BACTERIAL": "...", "FUNGAL": "..."},
            "event_prompts": {"NO_EVENTS": "...", "HEALTHY_TO_BACTERIAL": "...", ...},
            "default_dialog": {...}
        }
    }
}

state_prompts/event_prompts는 LLM에게 상황을 설명해주기 위한 프롬프트 조각이고,
default_dialog는 LLM에 접속할 수 없을 때 대신 사용자에게 그대로 보여줄, 식물
페르소나가 실제로 할 법한 짧은 대사다. state 이름과 event 이름을 같은 dict 안에
함께 적으면 되고, 두 종류 이름은 서로 겹치지 않으므로 이름만 보고 자동으로 구분한다.
default_dialog는 state_prompts/event_prompts에 있는 모든 state/event를 빠짐없이
커버해야 하며, 하나라도 빠지면 builder가 조립 시점에 바로 에러를 낸다.

"growth" 타입은 leaf_count(DiscreteCountMachine)를 기본으로 묶은 GrowthStateMachine을
만든다. JSON에 "trend_machine" 섹션이 있으면 canopy/leaf_size 추세(각각 독립된
TrendStateMachine 인스턴스, 문구는 그 섹션 하나를 공유)도 함께 붙고, 없으면 추세 없이
leaf_count만으로 조립된다. leaf_analyzer는 생략하면 기본 backend("onnx")로 생성된다.

"disease" 타입은 DiseaseAnalyzer(bacterial/fungal/healthy 3-클래스 분류기)를
DiscreteCountMachine과 같은 debounce 구조로 감싼 DiseaseStateMachine을 만든다 —
같은 진단이 confirm_streak(기본 5)번 연속 나와야 confirmed_state가 바뀐다.
disease_analyzer는 생략하면 기본 backend("onnx")로 생성된다.

engine "type"별 조립 방법은 _ENGINE_BUILDERS에 등록되어 있다. 지금은
"linear"/"growth"/"disease"가 있지만, 이후 새 타입이 추가되면 build_xxx 함수를 만들어
@register_engine_builder("xxx")로 등록하기만 하면 되고, 이 파일의 나머지 파싱/조립
로직이나 Persona는 전혀 건드릴 필요가 없다.
"""

import json
from pathlib import Path
from typing import Callable, Dict, Union

from planta_gochi.persona.persona import Persona
from planta_gochi.persona.state_engine import (
    CountEvents,
    CountState,
    DiscreteCountMachine,
    DiseaseEvents,
    DiseaseState,
    DiseaseStateMachine,
    GrowthStateMachine,
    LinearEvents,
    LinearState,
    LinearStateMachine,
    StateEngine,
    TrendEvents,
    TrendState,
    TrendStateMachine,
)
from planta_gochi.sensory import DiseaseAnalyzer, LeafAnalyzer

KNOWN_PLANTS_DIR = Path(__file__).parent / "known_plants"

_ENGINE_BUILDERS: Dict[str, Callable[[dict], StateEngine]] = {}


def register_engine_builder(engine_type: str):
    """spec["type"] == engine_type인 sensor spec을 StateEngine으로 조립하는 함수를 등록한다."""

    def decorator(builder_fn: Callable[[dict], StateEngine]):
        _ENGINE_BUILDERS[engine_type] = builder_fn
        return builder_fn

    return decorator


def _parse_enum_dict(raw: dict, enum_cls) -> dict:
    """key가 모두 같은 enum 클래스(예: state_prompts는 전부 state 이름)인 dict를 변환한다."""
    return {enum_cls[name]: value for name, value in raw.items()}


def _parse_state_or_event_dict(raw: dict, state_enum, event_enum) -> dict:
    """key가 state_enum 이름이든 event_enum 이름이든 알아서 구분해 enum으로 변환한다."""
    parsed = {}
    for name, value in raw.items():
        if name in state_enum.__members__:
            parsed[state_enum[name]] = value
        elif name in event_enum.__members__:
            parsed[event_enum[name]] = value
        else:
            raise ValueError(
                f"알 수 없는 state/event 이름: {name!r} "
                f"({state_enum.__name__}: {sorted(state_enum.__members__)}, "
                f"{event_enum.__name__}: {sorted(event_enum.__members__)})"
            )
    return parsed


def _validate_dialog_coverage(state_prompts: dict, event_prompts: dict, default_dialog: dict) -> None:
    missing = (set(state_prompts) | set(event_prompts)) - set(default_dialog)
    if missing:
        raise ValueError(
            "default_dialog에 다음 state/event의 대사가 없습니다: "
            f"{sorted(m.name for m in missing)}"
        )


@register_engine_builder("linear")
def build_linear_state_machine(spec: dict) -> LinearStateMachine:
    thresholds = spec["thresholds"]
    state_prompts = _parse_enum_dict(spec["state_prompts"], LinearState)
    event_prompts = _parse_enum_dict(spec["event_prompts"], LinearEvents)
    default_dialog = _parse_state_or_event_dict(spec["default_dialog"], LinearState, LinearEvents)
    _validate_dialog_coverage(state_prompts, event_prompts, default_dialog)

    return LinearStateMachine(
        thresholds=thresholds,
        state_prompts=state_prompts,
        event_prompts=event_prompts,
        default_dialog=default_dialog,
    )


def _build_trend_state_machine(spec: dict) -> TrendStateMachine:
    state_prompts = _parse_enum_dict(spec["state_prompts"], TrendState)
    event_prompts = _parse_enum_dict(spec["event_prompts"], TrendEvents)
    default_dialog = _parse_state_or_event_dict(spec["default_dialog"], TrendState, TrendEvents)
    _validate_dialog_coverage(state_prompts, event_prompts, default_dialog)

    kwargs = {key: spec[key] for key in ("window", "mad_k", "growth_eps", "decline_eps") if key in spec}

    return TrendStateMachine(
        state_prompts=state_prompts,
        event_prompts=event_prompts,
        default_dialog=default_dialog,
        **kwargs,
    )


def _build_discrete_count_machine(spec: dict) -> DiscreteCountMachine:
    state_prompts = _parse_enum_dict(spec["state_prompts"], CountState)
    event_prompts = _parse_enum_dict(spec["event_prompts"], CountEvents)
    default_dialog = _parse_state_or_event_dict(spec["default_dialog"], CountState, CountEvents)
    _validate_dialog_coverage(state_prompts, event_prompts, default_dialog)

    kwargs = {}
    if "confirm_streak" in spec:
        kwargs["confirm_streak"] = spec["confirm_streak"]

    return DiscreteCountMachine(
        state_prompts=state_prompts,
        event_prompts=event_prompts,
        default_dialog=default_dialog,
        **kwargs,
    )


@register_engine_builder("growth")
def build_growth_state_machine(spec: dict) -> GrowthStateMachine:
    leaf_analyzer_spec = spec.get("leaf_analyzer", {})
    count_machine = _build_discrete_count_machine(spec["count_machine"])
    canopy_machine = None
    leaf_size_machine = None
    if "trend_machine" in spec:
        # canopy/leaf_size는 같은 trend_machine 문구를 공유하되, 서로 다른 값을 추적해야
        # 하므로(각자 자기 history/EMA를 들고 있어야 함) 반드시 별도 인스턴스로 만든다.
        canopy_machine = _build_trend_state_machine(spec["trend_machine"])
        leaf_size_machine = _build_trend_state_machine(spec["trend_machine"])

    return GrowthStateMachine(
        count_machine=count_machine,
        canopy_machine=canopy_machine,
        leaf_size_machine=leaf_size_machine,
        leaf_analyzer=LeafAnalyzer(**leaf_analyzer_spec),
    )


@register_engine_builder("disease")
def build_disease_state_machine(spec: dict) -> DiseaseStateMachine:
    state_prompts = _parse_enum_dict(spec["state_prompts"], DiseaseState)
    event_prompts = _parse_enum_dict(spec["event_prompts"], DiseaseEvents)
    default_dialog = _parse_state_or_event_dict(spec["default_dialog"], DiseaseState, DiseaseEvents)
    _validate_dialog_coverage(state_prompts, event_prompts, default_dialog)

    disease_analyzer_spec = spec.get("disease_analyzer", {})
    kwargs = {}
    if "confirm_streak" in spec:
        kwargs["confirm_streak"] = spec["confirm_streak"]

    return DiseaseStateMachine(
        state_prompts=state_prompts,
        event_prompts=event_prompts,
        default_dialog=default_dialog,
        disease_analyzer=DiseaseAnalyzer(**disease_analyzer_spec),
        **kwargs,
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
    """온도/습도 linear state machine과 growth(vision) state machine을 가진 "상추" persona 예시."""
    return build_known_persona("상추")


if __name__ == "__main__":
    lettuce = build_lettuce_persona()

    readings = [
        {"temperature": 20, "humidity": 60},
        {"temperature": 40, "humidity": 60},
        {"temperature": 10, "humidity": 20},
        {"temperature": 0, "humidity": 95},
    ]
    for reading in readings:
        # update()는 한 번만 호출한다. prompts()/default_dialogs()를 같은 값에 각각
        # 호출하면 state machine의 prev_value가 두 번 갱신되어 두 번째 호출이
        # "변화 없음"으로 보인다.
        result = lettuce.update(reading)
        print({k: v["prompt"] for k, v in result.items()})

    # LLM에 접속할 수 없을 때는 default_dialog를 그대로 사용자에게 보여준다.
    result = lettuce.update({"temperature": 20, "humidity": 60})
    print({k: v["default_dialog"] for k, v in result.items()})
