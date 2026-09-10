from typing import Any, Dict

from planta_gochi.persona.state_engine import StateEngine


class Persona:
    """
    여러 개의 독립적인 StateEngine(예: 온도용 LinearStateMachine, 습도용 LinearStateMachine, ...)을
    센서 이름으로 묶어서 관리하는 객체.

    각 센서 값은 자신의 StateEngine에만 영향을 주며, 센서 간에는 서로 영향을 주고받지 않는다.
    즉 Persona는 "센서 이름 -> engine" 매핑을 들고 있다가, sensor_values를 받으면
    각 engine에 독립적으로 위임하고 결과를 센서 이름별로 모아 돌려줄 뿐이다.

    engine은 StateEngine 인터페이스(detect_event(value) -> dict)만 만족하면 되므로,
    지금은 LinearStateMachine만 쓰이지만 이후 BooleanStateMachine 등이 추가되어도
    Persona 코드는 바뀔 필요가 없다.
    """

    def __init__(self, name: str, engines: Dict[str, StateEngine]):
        self.name = name
        self.engines = engines

    def update(self, sensor_values: Dict[str, Any]) -> Dict[str, dict]:
        """
        sensor_values: {"온도": 23.5, "습도": 40} 처럼 센서 이름 -> 현재 값.
        각 센서에 대응하는 engine을 독립적으로 실행한 뒤,
        {"온도": {...detect_event 결과...}, "습도": {...}} 형태로 반환한다.
        Persona가 모르는 센서 이름은 무시한다.

        예외적으로 "image" 키는 특정 센서 하나를 가리키지 않는다 — 이 persona가 가진
        "이미지를 입력으로 받는" 엔진 전부(growth/disease 등, StateEngine.accepts_image
        참고)에 같은 값을 동시에 넣어주는 fan-out 별칭이다. 즉
            persona.update({"image": img})
        는 growth/disease 센서가 둘 다 있는 persona라면
            persona.update({"growth": img, "disease": img})
        와 정확히 같은 결과를 낸다. 센서 이름을 직접 같이 넘기면(예:
        {"image": img, "growth": other_img}) 그 명시적인 값이 "image"보다 우선한다.
        기존처럼 센서 이름을 직접 쓰는 방식은 전혀 바뀌지 않는다 — "image"는 추가된
        편의 기능일 뿐이다.
        """
        sensor_values = self._expand_image_alias(sensor_values)
        results: Dict[str, dict] = {}
        for sensor_name, value in sensor_values.items():
            engine = self.engines.get(sensor_name)
            if engine is None:
                continue
            results[sensor_name] = engine.detect_event(value)
        return results

    def _expand_image_alias(self, sensor_values: Dict[str, Any]) -> Dict[str, Any]:
        """"image" 키가 있으면, 그 값을 accepts_image인 모든 엔진의 센서 이름으로 복제해
        넣은 새 dict를 돌려준다. "image"가 없으면 원본을 그대로 돌려준다(불필요한 복사
        없음). 명시적으로 같이 넘긴 센서 값은 덮어쓰지 않는다."""
        if "image" not in sensor_values:
            return sensor_values

        expanded = dict(sensor_values)
        image_value = expanded.pop("image")
        for sensor_name, engine in self.engines.items():
            if getattr(engine, "accepts_image", False) and sensor_name not in expanded:
                expanded[sensor_name] = image_value
        return expanded

    def get_state(self) -> Dict[str, dict]:
        """
        {센서 이름: engine.get_state()} 형태로 모든 센서의 현재 상태를 모아 돌려준다.
        thresholds/prompts 같은 정적 설정은 빠져 있다 — JSON(known_plant_builder)에서 다시
        만들어지므로, 여기 담기는 건 순수하게 update()를 거치며 바뀌는 값들뿐이다.
        이 dict는 즉시 pickle/json 등으로 저장했다가 load_state()로 그대로 복원할 수 있다.
        """
        return {sensor_name: engine.get_state() for sensor_name, engine in self.engines.items()}

    def load_state(self, states: Dict[str, dict]) -> None:
        """
        get_state()가 만든 dict를 받아 각 센서의 engine 상태를 복원한다. Persona 자체(이름,
        engine 종류/설정)는 이미 known_plant_builder 등으로 구성되어 있다고 가정하고, 그
        위에 저장해둔 state만 덮어씌운다. Persona가 모르는 센서 이름은 무시한다.
        """
        for sensor_name, state in states.items():
            engine = self.engines.get(sensor_name)
            if engine is None:
                continue
            engine.load_state(state)