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
        """
        results: Dict[str, dict] = {}
        for sensor_name, value in sensor_values.items():
            engine = self.engines.get(sensor_name)
            if engine is None:
                continue
            results[sensor_name] = engine.detect_event(value)
        return results