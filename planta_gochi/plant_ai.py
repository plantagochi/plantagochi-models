"""
Persona(state machine 기반 prompt/default_dialog)와 LLMConnection(실제 OpenRouter 호출)을
묶어, 센서 값을 받아 최종적으로 사용자에게 보여줄 대사 목록을 만들어주는 조립 클래스.
"""

from typing import Any, Dict, List, Optional

from planta_gochi.llm.connection import LLMConnection
from planta_gochi.persona.persona import Persona


class PlantAI:
    def __init__(self, persona: Persona, llm: Optional[LLMConnection] = None):
        self.persona = persona
        self.llm = llm or LLMConnection()

    def speak(self, sensor_values: Dict[str, Any]) -> List[str]:
        """
        sensor_values: {"temperature": 23.5, "humidity": 40} 처럼 센서 이름 -> 현재 값.
        센서마다 개별 요청을 보내면 센서 수만큼 레이턴시가 쌓이므로, Persona가 만든
        센서별 prompt(list[str])를 모아 LLM에는 tool-calling으로 "한 번"만 요청해
        센서별 대사를 한 번에 받아온다.

        LLM 요청 자체가 실패하면(키 없음/timeout/응답 형식 이상 등) 전체를 Persona의
        default_dialog로 대체한다. 요청은 성공했지만 일부 센서만 응답에 담겨 왔다면,
        돌아온 센서는 LLM 응답을 쓰고 나머지만 그 센서의 default_dialog로 대체한다
        (돌아온 걸 통째로 버리지 않는다). 실패/누락 사유는 LLMConnection이 stderr에
        경고로 남기며, 이 반환값 자체에는 영향이 없다.

        반환값은 센서 이름과 무관하게, 사용자에게 보여줄 짧은 글들을 담은 flat list[str]이다
        (sensor_values 순서를 따른다). LLM 성공 시에는 센서당 문자열 하나씩이고, fallback
        시에는 그 센서의 default_dialog(list[str])를 이어붙이지 않고 그대로 펼쳐 넣는다 —
        최종적으로 사용자에게 보여줄 것들은 짧은 글이어야 하므로, 여러 줄을 하나로 뭉쳐서
        길게 만들지 않는다. INSUFFICIENT_DATA처럼 아직 할 말이 없는 신호는 애초에
        default_dialog가 빈 list라 자동으로 아무것도 추가되지 않는다.
        """
        results = self.persona.update(sensor_values)
        situations = {sensor_name: result["prompt"] for sensor_name, result in results.items()}
        reply = self.llm.ask_batch(self.persona.name, situations)

        messages: List[str] = []
        for sensor_name, result in results.items():
            if reply is not None and sensor_name in reply:
                messages.append(reply[sensor_name])
            else:
                messages.extend(result["default_dialog"])
        return messages
