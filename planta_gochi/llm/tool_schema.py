"""
OpenRouter/Ollama처럼 서로 다른 LLM 백엔드가 공유하는 "센서별 짧은 대사를 tool call
하나로 한 번에 받아오는" 스키마 조립 로직. 어떤 백엔드를 쓰든 스키마/컨텍스트/응답
파싱 방식은 동일하므로 여기 모아두고 각 Connection 클래스는 실제 API 호출만 담당한다.
"""

from typing import Callable, Dict, List

TOOL_NAME = "generate_persona_output"


def build_tool_schema(keys: List[str], species: str) -> dict:
    """situations의 키(센서 이름)마다 문자열 하나씩 요구하는 함수 스키마를 동적으로 만든다."""
    properties = {
        key: {
            "type": "string",
            "description": f'"{key}" 상태를 반영한 "{species}"의 짧은 대사 (30자 이내)',
        }
        for key in keys
    }
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": f'"{species}" 페르소나가 현재 상태들을 반영해 센서별로 짧은 대사를 생성',
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(keys),
            },
        },
    }


def render_context(species: str, situations: Dict[str, List[str]]) -> str:
    lines = [f'식물 종: "{species}"', ""]
    for key, messages in situations.items():
        lines.append(f"[{key}] " + " / ".join(messages))
    return "\n".join(lines)


def system_prompt(species: str) -> str:
    return (
        f'당신은 식물 "{species}"의 페르소나입니다. 주어진 각 상태에 대해, 그 식물이 '
        f"할 법한 짧은 대사를 30자 이내로 만들어 {TOOL_NAME} 함수를 호출해 응답하세요."
    )


def extract_partial_result(keys: List[str], arguments, warn: Callable[[str], None]) -> Dict[str, str]:
    """
    tool call의 arguments(dict여야 함)에서 요청한 키만큼만 뽑아 {키: 문자열} dict로 돌려준다.
    arguments가 dict가 아니거나 일부 키가 비어 있으면 그만큼만 비운 채로 돌려주고(성공한
    부분은 버리지 않음), warn(...)으로 사유를 알린다. 반환값은 항상 dict(최악의 경우 빈 dict).
    """
    if not isinstance(arguments, dict):
        warn(f"tool call arguments가 dict가 아닙니다: {arguments!r}. 전부 default_dialog로 대체됩니다.")
        return {}

    result = {key: str(arguments[key]) for key in keys if key in arguments}
    missing = [key for key in keys if key not in arguments]
    if missing:
        warn(f"응답에 없는 키 {missing}는 default_dialog로 대체됩니다(나머지는 LLM 응답 사용).")

    return result
