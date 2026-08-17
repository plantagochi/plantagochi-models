"""
OpenRouter의 chat/completions endpoint에 접속해, 식물 종/여러 상황을 바탕으로
persona가 할 법한 짧은 대사들을 "한 번의 요청"으로 받아오는 아주 얇은 wrapper.

Persona는 온도/습도/growth처럼 여러 dimension을 동시에 들고 있고, 각 dimension마다
따로 요청을 보내면(N개 dimension = N번 request) 레이턴시 병목이 심해진다. 그래서
OpenRouter의 tools(function calling) 필드로 "센서 이름 -> 짧은 대사" 스키마를 동적으로
만들어 한 번의 요청에 실어 보내고, 모델이 tool call로 구조화된 응답을 한 번에 돌려주게
한다.

키가 없거나, timeout이 나거나, 접근이 안 되거나, tool call 응답 파싱에 실패하는 등
어떤 이유로든 실패하면 예외를 던지지 않고 None을 돌려준다. 실패 시 default_dialog로
대체하는 것은 호출부(PlantAI)의 몫이다.

외부 HTTP 라이브러리(requests 등) 없이 표준 라이브러리 urllib만 사용한다.
"""

import json
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

OPENROUTER_AI_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# openrouter/auto는 tools(function calling)를 지원하지 않는 모델로 라우팅될 수 있어서,
# tool call을 강제하는 이 클래스는 function calling이 확인된 무료 모델을 기본값으로 쓴다.
DEFAULT_MODEL = "google/gemma-4-31b-it:free"
DEFAULT_TIMEOUT = 10  # seconds
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 300

_TOOL_NAME = "generate_persona_output"

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


def _load_env_file(path: Path) -> dict:
    """.env 파일을 KEY=VALUE 형태로 최소한만 파싱한다. (python-dotenv 등 의존성 없이)"""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


_DOTENV = _load_env_file(_ENV_PATH)


def _dotenv_get(key: str) -> Optional[str]:
    # 실제 환경변수가 이미 설정되어 있으면 그것을 우선한다.
    return os.environ.get(key) or _DOTENV.get(key) or None


def _build_tool_schema(keys: List[str], species: str) -> dict:
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
            "name": _TOOL_NAME,
            "description": f'"{species}" 페르소나가 현재 상태들을 반영해 센서별로 짧은 대사를 생성',
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(keys),
            },
        },
    }


def _render_context(species: str, situations: Dict[str, List[str]]) -> str:
    lines = [f'식물 종: "{species}"', ""]
    for key, messages in situations.items():
        lines.append(f"[{key}] " + " / ".join(messages))
    return "\n".join(lines)


class LLMConnection:
    """
    species(식물 종 이름)와 situations({센서 이름: 상황 설명 목록})를 받아 OpenRouter LLM에게
    센서별 짧은 대사를 "한 번의 tool-calling 요청"으로 받아오는 클래스.

    ask_batch()는 성공하면 {센서 이름: 대사} dict(situations의 모든 키를 포함), 실패하면
    None을 돌려준다. None을 돌려주는 경우:
    - API 키가 없을 때
    - 네트워크 오류/timeout이 났을 때
    - 모델이 tool call을 하지 않았거나, 응답이 JSON으로 파싱되지 않거나, 요청한 키를
      다 채우지 않는 등 응답 형식이 기대와 다를 때
    무엇이 됐든 실패로 간주하고 조용히 None을 돌려준다(부분 성공은 인정하지 않는다 —
    호출부가 "전부 성공 아니면 전부 fallback"만 신경 쓰면 되게 하기 위함).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.api_key = api_key or _dotenv_get("PLANTA_GOCHI_LLM_KEY")
        self.endpoint = endpoint or _dotenv_get("OPENROUTER_AI_ENDPOINT") or OPENROUTER_AI_ENDPOINT
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def ask_batch(self, species: str, situations: Dict[str, List[str]]) -> Optional[Dict[str, str]]:
        if not self.api_key or not situations:
            return None

        keys = list(situations.keys())
        system_content = (
            f'당신은 식물 "{species}"의 페르소나입니다. 주어진 각 상태에 대해, 그 식물이 '
            "할 법한 짧은 대사를 30자 이내로 만들어 generate_persona_output 함수를 호출해 응답하세요."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": _render_context(species, situations)},
            ],
            "tools": [_build_tool_schema(keys, species)],
            "tool_choice": {"type": "function", "function": {"name": _TOOL_NAME}},
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))

            tool_calls = body["choices"][0]["message"]["tool_calls"]
            arguments = json.loads(tool_calls[0]["function"]["arguments"])

            if not isinstance(arguments, dict) or not all(key in arguments for key in keys):
                return None

            return {key: str(arguments[key]) for key in keys}
        except Exception:
            # 키 없음/timeout/네트워크 오류/tool call 없음/JSON 파싱 실패 등 무엇이든 실패로 간주.
            return None
