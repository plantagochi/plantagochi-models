"""
OpenRouter의 chat/completions endpoint에 접속해, 식물 종/여러 상황을 바탕으로
persona가 할 법한 짧은 대사들을 "한 번의 요청"으로 받아오는 아주 얇은 wrapper.

Persona는 온도/습도/growth처럼 여러 dimension을 동시에 들고 있고, 각 dimension마다
따로 요청을 보내면(N개 dimension = N번 request) 레이턴시 병목이 심해진다. 그래서
OpenRouter의 tools(function calling) 필드로 "센서 이름 -> 짧은 대사" 스키마를 동적으로
만들어 한 번의 요청에 실어 보내고, 모델이 tool call로 구조화된 응답을 한 번에 돌려주게
한다.

키가 없거나, timeout이 나거나, 접근이 안 되거나, tool call 응답 파싱에 실패하는 등
어떤 이유로든 실패하면 예외를 던지지 않고 실패한 부분만큼만 비워서 돌려준다(전체 요청
자체가 아예 안 됐으면 None). 어느 부분이 왜 실패했는지는 stderr에 경고로 남기되,
반환값 자체에는 영향을 주지 않는다. default_dialog로 대체하는 것은 호출부(PlantAI)의 몫.

외부 HTTP 라이브러리(requests 등) 없이 표준 라이브러리 urllib만 사용한다.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


def _warn(message: str) -> None:
    print(f"[LLMConnection] {message}", file=sys.stderr)

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

    ask_batch()는 {센서 이름: 대사} dict를 돌려준다. 요청 자체가 실패하면(API 키 없음/
    네트워크 오류·timeout/tool call 없음/JSON 파싱 실패 등) None을 돌려준다. 요청은
    성공했지만 모델이 일부 키를 채우지 않았다면, 채워진 키만큼만 담은 "부분" dict를
    돌려준다(돌아온 게 하나도 없으면 빈 dict) — 성공한 부분까지 통째로 버리지 않기
    위함이다. 어느 경우든 실패/누락 사유는 stderr에 경고로 출력하지만 반환값은 그대로다.
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
        if not situations:
            return None

        if not self.api_key:
            _warn("PLANTA_GOCHI_LLM_KEY가 없어 LLM 호출을 건너뜁니다. default_dialog로 대체됩니다.")
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
        except Exception as e:
            _warn(f"요청 실패(model={self.model!r}): {e!r}. 전부 default_dialog로 대체됩니다.")
            return None

        try:
            tool_calls = body["choices"][0]["message"]["tool_calls"]
            arguments = json.loads(tool_calls[0]["function"]["arguments"])
        except Exception as e:
            _warn(f"응답 파싱 실패(model={self.model!r}): {e!r}. 전부 default_dialog로 대체됩니다.")
            return None

        if not isinstance(arguments, dict):
            _warn(f"tool call arguments가 dict가 아닙니다: {arguments!r}. 전부 default_dialog로 대체됩니다.")
            return None

        result = {key: str(arguments[key]) for key in keys if key in arguments}
        missing = [key for key in keys if key not in arguments]
        if missing:
            _warn(f"응답에 없는 키 {missing}는 default_dialog로 대체됩니다(나머지는 LLM 응답 사용).")

        return result
