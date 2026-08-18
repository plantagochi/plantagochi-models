"""
로컬에서 돌아가는 Ollama 서버로 프롬프트/대사를 생성하는, LLMConnection의 대체 구현.
"개발 환경에서 OpenRouter 없이 로컬 모델로 빠르게 테스트"하는 용도라, LLMConnection과
완전히 같은 ask_batch(species, situations) -> Optional[Dict[str, str]] 계약을 따른다 —
PlantAI(persona, llm=OllamaConnection())처럼 그대로 갈아끼울 수 있다.

ollama 파이썬 패키지가 필요하다:
    pip install ollama
    또는
    pip install "planta-gochi-models[ollama]"

기본 설치(pip install planta-gochi-models)에는 포함되지 않는 개발 전용 의존성이라,
이 모듈은 ollama가 없어도 import 자체는 실패하지 않는다(지연 import). 실제로
OllamaConnection을 생성하는 시점에만 ollama가 없으면 설치 방법을 안내하는 에러를 낸다.
"""

import json
import sys
from typing import Dict, List, Optional

from planta_gochi.llm import tool_schema

DEFAULT_MODEL = "llama3.1"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 300


def _warn(message: str) -> None:
    print(f"[OllamaConnection] {message}", file=sys.stderr)


class OllamaConnection:
    """
    species(식물 종 이름)와 situations({센서 이름: 상황 설명 목록})를 받아, 로컬 Ollama
    서버에게 센서별 짧은 대사를 "한 번의 tool-calling 요청"으로 받아오는 클래스.
    스키마 조립/응답 파싱 규칙(부분 성공 허용, stderr 경고)은 LLMConnection과 동일하다
    — 자세한 계약은 tool_schema.py와 LLMConnection의 docstring을 참고.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "OllamaConnection을 쓰려면 ollama 패키지가 필요합니다. "
                '`pip install ollama` 또는 `pip install "planta-gochi-models[ollama]"`로 '
                "설치하세요. (기본 설치에는 포함되지 않는 개발 전용 의존성입니다.)"
            ) from e

        self._client = ollama.Client(host=host)
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_tokens = max_tokens

    def ask_batch(self, species: str, situations: Dict[str, List[str]]) -> Optional[Dict[str, str]]:
        if not situations:
            return None

        keys = list(situations.keys())

        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": tool_schema.system_prompt(species)},
                    {"role": "user", "content": tool_schema.render_context(species, situations)},
                ],
                tools=[tool_schema.build_tool_schema(keys, species)],
                options={"temperature": self.temperature, "num_predict": self.max_tokens},
            )
        except Exception as e:
            _warn(
                f"요청 실패(model={self.model!r}, host={self.host!r}): "
                f"{e!r}. Ollama 서버가 떠 있는지, 모델이 pull되어 있는지 확인하세요. "
                "전부 default_dialog로 대체됩니다."
            )
            return None

        arguments = self._extract_arguments(response)
        if arguments is None:
            _warn(
                f"응답에서 tool call을 찾지 못했습니다(model={self.model!r}). "
                "이 모델이 tool-calling을 지원하지 않을 수 있습니다. 전부 default_dialog로 대체됩니다."
            )
            return None

        return tool_schema.extract_partial_result(keys, arguments, _warn)

    @staticmethod
    def _extract_arguments(response) -> Optional[dict]:
        """
        Ollama의 tool_calls는 arguments가 이미 dict로 오지만(OpenRouter 등은 JSON 문자열로
        옴), 모델 역량에 따라 native tool_calls 대신 answer 텍스트(content)에 JSON을 그대로
        써버리는 경우도 흔해서 그 경우까지 최대한 복구를 시도한다.
        """
        message = response.message
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            arguments = tool_calls[0].function.arguments
            if isinstance(arguments, str):
                try:
                    return json.loads(arguments)
                except json.JSONDecodeError:
                    return None
            return arguments

        # 일부 모델은 tool_calls 없이 content에 {"name": ..., "arguments": {...}} 형태의
        # JSON 텍스트를 그대로 흉내 내어 쓴다. 그 경우도 최대한 구조화된 값을 뽑아본다.
        content = getattr(message, "content", None)
        if not content:
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and isinstance(parsed.get("arguments"), dict):
            return parsed["arguments"]
        return parsed if isinstance(parsed, dict) else None
