# PlantAI (ai.speak / get_mood / get_growth_stage 사용법)

`PlantAI`는 `Persona`(state machine 기반 prompt/default_dialog)와 `LLMConnection`
(OpenRouter 호출)을 묶어서, 센서 값을 넣으면 최종적으로 사용자에게 보여줄 짧은 대사
목록, mood(표정), 성장 단계까지 만들어주는 클래스입니다.

## 기본 사용법

```python
from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.plant_ai import PlantAI

persona = build_known_persona("상추")
ai = PlantAI(persona)

messages = ai.speak({
    "temperature": 20, "humidity": 60,
    "soil_temp": 17, "soil_humidity": 55,
    "growth": "sample_easy.jpg",
})

for message in messages:
    print(" -", message)
```

`speak()`의 반환값은 **사용자에게 보여줄 짧은 글들을 담은 flat `list[str]`**입니다.
어떤 센서에서 나온 문장인지 구분하지 않고, 화면에 표시할 여러 개의 글로 그대로 씁니다.

`"growth"` 센서 값은 파일 경로에 국한되지 않습니다 — `sensor_values["growth"]`는
가공 없이 그대로 `LeafAnalyzer.analyze()`에 전달되므로, [LeafAnalyzer가 지원하는 모든
입력 형식](leaf_analyzer.md)(로컬 경로, http(s) URL, bytes/bytearray, `io.BytesIO`
같은 file-like 객체, PIL Image, numpy array)을 그대로 쓸 수 있습니다.

## 동작 방식

1. `persona.update(sensor_values)`로 센서별 state를 갱신하고, 각 센서의 `prompt`
   (LLM에 넘길 상황 설명, `list[str]`)를 모읍니다.
2. 센서마다 개별 요청을 보내면 센서 수만큼 레이턴시가 쌓이므로, 전부 모아서
   OpenRouter에 **tool-calling으로 요청 한 번만** 보냅니다 (`LLMConnection.ask_batch`).
   요청 스키마는 센서 이름마다 동적으로 필드를 하나씩 만들어 구성됩니다.
3. 모델이 tool call로 `{센서 이름: 대사}`를 돌려주면 그걸 그대로 씁니다.
4. 요청 자체가 실패했거나(API 키 없음, timeout, 네트워크 오류 등), 모델이 일부
   센서만 채워서 응답했다면:
   - **돌아온 센서는 LLM 응답을 그대로 사용**하고
   - **응답에 없는 센서만** 그 센서의 `default_dialog`로 대체합니다.

   즉 "하나라도 실패하면 전부 fallback"이 아니라, 성공한 부분은 최대한 살립니다.
5. `default_dialog`로 대체할 때는 그 센서의 `default_dialog`(`list[str]`)를 한 줄로
   합치지 않고 그대로 펼쳐 넣습니다 — 최종적으로 사용자에게 보여줄 것들은 짧은 글이어야
   하기 때문입니다. `growth`처럼 한 센서가 여러 줄(leaf_count/canopy/leaf_size)을 가진
   경우 fallback 시 여러 개의 짧은 글로 나뉘어 들어갑니다.
6. `TrendStateMachine`이 아직 추세를 판단할 데이터가 부족한 동안(`INSUFFICIENT_DATA`)은
   애초에 `prompt`/`default_dialog`가 빈 `list`라 아무 메시지도 추가되지 않습니다.

## get_mood() / get_growth_stage() — 표정/성장 단계 뽑기

`speak()`는 LLM을 호출하는 무거운 메서드입니다. UI 아바타 표정이나 성장 단계 표시처럼
자주 갱신해야 하는 값은 LLM 없이 `get_mood()`/`get_growth_stage()`로 바로 뽑을 수
있습니다.

**persona state를 실제로 갱신하는(=`persona.update()`를 부르는) 메서드는 `speak()`
하나뿐입니다.** `get_mood()`/`get_growth_stage()`는 인자를 받지 않는 순수 read
메서드로, `speak()`가 마지막으로 갱신해둔 결과를 읽기만 합니다 — 그래서 둘 다 몇 번을
불러도 state machine에는 아무 영향이 없고, `speak()`를 다시 부르기 전까지는 항상 같은
값을 돌려줍니다.

```python
from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.plant_ai import PlantAI

persona = build_known_persona("상추")
ai = PlantAI(persona)

sensor_values = {
    "temperature": 20, "humidity": 60,
    "soil_temp": 17, "soil_humidity": 55,
    "growth": "sample_easy.jpg",
}

messages = ai.speak(sensor_values)  # persona state를 갱신하는 유일한 지점
mood = ai.get_mood()                # 방금 speak()가 갱신한 결과를 읽기만 함
stage = ai.get_growth_stage()       # 마찬가지
```

- **`get_mood() -> Expression`**: `planta_gochi.persona.mood.Expression`
  (`HAPPY`/`NEUTRAL`/`EXCITED`/`DISTRESSED`) 하나를 돌려줍니다. mood 판단 규칙 자체는
  `planta_gochi/persona/mood.py`에 있습니다(센서별 상태 → mood_key → Expression, 여러
  센서 중 가장 심각한 것 채택, growth는 leaf_count/canopy/leaf_size 중 2개 이상
  나빠지면 무조건 DISTRESSED). `"disease"` 센서는 `confirmed_state`가 `BACTERIAL`/
  `FUNGAL`이면 그냥 "sick"(→ `DISTRESSED`)으로 묶입니다 — 어떤 병인지는 mood에서는
  구분하지 않습니다.
- **`get_growth_stage() -> int | None`**: `"growth"` 센서 결과의 성장 단계(**1~4만**
  나옵니다 — 5단계는 카메라 프레임 이탈로 데이터를 신뢰할 수 없어 아직 기준값이 없습니다.
  자세한 내용은 `planta_gochi/persona/growth_stage.py`). persona에 `"growth"` 센서가
  없거나 마지막 `speak()` 호출에 `"growth"` 값을 안 넣었으면 `None`을 돌려줍니다.

`speak()`를 한 번도 호출하지 않은 상태에서 `get_mood()`/`get_growth_stage()`를 부르면
`RuntimeError`가 납니다 — 아직 읽을 결과가 없다는 뜻입니다.

## 실패 지점 확인하기 (stderr 경고)

LLM 호출이 실패하거나 일부만 성공하면 `LLMConnection`이 **stderr에 경고**를 출력합니다.
반환값에는 영향이 없고, 순수하게 "지금 왜 default_dialog가 나오고 있는지"를 확인하기
위한 용도입니다.

```
[LLMConnection] 요청 실패(model='google/gemma-4-31b-it:free'): <HTTPError 429: 'Too Many Requests'>. 전부 default_dialog로 대체됩니다.
[LLMConnection] 응답에 없는 키 ['growth']는 default_dialog로 대체됩니다(나머지는 LLM 응답 사용).
[LLMConnection] PLANTA_GOCHI_LLM_KEY가 없어 LLM 호출을 건너뜁니다. default_dialog로 대체됩니다.
```

## 환경 설정

`.env`에 다음 값을 넣어두면 `LLMConnection`이 자동으로 읽습니다 (별도 라이브러리 없이
직접 파싱).

```
PLANTA_GOCHI_LLM_KEY=sk-or-...
OPENROUTER_AI_ENDPOINT=https://openrouter.ai/api/v1/chat/completions   # 생략 가능(기본값 동일)
```

키가 없거나 호출이 실패해도 예외를 던지지 않고 항상 `default_dialog` 기반 문장으로
대체되므로, LLM이 아예 연결되지 않은 환경에서도 `PlantAI.speak()`는 안전하게 동작합니다.

## 모델/파라미터 커스터마이즈

```python
from planta_gochi.llm.connection import LLMConnection

ai = PlantAI(persona, llm=LLMConnection(
    model="google/gemma-4-31b-it:free",  # tool-calling(함수 호출) 지원이 확인된 모델이어야 함
    timeout=10,
    temperature=0.8,
    max_tokens=300,
))
```

기본 모델은 OpenRouter에서 tool-calling을 지원하는 무료 모델(`google/gemma-4-31b-it:free`)로
고정되어 있습니다. `openrouter/auto`처럼 라우팅에 맡기는 모델은 tool-calling을 지원하지
않는 모델로 연결될 수 있어 권장하지 않습니다.

실행 예시는 [`main.py`](../main.py)를 참고하세요.

## 로컬 Ollama로 대체하기 (개발 환경 전용)

OpenRouter 없이 로컬 Ollama 서버로 같은 걸 해보고 싶으면 `LLMConnection` 대신
`OllamaConnection`을 꽂으면 됩니다. `ask_batch()` 계약이 완전히 동일해서 `PlantAI`
쪽 코드는 바뀔 게 없습니다.

```python
from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.plant_ai import PlantAI
from planta_gochi.llm.ollama_connection import OllamaConnection

persona = build_known_persona("상추")
ai = PlantAI(persona, llm=OllamaConnection(
    model="qwen2.5:32b",              # 로컬에 pull되어 있고 tools를 지원하는 모델
    host="http://localhost:11434",     # 기본값. 다른 머신/포트면 바꾸면 됨
))

messages = ai.speak({"temperature": 20, "humidity": 60})
```

**설치**: `ollama` 파이썬 패키지는 기본 설치(`pip install plantagochi-models`)에 포함되지
않습니다. 로컬 개발 환경에서만 다음처럼 따로 설치하세요.

```bash
pip install "plantagochi-models[ollama]"
# 또는 uv를 쓴다면
uv sync --extra ollama
```

`ollama` 패키지가 없는 상태에서 `from planta_gochi.llm.ollama_connection import
OllamaConnection`을 import하는 것 자체는 문제없이 되고, 실제로 `OllamaConnection(...)`을
생성하는 시점에만 설치 방법을 안내하는 `ImportError`가 납니다.

**주의할 점**: 로컬 모델의 tool-calling 신뢰도는 모델마다 편차가 큽니다.
- 일부 모델은 `tool_calls` 없이 답변 텍스트에 JSON을 그대로 흉내 내어 쓰는데,
  `OllamaConnection`은 이 경우도 최대한 파싱을 시도합니다.
- 스키마가 복잡해지면(센서가 많아지면) 아예 엉뚱한 인자를 만들어내는 모델도 있습니다 —
  이 경우 `[OllamaConnection] 응답에 없는 키 [...] 는 default_dialog로 대체됩니다`
  경고가 뜨고 해당 센서만 fallback 처리됩니다(다른 백엔드와 동일한 부분 성공 규칙).
- Ollama 서버가 꺼져 있거나 모델이 pull되어 있지 않으면 요청 자체가 실패로 잡혀
  경고가 뜨고 전부 fallback됩니다.
