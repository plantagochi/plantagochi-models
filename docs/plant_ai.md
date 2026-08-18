# PlantAI (ai.speak 사용법)

`PlantAI`는 `Persona`(state machine 기반 prompt/default_dialog)와 `LLMConnection`
(OpenRouter 호출)을 묶어서, 센서 값을 넣으면 최종적으로 사용자에게 보여줄 짧은 대사
목록을 만들어주는 클래스입니다.

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
