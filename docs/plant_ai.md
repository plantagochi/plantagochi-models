# PlantAI (ai.speak / get_mood / get_growth_stage / get_status_effect / get_achievements 사용법)

`PlantAI`는 `Persona`(state machine 기반 prompt/default_dialog)와 `LLMConnection`
(OpenRouter 호출)을 묶어서, 센서 값을 넣으면 최종적으로 사용자에게 보여줄 짧은 대사
목록, mood(표정), 성장 단계까지 만들어주는 클래스입니다.

종 선택(`"상추"` / `"상추_trend"`)과 여러 persona가 analyzer 모델을 공유하는 방법은
[`persona_builder.md`](persona_builder.md)에 있습니다.

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

`"growth"`/`"disease"` 센서 값은 파일 경로에 국한되지 않습니다 — 가공 없이 그대로
`LeafAnalyzer`/`DiseaseAnalyzer`의 `analyze()`에 전달되므로, [LeafAnalyzer가 지원하는
모든 입력 형식](leaf_analyzer.md)(로컬 경로, http(s) URL, bytes/bytearray,
`io.BytesIO` 같은 file-like 객체, PIL Image, numpy array)을 그대로 쓸 수 있습니다.

### `"image"` — 이미지 센서 전부에 같은 사진 한 장 넣기

카메라 한 대로 `growth`/`disease`를 동시에 판단하는 게 보통이라, 매번 같은 사진을
두 번 적어 넣지 않아도 되도록 `"image"`라는 특별한 키를 지원합니다. `sensor_values`에
`"image"`를 넣으면, persona가 가진 "이미지를 입력으로 받는" 센서(`accepts_image`가
`True`인 엔진 — 지금은 `growth`/`disease`) 전부에 그 값이 동시에 들어갑니다.

```python
# 아래 두 줄은 완전히 같은 결과를 냅니다.
ai.speak({"image": "sample_easy.jpg"})
ai.speak({"growth": "sample_easy.jpg", "disease": "sample_easy.jpg"})
```

센서 이름을 명시적으로 같이 넘기면 그 값이 `"image"`보다 우선합니다(그 센서만 다른
사진을 쓰고 싶을 때):

```python
# growth는 sample_hard.jpg, disease는 image로 넘긴 sample_easy.jpg를 쓴다.
ai.speak({"image": "sample_easy.jpg", "growth": "sample_hard.jpg"})
```

기존처럼 `"growth"`/`"disease"`를 직접 쓰는 방식은 전혀 바뀌지 않았습니다 —
`"image"`는 순전히 추가된 편의 기능입니다. 이미지를 받는 센서가 하나도 없는
persona라면 `"image"` 키는 조용히 무시됩니다.

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

## get_mood() / get_growth_stage() / get_status_effect() — 표정/성장 단계/상태이상 뽑기

`speak()`는 LLM을 호출하는 무거운 메서드입니다. UI 아바타 표정이나 성장 단계, 상태이상
표시처럼 자주 갱신해야 하는 값은 LLM 없이 `get_mood()`/`get_growth_stage()`/
`get_status_effect()`로 바로 뽑을 수 있습니다.

**persona state를 실제로 갱신하는(=`persona.update()`를 부르는) 메서드는 `speak()`
하나뿐입니다.** `get_mood()`/`get_growth_stage()`/`get_status_effect()`는 인자를 받지
않는 순수 read 메서드로, `speak()`가 마지막으로 갱신해둔 결과를 읽기만 합니다 — 그래서
셋 다 몇 번을 불러도 state machine에는 아무 영향이 없고, `speak()`를 다시 부르기 전까지는 항상 같은
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
effects = ai.get_status_effect()    # 마찬가지
```

- **`get_mood() -> Expression`**: `planta_gochi.persona.mood.Expression`
  (`HAPPY`/`NEUTRAL`/`EXCITED`/`DISTRESSED`) 하나를 돌려줍니다. mood 판단 규칙 자체는
  `planta_gochi/persona/mood.py`에 있습니다(센서별 상태 → mood_key → Expression, 여러
  센서 중 가장 심각한 것 채택, growth는 leaf_count/canopy/leaf_size 중 2개 이상
  나빠지면 무조건 DISTRESSED). `"disease"` 센서는 `confirmed_state`가 `BACTERIAL`/
  `FUNGAL`이면 그냥 "sick"(→ `DISTRESSED`)으로 묶입니다 — 어떤 병인지는 mood에서는
  구분하지 않습니다.
- **`get_growth_stage() -> int`**: `"growth"` 센서 결과의 성장 단계(**1~5**, 기준값은
  `planta_gochi/persona/growth_stage.py`)를 돌려줍니다. `leaf_count == 0`(잎이 하나도
  안 보임)이거나, persona에 `"growth"` 센서가 없거나, 마지막 `speak()` 호출에
  `"growth"` 값을 안 넣었으면 `0`을 돌려줍니다.
- **`get_status_effect() -> list[StatusEffect]`**: 게임의 "상태이상"처럼, 지금
  활성화된 상태이상 문구를 담은 리스트를 돌려줍니다(판단 규칙은
  `planta_gochi/persona/status_effect.py`). `mood`와 마찬가지로 항상 **현재 state**
  기준이고, "완만한" 상태가 아니라 **critical(극단) 구간에서만** 발동합니다:

  | 원인 | 상태이상 |
  |---|---|
  | disease = `BACTERIAL` | `"박테리아"` |
  | disease = `FUNGAL` | `"곰팡이"` |
  | temperature 극단 고온 / 저온 | `"타는듯한온도"` / `"어는듯한온도"` |
  | humidity 극단 고습 / 저습 | `"정글같은습도"` / `"사막같은습도"` |
  | soil_temp 극단 고온 / 저온 | `"타들어가는뿌리"` / `"얼어붙은뿌리"` |
  | soil_humidity 극단 고습 / 저습 | `"썩어가는뿌리"` / `"메마른뿌리"` |

  동시에 여러 개가 활성화되면 전부 리스트에 담겨 나옵니다(위 표 순서 고정). 아무
  상태이상도 없으면 빈 리스트입니다. `StatusEffect`도 `Expression`처럼 `(str, Enum)`이라
  그 자체로 문자열처럼 쓸 수 있습니다(`.value`로 순수 문자열도 꺼낼 수 있음).

**관측 안 됨과 해소됨을 구분하고 싶다면 `extract_every_status_effects()`를 쓰세요.**
`get_status_effect()`/`extract_status_effects()`는 활성인 것만 돌려주므로, 이번 결과에
없는 센서의 상태이상은 목록에서 빠질 뿐 "해소됐다"는 뜻이 아닙니다. 이 함수는 모든
`StatusEffect`를 `StatusEffectState` 3값으로 돌려줍니다(`planta_gochi.persona.status_effect`).

- `ACTIVE`: 이번 결과에서 원인 센서/질병이 그 상태이상의 state에 있다.
- `CLEARED`: 원인 센서/질병을 이번에 관측했고, 그 state가 아니다(해소됨).
- `UNKNOWN`: 이번 결과에 원인 센서/질병이 없다. 해소된 게 아니라 모르는 것이다.

```python
from planta_gochi.persona.status_effect import StatusEffect, StatusEffectState, extract_every_status_effects

result = persona.update({"temperature": 40})   # humidity 등은 관측하지 않음
states = extract_every_status_effects(result)
states[StatusEffect.SCORCHING_TEMPERATURE]      # StatusEffectState.ACTIVE
states[StatusEffect.FREEZING_TEMPERATURE]       # StatusEffectState.CLEARED
states[StatusEffect.JUNGLE_HUMIDITY]            # StatusEffectState.UNKNOWN
```

입력은 `persona.update()`가 돌려준 dict입니다. `PlantAI`에는 아직 이 3값 조회 메서드가
없습니다 — `PlantAI`를 통해 쓰려면 마지막 결과를 읽는 메서드를 따로 추가해야 합니다.

`speak()`를 한 번도 호출하지 않은 상태에서 `get_mood()`/`get_growth_stage()`/
`get_status_effect()`를 부르면 `RuntimeError`가 납니다 — 아직 읽을 결과가 없다는
뜻입니다. (`get_achievements()`는 예외입니다 — 아래 참고.)

## get_achievements() — 게임적 "업적" 뽑기

지금까지 이 `Persona` 인스턴스가 달성한 achievement(`planta_gochi.persona.
supported_achivements.SupportedAchivements`) 누적 목록을 돌려줍니다.

```python
ai.speak({"temperature": 40})  # 극단 고온
ai.speak({"temperature": 20})  # 정상 범위로 복구

print(ai.get_achievements())
# [<SupportedAchivements.SUDDEN_ENVIRONMENT_SHIFT: '식물이 죽는다고!!'>]
```

**다른 `get_*()`와 다른 점 — "마지막 한 틱"이 아니라 계속 쌓이는 값입니다.**
`get_mood()`/`get_growth_stage()`/`get_status_effect()`는 방금 `speak()`가 갱신한
결과 하나만 보고 매번 새로 계산하지만, achievement는 "지금까지 한 번이라도 조건을
만족한 적 있는가"를 기억해야 합니다. 그래서:

- `persona.update(sensor_values)`(따라서 `speak()`)를 부를 때마다 그 안에서
  `AchievementTracker`가 결과를 관찰하고, 새로 조건을 만족한 게 있으면 누적 집합에
  더합니다. 이미 달성한 건 조건이 더 이상 사실이 아니게 되어도 사라지지 않습니다.
- `persona.update()`는 achievement를 **관찰만** 하고 반환값에는 담지 않습니다. 읽는 방법은
  `get_achievements()` 하나뿐입니다.
- `PlantAI.get_achievements()`는 `persona.get_achievements()`를 그대로 위임해서
  읽기만 합니다. 그래서 다른 `get_*()`와 달리 **`speak()`를 한 번도 안 불렀어도
  `RuntimeError` 없이 빈 리스트를 돌려줍니다** — LLM 결과(`_last_result`)가 아니라
  `Persona`가 스스로 들고 있는 누적 상태를 읽는 것이기 때문입니다. `persona.update()`를
  (`speak()`를 거치지 않고) 직접 불러도 똑같이 누적됩니다.

**휘발성(volatile)입니다.** `Persona.get_state()`/`load_state()`로는 achievement가
전혀 저장/복원되지 않습니다 — state를 저장했다가 새 프로세스에서 다시 불러오면
achievement는 전부 초기화됩니다. 센서 값을 이어가기 위한 "진짜 상태"와는 다른 층위의
게임적/UI 개념이라 의도적으로 분리했습니다.

**지금 지원하는 achievement 9종**(`SupportedAchivements` 정의 순서, 판단 로직은
`planta_gochi/persona/achievement.py` 참고):

| 이름 | 조건 |
|---|---|
| 살려야한다 | 서로 다른 상태이상 3종을 각각 정상으로 복구 |
| 트러플 바이옴? 아직 하드모드도 아니야! | 곰팡이성 질병 완치 |
| 난 나보다 약한 세균의 명령따위 듣지 않는다. | 세균성 질병 완치 |
| 아무에게도 말하지 마라! | 성장 단계가 한 틱 사이에 2단계 이상 점프 ⚠️ |
| 이제 가망이 없어 | 성장 5단계 도달 |
| 식물이 죽는다고!! | 환경 센서(temperature/humidity/soil_temp/soil_humidity) 중 하나가 한 틱 사이에 2밴드 이상 점프 |
| Too much water | 공기 습도가 `VALUE_CRITICAL_HIGH`(상추 기준 90% 이상)에 도달 |
| 올빼미 농부 | 자정~새벽 4시 사이에 soil_humidity가 직전 관측값보다 상승(물주기로 추론) |
| Touch grass | growth의 캐노피 비율(`raw_metrics.canopy_ratio`)이 50% 이상 |

**⚠️ "아무에게도 말하지 마라!"의 알려진 한계**: `stage`는 canopy/leaf_size와 달리
`TrendStateMachine`으로 스무딩되지 않고, 그 프레임의 raw 지표로 `classify_growth_stage()`를
매번 새로 돌린 값입니다(`GrowthStateMachine` 참고). 그래서 실제로 며칠에 걸쳐 급성장한
경우뿐 아니라, 연속된 두 사진의 세그멘테이션 노이즈(각도/조명 차이 등)만으로 stage가
한 틱 만에 2단계 이상 튀면서 이 achievement가 터질 수 있습니다. 지금은 그대로 두고
한계로만 문서화하기로 확인했습니다(2026-09-29 대화) — 더 엄격하게 하려면 `stage`
자체에도 `DiscreteCountMachine`류의 debounce(`confirm_streak`)를 추가해 "확정된
stage"끼리 비교하는 방법이 있습니다.

**"위기 대응 전문가"(상태이상 알림 후 30분 이내 정상 복구)는 아직 없습니다.** 다른
9개와 달리 "이번 틱에 이벤트 한 번"이 아니라 경과 시간을 추적해야 해서 성격이 다르고,
지금 단계에서는 보류하기로 했습니다.

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
