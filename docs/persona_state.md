# Persona 상태 저장/복원 (save / load)

`Persona`는 센서 이름별로 `StateEngine`(`LinearStateMachine`/`TrendStateMachine`/
`DiscreteCountMachine`/`GrowthStateMachine`/`DiseaseStateMachine`)을 들고 있고,
`update()`를 부를 때마다
각 엔진 내부의 값(이전 값, 이동평균, 연속 관측 횟수 등)이 바뀝니다. 프로세스가
재시작되거나 다른 서버가 이어받아 처리해야 할 때 이 값을 잃어버리면 안 되므로,
`Persona.get_state()` / `Persona.load_state()`로 즉시 저장·복원할 수 있게 되어 있습니다.

## 기본 사용법

```python
from planta_gochi.persona.sheets import build_known_persona

persona = build_known_persona("상추")
persona.update({"temperature": 20, "humidity": 60, "growth": "sample_easy.jpg"})

# 저장
state = persona.get_state()   # {"temperature": {...}, "humidity": {...}, "growth": {...}}

# ... 나중에, 혹은 다른 프로세스에서 ...

restored = build_known_persona("상추")   # 같은 종의 Persona를 JSON으로 다시 조립
restored.load_state(state)               # 저장해둔 state를 그대로 얹는다
restored.update({"temperature": 21, "humidity": 58, "growth": "sample_easy.jpg"})
```

`load_state()`는 **Persona/엔진이 이미 만들어져 있다고 가정**하고 그 위에 동적 값만
덮어씁니다. thresholds나 prompt 문구 같은 정적 설정은 `get_state()`에 포함되지
않습니다 — 그런 설정은 `known_plant_builder`가 종별 JSON(예: `상추.json`)에서 다시
만들어주므로 굳이 같이 저장할 필요가 없습니다. 즉 저장/복원 순서는 항상:

1. `build_known_persona(...)` 등으로 Persona를 (다시) 조립
2. 저장해둔 state가 있으면 `load_state(state)`로 얹기

## `get_state()`가 돌려주는 값

`{센서 이름: 그 센서 엔진의 state}` 형태의 순수 dict입니다. 모든 값은
int/float/str/bool/None/list/dict로만 이루어져 있어 `json.dumps()`에 바로 넣을 수
있고, PostgreSQL의 JSONB 컬럼 등에 그대로 저장/업데이트하기 좋습니다. **enum은 절대
그대로 담기지 않고 항상 `.value`(int)로 변환되어 들어갑니다.**

엔진별로 실제 들어있는 키는 다음과 같습니다.

| 엔진 | state 키 | 비고 |
|---|---|---|
| `LinearStateMachine` (온도/습도/토양온도/토양습도) | `prev_value` | 원본 센서 값 그대로 (숫자) |
| `DiscreteCountMachine` (leaf_count) | `confirmed_count`, `candidate_count`, `streak` | 전부 스칼라 int |
| `TrendStateMachine` (canopy/leaf_size 추세) | `ema`, `prev_state`(int), `raw_history`, `smoothed_history` | ⚠️ 아래 참고 |
| `GrowthStateMachine` (growth) | `count_machine`, `canopy_machine`, `leaf_size_machine` | 위 두 엔진의 state를 그대로 묶은 것 |
| `DiseaseStateMachine` (disease) | `confirmed_state`(int), `candidate_state`(int), `streak` | 전부 스칼라. `DiscreteCountMachine`과 같은 debounce 구조라 모양도 같음 |

### ⚠️ 배열형 필드: `raw_history` / `smoothed_history`

`TrendStateMachine`의 state에는 `raw_history`, `smoothed_history`라는 **배열(list[float])**
필드가 들어 있습니다. 추세(성장 중/정체/감소) 판단이 "최근 N개 값의 회귀 기울기"로
이루어지기 때문에, 다음 `update()`부터 이어서 계산하려면 과거 값들이 그대로 있어야
합니다 — 스칼라 하나로 요약해서 저장할 수 없습니다.

- `raw_history`: 최대 `window * 3`개 (기본 window=7 → 최대 21개)
- `smoothed_history`: 최대 `window`개 (기본 7개)

`growth` 센서를 쓰는 경우, state 예시는 이런 모양입니다:

```jsonc
{
  "growth": {
    "canopy_machine": {
      "ema": 0.3955,
      "prev_state": 4,
      "raw_history": [0.334922, 0.658008, 0.334922],
      "smoothed_history": [0.334922, 0.4157, 0.3955]
    },
    "leaf_size_machine": { "...": "canopy_machine과 같은 모양" },
    "count_machine": { "confirmed_count": 15, "candidate_count": 15, "streak": 1 }
  }
}
```

JSONB 컬럼에 배열째로 넣든, 별도 컬럼/테이블로 분리하든은 이 라이브러리의 범위 밖이라
직접 결정하시면 됩니다.

## pickle로 저장/복원 (테스트/로컬용 예시)

```python
import pickle
from planta_gochi.persona.sheets import build_known_persona

persona = build_known_persona("상추")
persona.update({"temperature": 20, "humidity": 60})

with open("persona_state.pkl", "wb") as f:
    pickle.dump(persona.get_state(), f)

# ... 나중에 ...

with open("persona_state.pkl", "rb") as f:
    state = pickle.load(f)

restored = build_known_persona("상추")
restored.load_state(state)
```

`get_state()`가 만드는 값은 전부 JSON-safe한 순수 자료형이라 pickle뿐 아니라
`json.dump`/DB 저장 어디에도 그대로 쓸 수 있습니다. 실제 동작 검증은
[`test_persona_state.py`](../test_persona_state.py)에 있습니다 — 저장 전/후 값이
같은지뿐 아니라, 새 `Persona` 인스턴스에 state를 복원해 이어서 `update()`했을 때
계속 라이브로 돌린 것과 똑같은 `prompt`/`raw_event`가 나오는지까지 확인합니다.
