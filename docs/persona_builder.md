# 종별 Persona 조립 (known_plant_builder)

종(species)별 `Persona`는 `planta_gochi/persona/sheets/known_plants/*.json`에 있는 spec을 읽어서
`planta_gochi.persona.sheets` 모듈의 함수로 조립합니다. 문구(state_prompts / event_prompts /
default_dialog)와 threshold 같은 정적 설정은 전부 이 JSON에 있고, 코드에는 들어 있지 않습니다.

## 등록된 종 파일

| 로드 키 (`build_known_persona`의 인자) | 파일 | `name` 필드 | growth의 추세 머신 |
|---|---|---|---|
| `"상추"` | `known_plants/상추.json` | `"상추"` | 없음 (leaf_count만) |
| `"상추_trend"` | `known_plants/상추_trend.json` | `"상추_trend"` | 있음 (canopy / leaf_size 추세) |

- **로드 키는 파일 이름**입니다. `build_known_persona("상추_trend")`는 `상추_trend.json`을 읽습니다.
- **`name` 필드**는 `Persona.name`이 되고, `PlantAI.speak()`가 LLM 요청에 넘기는 종 이름입니다.
  두 파일이 같은 종이라도 `name`이 다르면 LLM 쪽에서도 구분됩니다.
- 두 파일은 `name`과 `growth`의 `trend_machine` 섹션 외에는 동일합니다.

```python
from planta_gochi.persona.sheets import build_known_persona

plain = build_known_persona("상추")          # leaf_count만 본다
trend = build_known_persona("상추_trend")    # leaf_count + canopy/leaf_size 추세를 본다
```

추세 머신을 쓰면 `persona.engines["growth"].canopy_machine`이 있고, 없으면 `None`입니다.
추세가 없는 growth에서는 `DISTRESSED` 규칙(부정 신호 2개 이상)이 growth만으로는 발동하지
않는다는 점도 [`plant_ai.md`](plant_ai.md)의 mood 설명과 같이 기억해두세요.

## 조립 함수

| 함수 | 입력 | 용도 |
|---|---|---|
| `build_known_persona(name, leaf_analyzer=None, disease_analyzer=None)` | 종 로드 키 | 가장 흔히 쓰는 진입점 |
| `build_persona_from_json(path, leaf_analyzer=None, disease_analyzer=None)` | JSON 파일 경로 | `known_plants/` 밖의 spec 파일 |
| `build_persona_from_spec(spec, leaf_analyzer=None, disease_analyzer=None)` | `{"name": ..., "sensors": {...}}` dict | 코드에서 spec을 직접 만들 때 |
| `build_engine(sensor_spec, leaf_analyzer=None, disease_analyzer=None)` | 센서 하나의 spec | 엔진 하나만 필요할 때 |

## analyzer 공유 (모델 재로딩 방지)

growth 엔진은 `LeafAnalyzer`(YOLO 세그멘테이션), disease 엔진은 `DiseaseAnalyzer`(분류기)를 씁니다.
이 둘은 생성할 때 모델 파일을 로드하므로, persona를 여러 개 만들 때마다 새로 만들면 비쌉니다.
`leaf_analyzer` / `disease_analyzer` 인자로 미리 만든 인스턴스를 넘기면 그 인스턴스를 그대로 씁니다.

```python
from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.sensory import DiseaseAnalyzer, LeafAnalyzer

leaf = LeafAnalyzer()
disease = DiseaseAnalyzer()

plain = build_known_persona("상추", leaf_analyzer=leaf, disease_analyzer=disease)
trend = build_known_persona("상추_trend", leaf_analyzer=leaf, disease_analyzer=disease)

assert plain.engines["growth"].leaf_analyzer is trend.engines["growth"].leaf_analyzer
```

- 인자를 넘기지 않으면 기존처럼 spec의 `"leaf_analyzer"` / `"disease_analyzer"` 설정(예: `{"backend": "onnx"}`)으로 엔진마다 새로 만듭니다.
- 인자를 넘기면 spec의 해당 설정은 **무시**되고, 넘긴 인스턴스가 쓰입니다. 백엔드를 바꾸고 싶다면 인스턴스를 만들 때 지정하세요.
- 넘긴 인스턴스는 persona 사이에서 공유되므로, 분석기 자체에 프레임 간 상태가 생기지 않는다는 전제입니다(`LeafAnalyzer`/`DiseaseAnalyzer`는 모델 wrapper라 상태가 없습니다).

## spec 구조

```jsonc
{
  "name": "상추",
  "sensors": {
    "temperature": {"type": "linear", "thresholds": [5, 15, 25, 35],
                    "state_prompts": {...}, "event_prompts": {...}, "default_dialog": {...}},
    "growth": {
      "type": "growth",
      "leaf_analyzer": {"backend": "onnx"},          // 선택, 주입 인자가 없을 때만 쓰인다
      "count_machine": {"confirm_streak": 3, "state_prompts": {...}, "event_prompts": {...}, "default_dialog": {...}},
      "trend_machine": {"window": 7, "mad_k": 3.5, "growth_eps": 0.001, "decline_eps": 0.001,  // 선택
                        "state_prompts": {...}, "event_prompts": {...}, "default_dialog": {...}}
    },
    "disease": {"type": "disease", "disease_analyzer": {"backend": "onnx"},  // 선택
                "confirm_streak": 5, "state_prompts": {...}, "event_prompts": {...}, "default_dialog": {...}}
  }
}
```

- `"type"`을 생략하면 `"linear"`로 봅니다. 등록되지 않은 타입은 조립 시점에 `ValueError`가 납니다.
- `default_dialog`는 `state_prompts` / `event_prompts`에 있는 모든 state·event를 빠짐없이 커버해야 합니다.
  빠지면 조립 시점에 `ValueError`가 나므로, 실행 중에 "대사가 없다"는 상황은 생기지 않습니다.
- `"trend_machine"`이 없으면 growth는 추세 머신 없이 조립됩니다. 결과에서는 `canopy_trend` / `leaf_size_trend`
  키만 빠지고, `leaf_count`와 `stage` / `raw_metrics`는 그대로 나옵니다.
- 새 엔진 타입은 `@register_engine_builder("이름")`으로 빌더 함수를 등록하면 됩니다. Persona나 PlantAI는
  수정할 필요가 없습니다.

## 관련 문서

- 조립한 persona의 상태를 저장/복원하는 방법: [`persona_state.md`](persona_state.md)
- `PlantAI`가 persona를 감싸서 대사/mood/성장 단계를 만드는 방법: [`plant_ai.md`](plant_ai.md)
