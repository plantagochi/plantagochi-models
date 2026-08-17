from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.plant_ai import PlantAI

# known_plants/상추.json에 등록된 온도/습도 linear state machine + growth(vision)
# state machine으로 Persona 생성. "growth" 센서는 GrowthStateMachine이라, 숫자가 아니라
# 이미지(파일 경로 등)를 받아 내부에서 LeafAnalyzer.analyze()를 직접 돌린다.
lettuce = build_known_persona("상추")

# 센서 값 시퀀스를 하나씩 흘려보내며 확인 (같은 persona 인스턴스라 이전 값을 기억함).
# growth는 repo 루트에 있는 sample_easy.jpg/sample_hard.jpg를 번갈아 사용한다.
readings = [
    {"temperature": 20, "humidity": 60, "growth": "sample_easy.jpg"},   # 온습도 적정 -> 변화 없음
    {"temperature": 40, "humidity": 60, "growth": "sample_hard.jpg"},   # 온도만 급상승 -> 이벤트 발생
    {"temperature": 10, "humidity": 20, "growth": "sample_easy.jpg"},   # 온도 하락 + 습도 급락
    {"temperature": 0,  "humidity": 95, "growth": "sample_hard.jpg"},    # 온습도 둘 다 극단으로
    {"temperature": 20, "humidity": 60, "growth": "sample_easy.jpg"},   # 온습도 적정 -> 변화 없음
    {"temperature": 20, "humidity": 60, "growth": "sample_easy.jpg"},   # 온습도 적정 -> 변화 없음
    {"temperature": 20, "humidity": 60, "growth": "sample_easy.jpg"},   # 온습도 적정 -> 변화 없음
    {"temperature": 20, "humidity": 60, "growth": "sample_easy.jpg"},   # 온습도 적정 -> 변화 없음
    {"temperature": 20, "humidity": 60, "growth": "sample_easy.jpg"},   # 온습도 적정 -> 변화 없음
    {"temperature": 20, "humidity": 60, "growth": "sample_easy.jpg"},   # 온습도 적정 -> 변화 없음
]

for reading in readings:
    # update()는 내부 state machine의 prev_value/history를 갱신하므로 한 reading당 한 번만
    # 호출한다. (prompts()/default_dialogs()를 같은 reading에 각각 호출하면 두 번째 호출이
    # "변화 없음"으로 보여 이벤트가 아닌 상태 텍스트로 새버린다.)
    result = lettuce.update(reading)
    print("입력:", reading)
    print("  prompt (LLM용):", {k: v["prompt"] for k, v in result.items()})
    print("  default_dialog (LLM 없을 때 사용자에게 그대로 보여줄 대사):", {k: v["default_dialog"] for k, v in result.items()})
    print()

growth_detail = result["growth"]
print("growth 세부 신호 (마지막 reading 기준):")
print("  leaf_count:", growth_detail["leaf_count"]["count"])
print("  canopy_trend:", growth_detail["canopy_trend"]["state"])
print("  leaf_size_trend:", growth_detail["leaf_size_trend"]["state"])
print()

# --- TrendStateMachine의 INSUFFICIENT_DATA 억제 확인 ---
# canopy_trend/leaf_size_trend는 smoothed_history가 3개 미만이면 "아직 잘 모르겠어요"류의
# insufficient_data 상태인데, 이 상태에서 이벤트 없이 매 프레임 그 얘기를 반복하는 건
# 의미가 없으므로 prompt/default_dialog를 비워서 emit하지 않는다(trend_state_machine.py).
# 대신 데이터가 3개 이상 쌓여 실제로 추세 판단이 가능해지면(=상태가 바뀌는 이벤트가
# 발생하면) 그 순간부터는 정상적으로 prompt가 채워져야 한다. 아래에서 그 전환을 직접 확인한다.
print("=== growth trend warm-up 테스트 (insufficient_data 동안 prompt 억제 확인) ===")
warmup_persona = build_known_persona("상추")
for step in range(1, 5):
    growth = warmup_persona.update({"growth": "sample_easy.jpg"})["growth"]
    canopy_prompt = growth["canopy_trend"]["prompt"]
    leaf_size_prompt = growth["leaf_size_trend"]["prompt"]
    print(
        f"  step {step}: canopy_trend={growth['canopy_trend']['state'].name} "
        f"prompt={canopy_prompt} | leaf_size_trend={growth['leaf_size_trend']['state'].name} "
        f"prompt={leaf_size_prompt}"
    )
    if step <= 2:
        assert canopy_prompt == [] and leaf_size_prompt == [], (
            "insufficient_data 구간(처음 2번)에서는 prompt가 비어 있어야 합니다"
        )
    else:
        assert canopy_prompt != [] and leaf_size_prompt != [], (
            "데이터가 3개 이상 쌓이면(3번째 호출부터) prompt가 실제로 emit되어야 합니다"
        )

# PlantAI: Persona가 만든 센서별 prompt(list[str])를 한데 모아 OpenRouter에 tool-calling으로
# "한 번만" 요청해서 센서별 최종 대사를 한 번에 받아온다(센서마다 따로 요청하지 않음).
# .env의 PLANTA_GOCHI_LLM_KEY가 없거나 호출이 실패하면 전체를 default_dialog로 대체한다.
# 반환값은 화면에 표시할 여러 개의 글(list[str])이다.
ai = PlantAI(build_known_persona("상추"))
final_messages = ai.speak({"temperature": 3, "humidity": 90, "growth": "sample_easy.jpg"})
print("PlantAI 최종 대사 (여러 개의 글):")
for message in final_messages:
    print(" -", message)
