from planta_gochi.persona.sheets import build_known_persona

# known_plants/상추.json에 등록된 온도/습도 linear state machine으로 Persona 생성
lettuce = build_known_persona("상추")

# 센서 값 시퀀스를 하나씩 흘려보내며 확인 (같은 persona 인스턴스라 이전 값을 기억함)
readings = [
    {"temperature": 20, "humidity": 60},   # 둘 다 적정 -> 변화 없음
    {"temperature": 40, "humidity": 60},   # 온도만 급상승 -> 이벤트 발생
    {"temperature": 10, "humidity": 20},   # 온도 하락 + 습도 급락
    {"temperature": 0, "humidity": 95},    # 둘 다 극단으로
]

for reading in readings:
    # update()는 내부 state machine의 prev_value를 갱신하므로 한 reading당 한 번만 호출한다.
    # (prompts()/default_dialogs()를 같은 reading에 각각 호출하면 두 번째 호출이 "변화 없음"으로 보여
    #  이벤트가 아닌 상태 텍스트로 새버린다.)
    result = lettuce.update(reading)
    print("입력:", reading)
    print("  prompt (LLM용):", {k: v["prompt"] for k, v in result.items()})
    print("  default_dialog (LLM 없을 때 사용자에게 그대로 보여줄 대사):", {k: v["default_dialog"] for k, v in result.items()})
    print()
