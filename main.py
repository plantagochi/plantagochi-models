from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.plant_ai import PlantAI

# known_plants/상추.json에 등록된 온도/습도/토양온도/토양습도 linear state machine +
# growth(vision) state machine으로 Persona 생성. "growth" 센서는 GrowthStateMachine이라,
# 숫자가 아니라 이미지(파일 경로 등)를 받아 내부에서 LeafAnalyzer.analyze()를 직접 돌린다.
lettuce = build_known_persona("상추")

# 센서 값 시퀀스를 하나씩 흘려보내며 확인 (같은 persona 인스턴스라 이전 값을 기억함).
# growth는 repo 루트에 있는 sample_easy.jpg/sample_hard.jpg를 번갈아 사용한다.
readings = [
    {"temperature": 20, "humidity": 60, "soil_temp": 17, "soil_humidity": 55, "growth": "sample_easy.jpg"},  # 전부 적정 -> 변화 없음
    {"temperature": 40, "humidity": 60, "soil_temp": 30, "soil_humidity": 55, "growth": "sample_easy.jpg"},  # 온도/토양온도만 급상승 -> 이벤트 발생
    {"temperature": 10, "humidity": 20, "soil_temp": 8,  "soil_humidity": 15, "growth": "sample_easy.jpg"},  # 온도/토양온도 하락 + 습도/토양습도 급락
    {"temperature": 0,  "humidity": 95, "soil_temp": 2,  "soil_humidity": 90, "growth": "sample_easy.jpg"},  # 전부 극단으로
    {"temperature": 20, "humidity": 60, "soil_temp": 17, "soil_humidity": 55, "growth": "sample_easy.jpg"},  # 전부 적정 -> 변화 없음
    {"temperature": 20, "humidity": 60, "soil_temp": 17, "soil_humidity": 55, "growth": "sample_easy.jpg"},  # 전부 적정 -> 변화 없음
    {"temperature": 20, "humidity": 60, "soil_temp": 17, "soil_humidity": 55, "growth": "sample_easy.jpg"},  # 전부 적정 -> 변화 없음
    {"temperature": 20, "humidity": 60, "soil_temp": 17, "soil_humidity": 55, "growth": "sample_easy.jpg"},  # 전부 적정 -> 변화 없음
    {"temperature": 20, "humidity": 60, "soil_temp": 17, "soil_humidity": 55, "growth": "sample_easy.jpg"},  # 전부 적정 -> 변화 없음
    {"temperature": 20, "humidity": 60, "soil_temp": 17, "soil_humidity": 55, "growth": "sample_easy.jpg"},  # 전부 적정 -> 변화 없음
]

#update force
for reading in readings:
    result = lettuce.update(reading)

ai = PlantAI(lettuce)
final_messages = ai.speak(
    {"temperature": 20, "humidity": 60, "soil_temp": 17, "soil_humidity": 55, "growth": "sample_easy.jpg"}
)

print("PlantAI 최종 대사 (여러 개의 글):")
for message in final_messages:
    print(" -", message)
