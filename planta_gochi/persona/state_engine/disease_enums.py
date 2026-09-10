from enum import Enum

"""
linear_state_enums.py/growth_enums.py와 같은 방식(state 번호를 prev*10+curr로
인코딩해 전환 이벤트를 표현)을 disease dimension에도 그대로 적용한 enum들.

LinearState와 달리 "구간"이 아니라 DiseaseAnalyzer가 내놓는 3개 분류 라벨
(bacterial/fungal/healthy) 그 자체가 state다 — 그래서 순서에 우열은 없다.
"""


class DiseaseState(Enum):
    HEALTHY = 1
    BACTERIAL = 2
    FUNGAL = 3


class DiseaseEvents(Enum):
    NO_EVENTS = 0

    HEALTHY_TO_BACTERIAL = 12
    HEALTHY_TO_FUNGAL = 13

    BACTERIAL_TO_HEALTHY = 21
    BACTERIAL_TO_FUNGAL = 23

    FUNGAL_TO_HEALTHY = 31
    FUNGAL_TO_BACTERIAL = 32
