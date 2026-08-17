from enum import Enum

"""
linear_state_enums.py와 같은 방식(state 번호를 prev*10+curr로 인코딩해 전환 이벤트를
표현)을 growth tracking dimension에도 그대로 적용한 enum들.
"""


class TrendState(Enum):
    INSUFFICIENT_DATA = 1
    DECLINING = 2
    STAGNANT = 3
    GROWING = 4


class TrendEvents(Enum):
    NO_EVENTS = 0

    INSUFFICIENT_TO_DECLINING = 12
    INSUFFICIENT_TO_STAGNANT = 13
    INSUFFICIENT_TO_GROWING = 14

    DECLINING_TO_INSUFFICIENT = 21
    DECLINING_TO_STAGNANT = 23
    DECLINING_TO_GROWING = 24

    STAGNANT_TO_INSUFFICIENT = 31
    STAGNANT_TO_DECLINING = 32
    STAGNANT_TO_GROWING = 34

    GROWING_TO_INSUFFICIENT = 41
    GROWING_TO_DECLINING = 42
    GROWING_TO_STAGNANT = 43


class CountState(Enum):
    # leaf_count는 물리적 임계값 구간이 없는 임의의 정수라 LinearState처럼 여러 구간으로
    # 나누지 않는다. "현재 확정된 값을 유지 중"이라는 단일 상태만 존재.
    STABLE = 1


class CountEvents(Enum):
    NO_EVENTS = 0
    INCREASED = 1
    DECREASED = 2
