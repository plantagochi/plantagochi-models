from enum import Enum

"""
기본적으로 이 State들은 서로 독립적이며 영향을 주고 받을 수 없게 설계.
실제로 절대적인 현재 값 / 추세 / 그런지 아닌자
"""

class LinearState(Enum):
    VALUE_CRITICAL_LOW = 1
    VALUE_LOW = 2
    VALUE_MID = 3
    VALUE_HIGH = 4
    VALUE_CRITICAL_HIGH = 5

class LinearEvents(Enum):
    NO_EVENTS = 0

    C_LOW_TO_LOW = 12
    C_LOW_TO_MID = 13
    C_LOW_TO_HIGH = 14
    C_LOW_TO_C_HIGH = 15

    LOW_TO_C_LOW = 21
    LOW_TO_MID = 23
    LOW_TO_HIGH = 24
    LOW_TO_C_HIGH = 25

    MID_TO_C_LOW = 31
    MID_TO_LOW = 32
    MID_TO_HIGH = 34
    MID_TO_C_HIGH = 35

    HIGH_TO_C_LOW = 41
    HIGH_TO_LOW = 42
    HIGH_TO_MID = 43
    HIGH_TO_C_HIGH = 45

    C_HIGH_TO_C_LOW = 51
    C_HIGH_TO_LOW = 52
    C_HIGH_TO_MID = 53
    C_HIGH_TO_HIGH = 54


