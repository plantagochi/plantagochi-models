from abc import ABC, abstractmethod
from typing import Any


class StateEngine(ABC):
    """
    Persona가 다루는 모든 state machine(선형/불리언 등)이 따라야 하는 공통 인터페이스.
    Persona는 이 인터페이스에만 의존하므로, 새로운 종류의 state machine을 추가해도
    Persona나 builder 쪽 코드를 건드릴 필요가 없다.
    """

    @abstractmethod
    def detect_event(self, curr_value: Any) -> dict:
        """
        새 센서 값을 반영하고, 그에 대한 prompt 정보를 담은 dict를 반환한다.
        반환 dict는 최소한 "prompt" key를 포함해야 한다.
        """
        raise NotImplementedError
