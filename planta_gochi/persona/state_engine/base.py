from abc import ABC, abstractmethod
from typing import Any


class StateEngine(ABC):
    """
    Persona가 다루는 모든 state machine(선형/불리언 등)이 따라야 하는 공통 인터페이스.
    Persona는 이 인터페이스에만 의존하므로, 새로운 종류의 state machine을 추가해도
    Persona나 builder 쪽 코드를 건드릴 필요가 없다.
    """

    #: detect_event(curr_value)가 이미지(sensory._image_io.ImageInput 형식)를 받는
    #: 엔진이면 True로 오버라이드한다 (예: GrowthStateMachine, DiseaseStateMachine).
    #: Persona.update()가 "image" 별칭 하나를 여러 이미지 기반 센서에 동시에 뿌려줄 때
    #: 이 값으로 대상 엔진을 찾는다 — Persona가 구체적인 엔진 클래스를 몰라도 되도록
    #: 하기 위한 마커라, 클래스 이름으로 분기하지 않는다.
    accepts_image: bool = False

    @abstractmethod
    def detect_event(self, curr_value: Any) -> dict:
        """
        새 센서 값을 반영하고, 그에 대한 prompt 정보를 담은 dict를 반환한다.
        반환 dict는 최소한 "prompt" key를 포함해야 하며, "prompt"/"default_dialog"는
        하나 이상의 메시지를 담은 list[str]이어야 한다(단일 신호 엔진은 1개짜리 list,
        GrowthStateMachine처럼 여러 하위 신호를 합성하는 엔진은 여러 개짜리 list).
        이렇게 통일해두면 Persona/PlantAI 쪽에서 엔진 종류와 상관없이 항상 같은
        모양으로 다룰 수 있다.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict:
        """
        다음 update를 이어가는 데 필요한 최소한의 순수 상태(state)를 dict로 돌려준다.
        thresholds/prompts 같은 정적 설정은 포함하지 않는다 — 그건 JSON(known_plant_builder)이
        갖고 있고 로드 시점에 다시 만들어지므로 여기서 또 담을 필요가 없다.
        PostgreSQL 등 외부 저장소에 그대로 넣을 수 있도록, enum은 반드시 .value(int)로
        변환해서 담아야 하고 값은 JSON 호환 타입(int/float/str/bool/None/list/dict)이어야 한다.
        """
        raise NotImplementedError

    @abstractmethod
    def load_state(self, state: dict) -> None:
        """get_state()가 만든 dict를 받아 내부 상태를 그 시점으로 복원한다."""
        raise NotImplementedError
