from __future__ import annotations

from typing import Any, Optional

from planta_gochi.persona.state_engine.base import StateEngine
from planta_gochi.persona.state_engine.disease_enums import DiseaseEvents, DiseaseState
from planta_gochi.sensory import DiseaseAnalyzer


class DiseaseStateMachine(StateEngine):
    """
    DiseaseAnalyzer(YOLOv8-cls 기반 질병 분류기) + DiscreteCountMachine과 같은 debounce
    구조를 묶은 state machine. GrowthStateMachine과 마찬가지로 detect_event()가 이미지
    자체를 curr_value로 받아 내부에서 DiseaseAnalyzer.analyze()를 직접 돌린다.

    분류 결과(bacterial/fungal/healthy)는 세그멘테이션과 달리 프레임 단위로 잘 안 튀는
    편이지만, 그래도 한두 프레임의 오분류로 병에 걸렸다/나았다를 성급하게 확정하지
    않도록 DiscreteCountMachine과 동일한 debounce 알고리즘을 쓴다 — 같은 라벨이
    confirm_streak번 연속 관측되어야 confirmed_state가 바뀐다. 기본값 5는 "같은 진단이
    대략 5번(=현실적으로 촬영 주기를 고려하면 1시간 이상) 연속으로 나와야 진짜로
    상태가 바뀐 것으로 본다"는 의미다.

    LinearStateMachine과 마찬가지로 문구를 내장하지 않는다 — state_prompts/event_prompts/
    default_dialog는 known_plant_builder.py가 종별 JSON(예: 상추.json의 "disease" 센서)
    에서 읽어 넘겨준다.
    """

    def __init__(
        self,
        state_prompts: dict,
        event_prompts: dict,
        default_dialog: dict,
        disease_analyzer: Optional[DiseaseAnalyzer] = None,
        confirm_streak: int = 5,
        confirmed_state: Optional[DiseaseState] = None,
    ):
        self.state_prompts = state_prompts
        self.event_prompts = event_prompts
        self.default_dialog = default_dialog

        self.disease_analyzer = disease_analyzer or DiseaseAnalyzer()
        self.confirm_streak = confirm_streak
        self.confirmed_state = confirmed_state
        self.candidate_state = confirmed_state
        self.streak = 0

    def _debounce(self, observed_state: DiseaseState) -> None:
        if self.confirmed_state is None:
            self.confirmed_state = observed_state
            self.candidate_state = observed_state
            self.streak = 1
            return

        if observed_state == self.candidate_state:
            self.streak += 1
        else:
            self.candidate_state, self.streak = observed_state, 1

        if self.streak >= self.confirm_streak and observed_state != self.confirmed_state:
            self.confirmed_state = observed_state

    def detect_event(self, curr_value: Any) -> dict:
        """
        curr_value: DiseaseAnalyzer.analyze()가 받는 이미지 입력
        (파일 경로 / bytes / PIL.Image / np.ndarray 등 — sensory._image_io.ImageInput 참고).
        """
        payload = self.disease_analyzer.analyze(curr_value)
        observed_state = DiseaseState[payload["label"].upper()]

        prev_confirmed = self.confirmed_state
        self._debounce(observed_state)
        confirmed = self.confirmed_state

        if prev_confirmed is None or confirmed == prev_confirmed:
            event = DiseaseEvents.NO_EVENTS
        else:
            event = DiseaseEvents(prev_confirmed.value * 10 + confirmed.value)

        is_there_a_event = event != DiseaseEvents.NO_EVENTS

        event_prompt = self.event_prompts[event]
        state_prompt = self.state_prompts[confirmed]

        result = {
            "prompt": [event_prompt if is_there_a_event else state_prompt],
            "default_dialog": [self.default_dialog[event if is_there_a_event else confirmed]],
            "is_there_a_event": is_there_a_event,
            "event_prompt": event_prompt,
            "state_prompt": state_prompt,
            "raw_event": event,
            "state": confirmed,
        }

        return result

    def get_state(self) -> dict:
        return {
            "confirmed_state": self.confirmed_state.value if self.confirmed_state is not None else None,
            "candidate_state": self.candidate_state.value if self.candidate_state is not None else None,
            "streak": self.streak,
        }

    def load_state(self, state: dict) -> None:
        confirmed_value = state.get("confirmed_state")
        self.confirmed_state = DiseaseState(confirmed_value) if confirmed_value is not None else None
        candidate_value = state.get("candidate_state")
        self.candidate_state = DiseaseState(candidate_value) if candidate_value is not None else None
        self.streak = state.get("streak", 0)
