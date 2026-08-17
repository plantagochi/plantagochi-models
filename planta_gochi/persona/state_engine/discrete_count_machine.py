from planta_gochi.persona.state_engine import CountEvents, CountState, StateEngine


class DiscreteCountMachine(StateEngine):
    """
    LinearStateMachine과 같은 모양(detect_event()가 prompt/default_dialog/
    is_there_a_event/event_prompt/state_prompt/raw_event를 담은 dict를 반환)을 갖되,
    leaf_count처럼 물리적 threshold 구간이 없는 임의의 정수 카운트를 다루는 state machine.

    세그멘테이션 프레임 단위 오검출로 한두 프레임만 값이 튈 수 있으므로, 같은 값이
    confirm_streak번 연속 관측되어야 confirmed_count를 갱신하는 debounce 알고리즘을 쓴다.

    LinearStateMachine과 마찬가지로 문구를 내장하지 않는다 — state_prompts/event_prompts/
    default_dialog는 known_plant_builder.py가 종별 JSON(예: 상추.json의 "growth" 센서)에서
    읽어 넘겨준다. 각 문구는 {count}를 confirmed_count로 채우는 템플릿이다.
    """

    def __init__(
        self,
        state_prompts: dict,
        event_prompts: dict,
        default_dialog: dict,
        confirm_streak: int = 3,
        confirmed_count: int = None,
    ):
        self.state_prompts = state_prompts
        self.event_prompts = event_prompts
        self.default_dialog = default_dialog

        self.confirm_streak = confirm_streak
        self.confirmed_count = confirmed_count
        self.candidate_count = confirmed_count
        self.streak = 0

    def _debounce(self, observed_count):
        if self.confirmed_count is None:
            self.confirmed_count = observed_count
            self.candidate_count = observed_count
            self.streak = 1
            return

        if observed_count == self.candidate_count:
            self.streak += 1
        else:
            self.candidate_count, self.streak = observed_count, 1

        if self.streak >= self.confirm_streak and observed_count != self.confirmed_count:
            self.confirmed_count = observed_count

    def detect_event(self, curr_value):
        prev_confirmed = self.confirmed_count
        self._debounce(curr_value)
        confirmed = self.confirmed_count

        if prev_confirmed is None or confirmed == prev_confirmed:
            event = CountEvents.NO_EVENTS
        elif confirmed > prev_confirmed:
            event = CountEvents.INCREASED
        else:
            event = CountEvents.DECREASED

        is_there_a_event = event != CountEvents.NO_EVENTS

        event_prompt = self.event_prompts[event].format(count=confirmed)
        state_prompt = self.state_prompts[CountState.STABLE].format(count=confirmed)
        dialog_key = event if is_there_a_event else CountState.STABLE
        default_dialog = self.default_dialog[dialog_key].format(count=confirmed)

        result = {
            "prompt": [event_prompt if is_there_a_event else state_prompt],
            "default_dialog": [default_dialog],
            "is_there_a_event": is_there_a_event,
            "event_prompt": event_prompt,
            "state_prompt": state_prompt,
            "raw_event": event,
            "count": confirmed,
        }

        return result
