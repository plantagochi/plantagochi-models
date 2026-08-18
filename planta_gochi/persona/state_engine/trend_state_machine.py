import statistics
from collections import deque

from planta_gochi.persona.state_engine import TrendEvents, TrendState, StateEngine


class TrendStateMachine(StateEngine):
    """
    LinearStateMachine과 같은 모양(state_prompts/event_prompts/default_dialog +
    detect_event()가 prompt/default_dialog/is_there_a_event/event_prompt/state_prompt/
    raw_event를 담은 dict를 반환)을 갖되, 절대값 threshold 대신 노이즈에 강건한
    "추세"(growing/stagnant/declining)를 state로 사용하는 state machine.

    curr_value 하나하나가 들어올 때마다 내부적으로
    1) MAD 기반 이상치 제거 -> 2) EMA 스무딩 -> 3) 선형회귀 slope -> 4) deadzone 3구간 분류
    를 거쳐 TrendState를 갱신한다. 각 단계가 서로 다른 종류의 노이즈를 담당하므로 순서를
    바꾸거나 생략하면 안 된다.

    LinearStateMachine과 마찬가지로 문구를 내장하지 않는다 — state_prompts/event_prompts/
    default_dialog는 known_plant_builder.py가 종별 JSON(예: 상추.json의 "growth" 센서)에서
    읽어 넘겨준다.
    """

    def __init__(
        self,
        state_prompts: dict,
        event_prompts: dict,
        default_dialog: dict,
        window: int = 7,
        mad_k: float = 3.5,
        growth_eps: float = 0.001,
        decline_eps: float = 0.001,
        prev_state: TrendState = None,
    ):
        self.state_prompts = state_prompts
        self.event_prompts = event_prompts
        self.default_dialog = default_dialog

        self.window = window
        self.mad_k = mad_k
        self.growth_eps = growth_eps
        self.decline_eps = decline_eps

        self.raw_history = deque(maxlen=window * 3)
        self.smoothed_history = deque(maxlen=window)
        self.ema = None
        self.alpha = 2 / (window + 1)

        self.prev_state = prev_state

    # ---- 1단계: MAD 기반 이상치 판정 ----
    def _is_outlier(self, value):
        if len(self.raw_history) < 5:
            return False
        median = statistics.median(self.raw_history)
        mad = statistics.median(abs(x - median) for x in self.raw_history) or 1e-6
        modified_z = 0.6745 * (value - median) / mad
        return abs(modified_z) > self.mad_k

    # ---- 2단계: EMA 스무딩 (이상치가 아닐 때만 raw_history/EMA/smoothed_history 갱신) ----
    def _update_ema(self, value):
        if self._is_outlier(value):
            return
        self.raw_history.append(value)
        self.ema = value if self.ema is None else self.alpha * value + (1 - self.alpha) * self.ema
        self.smoothed_history.append(self.ema)

    # ---- 3단계: 최근 window 구간 스무딩 값 전체에 대한 최소제곱 회귀 기울기 ----
    def _slope(self):
        ys = list(self.smoothed_history)
        n = len(ys)
        xs = list(range(n))
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs)
        return num / den if den else 0.0

    # ---- 4단계: epsilon(deadzone) 기반 3구간 분류 ----
    def _current_state(self) -> TrendState:
        if len(self.smoothed_history) < 3:
            return TrendState.INSUFFICIENT_DATA
        slope = self._slope()
        if slope > self.growth_eps:
            return TrendState.GROWING
        if slope < -self.decline_eps:
            return TrendState.DECLINING
        return TrendState.STAGNANT

    def detect_event(self, curr_value):
        self._update_ema(curr_value)
        curr_state = self._current_state()

        prev_state = self.prev_state
        self.prev_state = curr_state

        if prev_state is None or prev_state == curr_state:
            event = TrendEvents.NO_EVENTS
        else:
            event = TrendEvents(prev_state.value * 10 + curr_state.value)

        is_there_a_event = event != TrendEvents.NO_EVENTS

        event_prompt = self.event_prompts[event]
        state_prompt = self.state_prompts[curr_state]

        # 아직 판단할 데이터가 부족해 "그대로"인 경우("아직 잘 모르겠어요" 류)는 매 프레임
        # 반복해서 떠들 만한 내용이 아니므로 prompt/default_dialog를 아예 비워서 emit하지
        # 않는다. INSUFFICIENT_DATA로 전환/이탈하는 이벤트 자체는 의미가 있으므로 그대로 둔다.
        if not is_there_a_event and curr_state == TrendState.INSUFFICIENT_DATA:
            prompt = []
            default_dialog = []
        else:
            prompt = [event_prompt if is_there_a_event else state_prompt]
            default_dialog = [self.default_dialog[event if is_there_a_event else curr_state]]

        result = {
            "prompt": prompt,
            "default_dialog": default_dialog,
            "is_there_a_event": is_there_a_event,
            "event_prompt": event_prompt,
            "state_prompt": state_prompt,
            "raw_event": event,
            "state": curr_state,
        }

        return result

    def get_state(self) -> dict:
        # raw_history/smoothed_history는 slope 계산에 그대로 필요한 배열형 데이터다
        # (다음 update부터 다시 이어서 MAD/EMA/회귀를 계산하려면 window개 분량의 과거
        # 값이 있어야 함). ema/prev_state는 스칼라라 그대로 담고, prev_state(enum)만
        # .value로 변환한다.
        return {
            "ema": self.ema,
            "prev_state": self.prev_state.value if self.prev_state is not None else None,
            "raw_history": list(self.raw_history),
            "smoothed_history": list(self.smoothed_history),
        }

    def load_state(self, state: dict) -> None:
        self.ema = state.get("ema")
        prev_state_value = state.get("prev_state")
        self.prev_state = TrendState(prev_state_value) if prev_state_value is not None else None
        self.raw_history = deque(state.get("raw_history", []), maxlen=self.window * 3)
        self.smoothed_history = deque(state.get("smoothed_history", []), maxlen=self.window)
