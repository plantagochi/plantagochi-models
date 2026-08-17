import random
import unittest

from planta_gochi.persona.state_engine import (
    CountEvents,
    CountState,
    DiscreteCountMachine,
    GrowthStateMachine,
    TrendEvents,
    TrendState,
    TrendStateMachine,
)

# 텍스트 내용 자체는 known_plant_builder.py/상추.json이 책임지므로, 여기서는 알고리즘만
# 검증한다. enum 이름을 그대로 문구로 써서 최소한의 유효한 dict만 만든다.
_TREND_STATE_PROMPTS = {s: s.name for s in TrendState}
_TREND_EVENT_PROMPTS = {e: e.name for e in TrendEvents}
_TREND_DEFAULT_DIALOG = {**_TREND_STATE_PROMPTS, **_TREND_EVENT_PROMPTS}

_COUNT_STATE_PROMPTS = {s: s.name for s in CountState}
_COUNT_EVENT_PROMPTS = {e: e.name for e in CountEvents}
_COUNT_DEFAULT_DIALOG = {**_COUNT_STATE_PROMPTS, **_COUNT_EVENT_PROMPTS}


def _trend_machine(**kwargs) -> TrendStateMachine:
    return TrendStateMachine(
        state_prompts=_TREND_STATE_PROMPTS,
        event_prompts=_TREND_EVENT_PROMPTS,
        default_dialog=_TREND_DEFAULT_DIALOG,
        **kwargs,
    )


def _count_machine(**kwargs) -> DiscreteCountMachine:
    return DiscreteCountMachine(
        state_prompts=_COUNT_STATE_PROMPTS,
        event_prompts=_COUNT_EVENT_PROMPTS,
        default_dialog=_COUNT_DEFAULT_DIALOG,
        **kwargs,
    )


def _make_payload(leaf_count: int, total_ratio: float, leaf_ratios: list) -> dict:
    px_per_ratio = 50000  # 임의 스케일. invariant만 지키면 값 자체는 중요하지 않음.
    leaf_pixels = [round(r * px_per_ratio) for r in leaf_ratios]
    return {
        "leaf_count": leaf_count,
        "leaf_areas": {"pixels": leaf_pixels, "ratio": leaf_ratios},
        "total_area": {"pixels": sum(leaf_pixels), "ratio": total_ratio},
    }


class _StubLeafAnalyzer:
    """실제 YOLO 모델 없이 GrowthStateMachine을 테스트하기 위한 가짜 LeafAnalyzer.
    detect_event()에 넘긴 curr_value(원본 이미지)가 그대로 analyze()로 전달되는지도 기록한다."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    def analyze(self, image):
        self.calls.append(image)
        return self._payloads.pop(0)


class TrendStateMachineTests(unittest.TestCase):
    def test_linear_increase_with_noise_is_growing(self):
        random.seed(42)
        machine = _trend_machine(window=7)
        value = 0.05
        result = None
        for _ in range(30):
            value += 0.002
            result = machine.detect_event(value + random.gauss(0, 0.0005))
        self.assertEqual(result["state"], TrendState.GROWING)

    def test_flat_with_noise_is_stagnant(self):
        random.seed(7)
        machine = _trend_machine(window=7)
        result = None
        for _ in range(30):
            result = machine.detect_event(0.05 + random.gauss(0, 0.0005))
        self.assertEqual(result["state"], TrendState.STAGNANT)

    def test_single_extreme_spike_is_filtered_and_growth_survives(self):
        random.seed(123)
        machine = _trend_machine(window=7)
        value = 0.05

        # raw_history가 5개 이상 쌓일 때까지는 outlier 판정을 하지 않으므로 먼저 채워준다.
        for _ in range(6):
            value += 0.002
            machine.detect_event(value + random.gauss(0, 0.0005))

        ema_before_spike = machine.ema
        spike_value = value * 5  # 정상값의 5배
        self.assertTrue(machine._is_outlier(spike_value))

        machine.detect_event(spike_value)
        self.assertEqual(machine.ema, ema_before_spike)  # EMA가 급변하지 않아야 함

        # 나머지 정상 시퀀스를 마저 흘려보내도 추세는 그대로 growing이어야 한다.
        result = None
        for _ in range(24):
            value += 0.002
            result = machine.detect_event(value + random.gauss(0, 0.0005))
        self.assertEqual(result["state"], TrendState.GROWING)


class DiscreteCountMachineTests(unittest.TestCase):
    def test_blip_below_confirm_streak_does_not_change_confirmed_count(self):
        machine = _count_machine(confirm_streak=3)
        machine.detect_event(5)  # 초기값 확정
        machine.detect_event(5)
        machine.detect_event(6)  # 한 프레임만 튐
        result = machine.detect_event(5)  # 원래 값으로 복귀
        self.assertEqual(result["count"], 5)
        self.assertEqual(machine.confirmed_count, 5)
        self.assertFalse(result["is_there_a_event"])

    def test_streak_at_or_above_confirm_streak_updates_confirmed_count(self):
        machine = _count_machine(confirm_streak=3)
        machine.detect_event(5)  # 초기값 확정
        machine.detect_event(6)
        machine.detect_event(6)
        result = machine.detect_event(6)  # 3번 연속
        self.assertEqual(result["count"], 6)
        self.assertEqual(machine.confirmed_count, 6)
        self.assertTrue(result["is_there_a_event"])
        self.assertEqual(result["raw_event"], CountEvents.INCREASED)


class GrowthStateMachineTests(unittest.TestCase):
    def test_detect_event_feeds_raw_image_to_leaf_analyzer_and_returns_expected_shape(self):
        stub = _StubLeafAnalyzer([
            _make_payload(4, 0.05, [0.01, 0.01, 0.01, 0.01]),
        ])
        machine = GrowthStateMachine(
            count_machine=_count_machine(),
            canopy_machine=_trend_machine(),
            leaf_size_machine=_trend_machine(),
            leaf_analyzer=stub,
        )

        result = machine.detect_event("frame_001.jpg")

        # GrowthState.update()처럼 미리 분석된 dict가 아니라, 원본 이미지가 그대로
        # LeafAnalyzer.analyze()에 전달되어야 한다.
        self.assertEqual(stub.calls, ["frame_001.jpg"])

        self.assertEqual(
            set(result.keys()),
            {
                "prompt",
                "default_dialog",
                "is_there_a_event",
                "leaf_count",
                "canopy_trend",
                "leaf_size_trend",
                "raw_event",
            },
        )
        self.assertEqual(result["leaf_count"]["count"], 4)
        # 첫 호출이라 canopy/leaf_size는 아직 INSUFFICIENT_DATA -> prompt를 emit하지 않는다.
        # leaf_count(1개)만 남는다.
        self.assertEqual(result["prompt"], [result["leaf_count"]["prompt"][0]])
        self.assertTrue(all(isinstance(m, str) for m in result["prompt"]))
        self.assertEqual(result["default_dialog"], [result["leaf_count"]["default_dialog"][0]])
        self.assertTrue(all(isinstance(m, str) for m in result["default_dialog"]))
        self.assertEqual(result["canopy_trend"]["prompt"], [])
        self.assertEqual(result["leaf_size_trend"]["prompt"], [])
        self.assertIsInstance(result["is_there_a_event"], bool)

    def test_leaf_count_increase_detected_after_confirm_streak(self):
        stub = _StubLeafAnalyzer([
            _make_payload(4, 0.05, [0.01, 0.01, 0.01, 0.01]),
            _make_payload(4, 0.05, [0.01, 0.01, 0.01, 0.01]),
            _make_payload(5, 0.06, [0.01, 0.01, 0.01, 0.01, 0.01]),
            _make_payload(5, 0.06, [0.01, 0.01, 0.01, 0.01, 0.01]),
        ])
        machine = GrowthStateMachine(
            count_machine=_count_machine(confirm_streak=2),
            canopy_machine=_trend_machine(),
            leaf_size_machine=_trend_machine(),
            leaf_analyzer=stub,
        )

        machine.detect_event("f1")
        machine.detect_event("f2")
        result_3 = machine.detect_event("f3")  # 5장 관측 1회차 -> streak 미달
        self.assertEqual(result_3["leaf_count"]["raw_event"], CountEvents.NO_EVENTS)

        result_4 = machine.detect_event("f4")  # 5장 관측 2회차 -> confirm_streak 도달
        self.assertEqual(result_4["leaf_count"]["raw_event"], CountEvents.INCREASED)
        self.assertTrue(result_4["is_there_a_event"])


if __name__ == "__main__":
    unittest.main()
