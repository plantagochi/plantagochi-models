import unittest

from planta_gochi.persona.mood import Expression, extract_expression
from planta_gochi.persona.sheets.known_plant_builder import build_engine
from planta_gochi.persona.state_engine import GrowthStateMachine


class _StubLeafAnalyzer:
    def __init__(self, leaf_count=0, canopy=0.0, leaf_size=0.0):
        self.leaf_count = leaf_count
        self.canopy = canopy
        self.leaf_size = leaf_size

    def analyze(self, image):
        return {
            "leaf_count": self.leaf_count,
            "leaf_areas": {"pixels": [], "ratio": [self.leaf_size] * self.leaf_count},
            "total_area": {"pixels": 0, "ratio": self.canopy},
        }


COUNT_ONLY_SPEC = {
    "type": "growth",
    "confirm_streak": 3,
    "count_machine": {
        "confirm_streak": 3,
        "state_prompts": {"STABLE": "잎이 {count}장이에요"},
        "event_prompts": {
            "NO_EVENTS": "잎이 {count}장이에요",
            "INCREASED": "새 잎이 났어요! 지금 {count}장",
            "DECREASED": "잎이 줄었어요... 지금 {count}장",
        },
        "default_dialog": {
            "STABLE": "잎이 {count}장이에요",
            "NO_EVENTS": "잎이 {count}장이에요",
            "INCREASED": "새 잎이 났어요! 지금 {count}장",
            "DECREASED": "잎이 줄었어요... 지금 {count}장",
        },
    },
}


def _build_count_only_engine() -> GrowthStateMachine:
    engine = build_engine(COUNT_ONLY_SPEC)
    engine.leaf_analyzer = _StubLeafAnalyzer(leaf_count=10)
    return engine


class CountOnlyGrowthBuildTests(unittest.TestCase):
    def test_builder_without_trend_section_leaves_trend_machines_unset(self):
        engine = _build_count_only_engine()
        self.assertIsNone(engine.canopy_machine)
        self.assertIsNone(engine.leaf_size_machine)
        self.assertIsNotNone(engine.count_machine)


class CountOnlyGrowthDetectTests(unittest.TestCase):
    def test_result_has_no_trend_keys(self):
        engine = _build_count_only_engine()
        result = engine.detect_event("any.jpg")
        self.assertNotIn("canopy_trend", result)
        self.assertNotIn("leaf_size_trend", result)
        self.assertEqual(set(result["raw_event"].keys()), {"leaf_count"})

    def test_prompt_and_dialog_come_only_from_count_machine(self):
        engine = _build_count_only_engine()
        result = engine.detect_event("any.jpg")
        self.assertEqual(len(result["prompt"]), 1)
        self.assertEqual(len(result["default_dialog"]), 1)
        self.assertEqual(result["prompt"][0], "잎이 10장이에요")

    def test_raw_metrics_and_stage_still_present(self):
        engine = _build_count_only_engine()
        result = engine.detect_event("any.jpg")
        self.assertEqual(result["raw_metrics"]["leaf_count"], 10)
        self.assertIn(result["stage"], (0, 1, 2, 3, 4, 5))

    def test_leaf_count_change_still_emits_event_after_debounce(self):
        engine = _build_count_only_engine()
        engine.detect_event("first.jpg")  # 10장 확정
        engine.leaf_analyzer.leaf_count = 15
        results = [engine.detect_event("next.jpg") for _ in range(3)]
        self.assertTrue(results[-1]["is_there_a_event"])
        self.assertEqual(results[-1]["raw_event"]["leaf_count"].name, "INCREASED")


class CountOnlyGrowthStateTests(unittest.TestCase):
    def test_get_state_has_only_count_machine(self):
        engine = _build_count_only_engine()
        engine.detect_event("any.jpg")
        self.assertEqual(set(engine.get_state().keys()), {"count_machine"})

    def test_load_state_ignores_trend_entries_it_has_no_machine_for(self):
        """trend 포함 state를 trend 없는 엔진에 넣어도 죽지 않고, count만 복원된다."""
        engine = _build_count_only_engine()
        engine.detect_event("any.jpg")
        full_state = {
            "count_machine": engine.get_state()["count_machine"],
            "canopy_machine": {"ema": 0.1, "prev_state": None, "raw_history": [], "smoothed_history": []},
            "leaf_size_machine": {"ema": 0.1, "prev_state": None, "raw_history": [], "smoothed_history": []},
        }
        fresh = _build_count_only_engine()
        fresh.load_state(full_state)
        self.assertEqual(fresh.count_machine.confirmed_count, 10)


class CountOnlyGrowthMoodTests(unittest.TestCase):
    def test_leaf_loss_alone_is_neutral_because_two_signals_are_required(self):
        """추세가 없으면 growth 부정 신호는 leaf_count 하나뿐이라, 2개 이상 겹쳐야 하는
        DISTRESSED 규칙은 growth만으로는 절대 발동하지 않는다 — 이 조합의 의도된 결과다."""
        engine = _build_count_only_engine()
        engine.detect_event("first.jpg")  # 10장
        engine.leaf_analyzer.leaf_count = 5
        for _ in range(3):
            result = engine.detect_event("next.jpg")
        self.assertEqual(result["raw_event"]["leaf_count"].name, "DECREASED")
        self.assertEqual(extract_expression({"growth": result}), Expression.NEUTRAL)


if __name__ == "__main__":
    unittest.main()
