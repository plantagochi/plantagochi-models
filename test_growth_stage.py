import unittest

from planta_gochi.persona.growth_stage import STAGE_REFERENCE_MEANS, classify_growth_stage
from planta_gochi.persona.sheets import build_known_persona


class ClassifyGrowthStageTests(unittest.TestCase):
    def test_exact_reference_values_return_that_stage(self):
        for stage, ref in STAGE_REFERENCE_MEANS.items():
            result = classify_growth_stage(ref["leaf_count"], ref["canopy_ratio"], ref["leaf_size_ratio"])
            self.assertEqual(result, stage)

    def test_never_returns_5_even_for_extreme_values(self):
        """5단계는 기준값이 아예 없으므로, 아무리 값이 크거나 작아도 1~4만 나와야 한다."""
        extreme_cases = [
            (0, 0.0, 0.0),          # 관측치가 거의 없는 극단
            (500, 5.0, 5.0),        # 5단계보다 훨씬 더 "자란" 것처럼 보이는 극단
            (1000000, 100.0, 100.0),
        ]
        for leaf_count, canopy_ratio, leaf_size_ratio in extreme_cases:
            stage = classify_growth_stage(leaf_count, canopy_ratio, leaf_size_ratio)
            self.assertIn(stage, (1, 2, 3, 4))

    def test_far_beyond_stage_4_clamps_to_stage_4(self):
        stage = classify_growth_stage(1000, 10.0, 10.0)
        self.assertEqual(stage, 4)

    def test_far_below_stage_1_clamps_to_stage_1(self):
        stage = classify_growth_stage(0, 0.0, 0.0)
        self.assertEqual(stage, 1)

    def test_only_four_stages_are_registered(self):
        """5단계 기준값이 실수로라도 들어가면 안 되므로, reference table 자체도 확인."""
        self.assertEqual(set(STAGE_REFERENCE_MEANS.keys()), {1, 2, 3, 4})


class GrowthStateMachineStageIntegrationTests(unittest.TestCase):
    def test_growth_result_includes_stage_within_1_to_4(self):
        persona = build_known_persona("상추")
        result = persona.update({"growth": "sample_easy.jpg"})
        growth = result["growth"]

        self.assertIn("stage", growth)
        self.assertIn(growth["stage"], (1, 2, 3, 4))

        self.assertIn("raw_metrics", growth)
        self.assertIsInstance(growth["raw_metrics"]["leaf_count"], int)
        self.assertIsInstance(growth["raw_metrics"]["canopy_ratio"], float)
        self.assertIsInstance(growth["raw_metrics"]["leaf_size_ratio"], float)


if __name__ == "__main__":
    unittest.main()
