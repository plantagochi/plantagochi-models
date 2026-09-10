import unittest

from planta_gochi.persona.growth_stage import STAGE_REFERENCE_MEANS, classify_growth_stage
from planta_gochi.persona.sheets import build_known_persona


class ClassifyGrowthStageTests(unittest.TestCase):
    def test_exact_reference_values_return_that_stage(self):
        for stage, ref in STAGE_REFERENCE_MEANS.items():
            result = classify_growth_stage(ref["leaf_count"], ref["canopy_ratio"], ref["leaf_size_ratio"])
            self.assertEqual(result, stage)

    def test_never_returns_5_even_for_extreme_values(self):
        """5단계는 기준값이 아예 없으므로, 아무리 값이 크거나 작아도 1~4(혹은 leaf_count==0일 때 0)만 나와야 한다."""
        extreme_cases = [
            (1, 0.0, 0.0),           # leaf_count는 0이 아니지만 관측치가 거의 없는 극단
            (500, 5.0, 5.0),         # 5단계보다 훨씬 더 "자란" 것처럼 보이는 극단
            (1000000, 100.0, 100.0),
        ]
        for leaf_count, canopy_ratio, leaf_size_ratio in extreme_cases:
            stage = classify_growth_stage(leaf_count, canopy_ratio, leaf_size_ratio)
            self.assertIn(stage, (1, 2, 3, 4))

    def test_far_beyond_stage_4_clamps_to_stage_4(self):
        stage = classify_growth_stage(1000, 10.0, 10.0)
        self.assertEqual(stage, 4)

    def test_far_below_stage_1_clamps_to_stage_1(self):
        """leaf_count가 0이 아닌 한, 아무리 작아도 0단계가 아니라 1단계로 클램프된다."""
        stage = classify_growth_stage(1, 0.0, 0.0)
        self.assertEqual(stage, 1)

    def test_only_four_stages_are_registered(self):
        """5단계 기준값이 실수로라도 들어가면 안 되므로, reference table 자체도 확인."""
        self.assertEqual(set(STAGE_REFERENCE_MEANS.keys()), {1, 2, 3, 4})

    def test_leaf_count_zero_overrides_to_stage_0(self):
        """leaf_count가 정확히 0이면, canopy/leaf_size 값과 무관하게 항상 0단계다."""
        cases = [
            (0, 0.0, 0.0),      # 정말 아무것도 없는 경우
            (0, 0.6177, 0.0505),  # canopy/leaf_size는 4단계 평균과 똑같아도(카메라 노이즈 등으로 생길 수 있음)
            (0, 999.0, 999.0),  # 극단적인 canopy/leaf_size라도
        ]
        for leaf_count, canopy_ratio, leaf_size_ratio in cases:
            stage = classify_growth_stage(leaf_count, canopy_ratio, leaf_size_ratio)
            self.assertEqual(stage, 0, f"leaf_count=0인데 stage={stage} (canopy={canopy_ratio}, leaf_size={leaf_size_ratio})")

    def test_leaf_count_one_does_not_trigger_zero_override(self):
        """0단계 override는 정확히 leaf_count == 0일 때만 발동해야 한다 (1은 정상적으로 1~4단계 분류)."""
        stage = classify_growth_stage(1, 0.3194, 0.0275)  # 1단계 평균과 거의 같은 canopy/leaf_size
        self.assertNotEqual(stage, 0)
        self.assertIn(stage, (1, 2, 3, 4))


class _StubZeroLeafAnalyzer:
    """실제 LeafAnalyzer 대신 꽂아서, leaf_count=0인 프레임을 결정적으로 재현하는 스텁."""

    def analyze(self, image):
        return {
            "leaf_count": 0,
            "leaf_areas": {"pixels": [], "ratio": []},
            "total_area": {"pixels": 0, "ratio": 0.0},
        }


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

    def test_real_pipeline_leaf_count_zero_yields_stage_0(self):
        """LeafAnalyzer가 실제로 leaf_count=0을 내놓는 프레임을 만나면, Persona.update()를
        거친 최종 결과의 growth.stage도 0이어야 한다(known_plant_builder가 만든 실제
        상추 JSON 기반 GrowthStateMachine 그대로, LeafAnalyzer만 스텁으로 교체)."""
        persona = build_known_persona("상추")
        persona.engines["growth"].leaf_analyzer = _StubZeroLeafAnalyzer()

        result = persona.update({"growth": "이 값은 스텁이라 실제로 안 쓰임.jpg"})
        growth = result["growth"]

        self.assertEqual(growth["raw_metrics"]["leaf_count"], 0)
        self.assertEqual(growth["stage"], 0)


if __name__ == "__main__":
    unittest.main()
