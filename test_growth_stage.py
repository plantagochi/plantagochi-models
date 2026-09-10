import unittest

from planta_gochi.persona.growth_stage import STAGE_REFERENCE_MEANS, classify_growth_stage
from planta_gochi.persona.sheets import build_known_persona


class ClassifyGrowthStageTests(unittest.TestCase):
    def test_exact_reference_values_return_that_stage(self):
        for stage, ref in STAGE_REFERENCE_MEANS.items():
            result = classify_growth_stage(ref["leaf_count"], ref["canopy_ratio"], ref["leaf_size_ratio"])
            self.assertEqual(result, stage)

    def test_extreme_values_still_clamp_to_a_registered_stage(self):
        """STAGE_REFERENCE_MEANS에 없는 값이 나와도(범위 밖), 등록된 단계 중 가장
        가까운 것으로만 클램프되어야 한다 — 등록되지 않은 새 정수를 반환하면 안 된다."""
        extreme_cases = [
            (1, 0.0, 0.0),           # leaf_count는 0이 아니지만 관측치가 거의 없는 극단
            (500, 5.0, 5.0),         # 5단계보다 훨씬 더 "자란" 것처럼 보이는 극단
            (1000000, 100.0, 100.0),
        ]
        for leaf_count, canopy_ratio, leaf_size_ratio in extreme_cases:
            stage = classify_growth_stage(leaf_count, canopy_ratio, leaf_size_ratio)
            self.assertIn(stage, (1, 2, 3, 4, 5))

    def test_far_beyond_stage_5_clamps_to_stage_5(self):
        """가장 큰 값(현재 5단계)보다 훨씬 큰 값을 넣어도 5단계로 클램프된다."""
        stage = classify_growth_stage(1000, 10.0, 10.0)
        self.assertEqual(stage, 5)

    def test_far_below_stage_1_clamps_to_stage_1(self):
        """leaf_count가 0이 아닌 한, 아무리 작아도 0단계가 아니라 1단계로 클램프된다."""
        stage = classify_growth_stage(1, 0.0, 0.0)
        self.assertEqual(stage, 1)

    def test_five_stages_are_registered(self):
        """1~5단계 기준값이 전부 있는지, 실수로 더 늘거나 줄지 않았는지 확인."""
        self.assertEqual(set(STAGE_REFERENCE_MEANS.keys()), {1, 2, 3, 4, 5})

    def test_stage_5_continues_upward_trend_from_stage_4(self):
        """1~4단계 사이엔 실측 노이즈로 완벽한 단조 증가가 아닌 지표가 이미 있었지만
        (예: leaf_size_ratio는 1단계가 2단계보다 살짝 높음), 새로 추가된 5단계만큼은
        4단계보다 세 지표 전부 커야 한다 — 값을 잘못 입력했으면 여기서 걸린다."""
        stage4, stage5 = STAGE_REFERENCE_MEANS[4], STAGE_REFERENCE_MEANS[5]
        for key in ("leaf_count", "canopy_ratio", "leaf_size_ratio"):
            self.assertGreater(
                stage5[key], stage4[key],
                f"5단계의 {key}({stage5[key]})가 4단계({stage4[key]})보다 크지 않음",
            )

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
        """0단계 override는 정확히 leaf_count == 0일 때만 발동해야 한다 (1은 정상적으로 1~5단계 분류)."""
        stage = classify_growth_stage(1, 0.3194, 0.0275)  # 1단계 평균과 거의 같은 canopy/leaf_size
        self.assertNotEqual(stage, 0)
        self.assertIn(stage, (1, 2, 3, 4, 5))


class _StubZeroLeafAnalyzer:
    """실제 LeafAnalyzer 대신 꽂아서, leaf_count=0인 프레임을 결정적으로 재현하는 스텁."""

    def analyze(self, image):
        return {
            "leaf_count": 0,
            "leaf_areas": {"pixels": [], "ratio": []},
            "total_area": {"pixels": 0, "ratio": 0.0},
        }


class GrowthStateMachineStageIntegrationTests(unittest.TestCase):
    def test_growth_result_includes_stage_within_1_to_5(self):
        persona = build_known_persona("상추")
        result = persona.update({"growth": "sample_easy.jpg"})
        growth = result["growth"]

        self.assertIn("stage", growth)
        self.assertIn(growth["stage"], (1, 2, 3, 4, 5))

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
