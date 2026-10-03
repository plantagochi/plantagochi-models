import unittest
from unittest import mock

from planta_gochi.persona.sheets import build_known_persona, build_persona_from_spec
from planta_gochi.persona.sheets.known_plant_builder import KNOWN_PLANTS_DIR
from planta_gochi.sensory import DiseaseAnalyzer, LeafAnalyzer


class _StubLeafAnalyzer:
    def analyze(self, image):
        return {"leaf_count": 0, "leaf_areas": {"pixels": [], "ratio": []}, "total_area": {"pixels": 0, "ratio": 0.0}}


class _StubDiseaseAnalyzer:
    def analyze(self, image):
        return {"label": "healthy"}


class InjectedAnalyzerTests(unittest.TestCase):
    def test_build_known_persona_uses_injected_instances(self):
        leaf, disease = _StubLeafAnalyzer(), _StubDiseaseAnalyzer()
        persona = build_known_persona("상추", leaf_analyzer=leaf, disease_analyzer=disease)
        self.assertIs(persona.engines["growth"].leaf_analyzer, leaf)
        self.assertIs(persona.engines["disease"].disease_analyzer, disease)

    def test_build_persona_from_spec_uses_injected_instances(self):
        import json
        spec = json.loads((KNOWN_PLANTS_DIR / "상추.json").read_text(encoding="utf-8"))
        leaf, disease = _StubLeafAnalyzer(), _StubDiseaseAnalyzer()
        persona = build_persona_from_spec(spec, leaf_analyzer=leaf, disease_analyzer=disease)
        self.assertIs(persona.engines["growth"].leaf_analyzer, leaf)
        self.assertIs(persona.engines["disease"].disease_analyzer, disease)

    def test_two_personas_can_share_one_analyzer_instance(self):
        leaf, disease = _StubLeafAnalyzer(), _StubDiseaseAnalyzer()
        first = build_known_persona("상추", leaf_analyzer=leaf, disease_analyzer=disease)
        second = build_known_persona("상추", leaf_analyzer=leaf, disease_analyzer=disease)
        self.assertIs(first.engines["growth"].leaf_analyzer, second.engines["growth"].leaf_analyzer)
        self.assertIs(first.engines["disease"].disease_analyzer, second.engines["disease"].disease_analyzer)

    def test_injected_analyzers_skip_model_construction(self):
        """주입하면 LeafAnalyzer/DiseaseAnalyzer를 새로 만들지 않는다(모델 재로딩 방지가 목적)."""
        with mock.patch("planta_gochi.persona.sheets.known_plant_builder.LeafAnalyzer") as leaf_cls, \
             mock.patch("planta_gochi.persona.sheets.known_plant_builder.DiseaseAnalyzer") as disease_cls:
            build_known_persona("상추", leaf_analyzer=_StubLeafAnalyzer(), disease_analyzer=_StubDiseaseAnalyzer())
        leaf_cls.assert_not_called()
        disease_cls.assert_not_called()

    def test_without_injection_analyzers_are_built_from_spec(self):
        with mock.patch("planta_gochi.persona.sheets.known_plant_builder.LeafAnalyzer", wraps=LeafAnalyzer) as leaf_cls, \
             mock.patch("planta_gochi.persona.sheets.known_plant_builder.DiseaseAnalyzer", wraps=DiseaseAnalyzer) as disease_cls:
            build_known_persona("상추")
        leaf_cls.assert_called_once()
        disease_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
