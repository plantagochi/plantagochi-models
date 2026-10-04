import unittest

from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.persona.state_engine import LinearState
from planta_gochi.plant_ai import PlantAI


class _RecordingLLM:
    def __init__(self):
        self.calls = []

    def ask_batch(self, species, situations):
        self.calls.append(situations)
        return None


class _StubLeafAnalyzer:
    def analyze(self, image):
        return {"leaf_count": 10, "leaf_areas": {"pixels": [], "ratio": [0.03] * 10}, "total_area": {"pixels": 0, "ratio": 0.3}}


class _StubDiseaseAnalyzer:
    def __init__(self, label="healthy"):
        self.label = label

    def analyze(self, image):
        return {"label": self.label}


def _make_ai(disease_label="healthy"):
    persona = build_known_persona(
        "상추",
        leaf_analyzer=_StubLeafAnalyzer(),
        disease_analyzer=_StubDiseaseAnalyzer(disease_label),
    )
    llm = _RecordingLLM()
    return PlantAI(persona, llm=llm), llm


class SpeakStatusEffectFilterTests(unittest.TestCase):
    def test_without_status_effect_every_sensor_is_used(self):
        ai, llm = _make_ai()
        ai.speak({"temperature": 20, "humidity": 60, "growth": "any.jpg"})
        self.assertEqual(set(llm.calls[-1].keys()), {"temperature", "humidity", "growth"})

    def test_temperature_critical_returns_only_temperature_messages(self):
        ai, llm = _make_ai()
        messages = ai.speak({"temperature": 40, "humidity": 60, "growth": "any.jpg"})
        self.assertEqual(set(llm.calls[-1].keys()), {"temperature"})
        expected = ai.persona.engines["temperature"].default_dialog[LinearState.VALUE_CRITICAL_HIGH]
        self.assertEqual(messages, [expected])

    def test_non_critical_sensors_are_dropped_when_status_effect_exists(self):
        ai, llm = _make_ai()
        ai.speak({"temperature": 40, "humidity": 60, "growth": "any.jpg"})
        self.assertNotIn("humidity", llm.calls[-1])
        self.assertNotIn("growth", llm.calls[-1])

    def test_disease_status_effect_keeps_only_disease(self):
        ai, llm = _make_ai(disease_label="bacterial")
        ai.speak({"temperature": 20, "disease": "any.jpg"})
        self.assertEqual(set(llm.calls[-1].keys()), {"disease"})

    def test_full_result_is_kept_for_read_methods(self):
        ai, _ = _make_ai()
        ai.speak({"temperature": 40, "growth": "any.jpg"})
        self.assertEqual(ai.get_growth_stage(), 1)
        self.assertTrue(ai.get_status_effect())


if __name__ == "__main__":
    unittest.main()
