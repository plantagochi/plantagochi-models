import unittest

from planta_gochi.llm.connection import LLMConnection
from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.persona.state_engine import DiseaseState, LinearState
from planta_gochi.persona.status_effect import StatusEffect, extract_status_effects
from planta_gochi.plant_ai import PlantAI


def _linear(state: LinearState) -> dict:
    """실제 LinearStateMachine.detect_event() 결과 중 상태이상 추출이 실제로 읽는 부분만 흉내."""
    return {"state": state}


class _NoOpLLM(LLMConnection):
    def ask_batch(self, species, situations):
        return None


class ExtractStatusEffectsTests(unittest.TestCase):
    def test_no_effects_when_everything_normal(self):
        result = {
            "temperature": _linear(LinearState.VALUE_MID),
            "humidity": _linear(LinearState.VALUE_MID),
            "soil_temp": _linear(LinearState.VALUE_MID),
            "soil_humidity": _linear(LinearState.VALUE_MID),
            "disease": _linear(DiseaseState.HEALTHY),
        }
        self.assertEqual(extract_status_effects(result), [])

    def test_mild_low_high_are_not_status_effects(self):
        """"극단적으로"가 아니라 완만한 LOW/HIGH는 상태이상이 아니다."""
        result = {
            "temperature": _linear(LinearState.VALUE_LOW),
            "humidity": _linear(LinearState.VALUE_HIGH),
        }
        self.assertEqual(extract_status_effects(result), [])

    def test_temperature_critical_high_and_low(self):
        self.assertEqual(
            extract_status_effects({"temperature": _linear(LinearState.VALUE_CRITICAL_HIGH)}),
            [StatusEffect.SCORCHING_TEMPERATURE],
        )
        self.assertEqual(
            extract_status_effects({"temperature": _linear(LinearState.VALUE_CRITICAL_LOW)}),
            [StatusEffect.FREEZING_TEMPERATURE],
        )

    def test_humidity_critical_high_and_low(self):
        self.assertEqual(
            extract_status_effects({"humidity": _linear(LinearState.VALUE_CRITICAL_HIGH)}),
            [StatusEffect.JUNGLE_HUMIDITY],
        )
        self.assertEqual(
            extract_status_effects({"humidity": _linear(LinearState.VALUE_CRITICAL_LOW)}),
            [StatusEffect.DESERT_HUMIDITY],
        )

    def test_soil_temp_critical_high_and_low(self):
        self.assertEqual(
            extract_status_effects({"soil_temp": _linear(LinearState.VALUE_CRITICAL_HIGH)}),
            [StatusEffect.ROOTS_BURNING],
        )
        self.assertEqual(
            extract_status_effects({"soil_temp": _linear(LinearState.VALUE_CRITICAL_LOW)}),
            [StatusEffect.ROOTS_FROZEN],
        )

    def test_soil_humidity_critical_high_and_low(self):
        self.assertEqual(
            extract_status_effects({"soil_humidity": _linear(LinearState.VALUE_CRITICAL_HIGH)}),
            [StatusEffect.ROOTS_ROTTING],
        )
        self.assertEqual(
            extract_status_effects({"soil_humidity": _linear(LinearState.VALUE_CRITICAL_LOW)}),
            [StatusEffect.ROOTS_PARCHED],
        )

    def test_disease_bacterial_and_fungal(self):
        self.assertEqual(
            extract_status_effects({"disease": _linear(DiseaseState.BACTERIAL)}),
            [StatusEffect.BACTERIAL_INFECTION],
        )
        self.assertEqual(
            extract_status_effects({"disease": _linear(DiseaseState.FUNGAL)}),
            [StatusEffect.FUNGAL_INFECTION],
        )

    def test_disease_healthy_has_no_effect(self):
        self.assertEqual(extract_status_effects({"disease": _linear(DiseaseState.HEALTHY)}), [])

    def test_multiple_simultaneous_effects_all_included_in_order(self):
        result = {
            "disease": _linear(DiseaseState.FUNGAL),
            "temperature": _linear(LinearState.VALUE_CRITICAL_HIGH),
            "humidity": _linear(LinearState.VALUE_CRITICAL_LOW),
            "soil_temp": _linear(LinearState.VALUE_CRITICAL_LOW),
            "soil_humidity": _linear(LinearState.VALUE_CRITICAL_HIGH),
        }
        self.assertEqual(
            extract_status_effects(result),
            [
                StatusEffect.FUNGAL_INFECTION,
                StatusEffect.SCORCHING_TEMPERATURE,
                StatusEffect.DESERT_HUMIDITY,
                StatusEffect.ROOTS_FROZEN,
                StatusEffect.ROOTS_ROTTING,
            ],
        )

    def test_status_effect_values_are_plain_korean_strings(self):
        self.assertEqual(StatusEffect.BACTERIAL_INFECTION.value, "박테리아")
        self.assertEqual(StatusEffect.FUNGAL_INFECTION.value, "곰팡이")
        self.assertIsInstance(StatusEffect.BACTERIAL_INFECTION, str)


class PlantAIGetStatusEffectTests(unittest.TestCase):
    def setUp(self):
        self.persona = build_known_persona("상추")
        self.ai = PlantAI(self.persona, llm=_NoOpLLM())

    def test_reads_last_speak_result(self):
        self.ai.speak({"temperature": 40})  # critical high
        self.assertEqual(self.ai.get_status_effect(), [StatusEffect.SCORCHING_TEMPERATURE])

    def test_empty_list_when_nothing_critical(self):
        self.ai.speak({"temperature": 20, "humidity": 60})
        self.assertEqual(self.ai.get_status_effect(), [])

    def test_raises_without_prior_speak(self):
        fresh_ai = PlantAI(build_known_persona("상추"), llm=_NoOpLLM())
        with self.assertRaises(RuntimeError):
            fresh_ai.get_status_effect()

    def test_does_not_update_persona_state(self):
        self.ai.speak({"growth": "sample_easy.jpg"})
        streak_before = self.persona.engines["growth"].count_machine.streak
        for _ in range(5):
            self.ai.get_status_effect()
        streak_after = self.persona.engines["growth"].count_machine.streak
        self.assertEqual(streak_before, streak_after)


if __name__ == "__main__":
    unittest.main()
