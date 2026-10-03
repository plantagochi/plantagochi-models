import unittest

from planta_gochi.llm.connection import LLMConnection
from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.persona.state_engine import DiseaseState, LinearEvents, LinearState
from planta_gochi.persona.status_effect import (
    DISEASE_STATUS_EFFECTS,
    LINEAR_CRITICAL_STATUS_EFFECTS,
    StatusEffect,
    StatusEffectState,
    extract_every_status_effects,
    extract_status_effects,
)
from planta_gochi.plant_ai import PlantAI


def _linear(state: LinearState) -> dict:
    """실제 LinearStateMachine.detect_event() 결과 중 상태이상 추출이 실제로 읽는 부분만 흉내."""
    return {"state": state}


def _disease(state: DiseaseState) -> dict:
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


class ExtractEveryStatusEffectsTests(unittest.TestCase):
    A, C, U = StatusEffectState.ACTIVE, StatusEffectState.CLEARED, StatusEffectState.UNKNOWN

    def test_covers_every_status_effect(self):
        self.assertEqual(set(extract_every_status_effects({}).keys()), set(StatusEffect))

    def test_empty_result_is_all_unknown(self):
        self.assertEqual(set(extract_every_status_effects({}).values()), {self.U})

    def test_critical_high_observed_is_active_and_opposite_is_cleared(self):
        states = extract_every_status_effects({"temperature": _linear(LinearState.VALUE_CRITICAL_HIGH)})
        self.assertEqual(states[StatusEffect.SCORCHING_TEMPERATURE], self.A)
        self.assertEqual(states[StatusEffect.FREEZING_TEMPERATURE], self.C)

    def test_observed_normal_range_clears_both_temperature_effects(self):
        """관측했고 정상 범위면 해소(CLEARED)다 — 관측 안 됨(UNKNOWN)과 구분되어야 한다."""
        states = extract_every_status_effects({"temperature": _linear(LinearState.VALUE_MID)})
        self.assertEqual(states[StatusEffect.SCORCHING_TEMPERATURE], self.C)
        self.assertEqual(states[StatusEffect.FREEZING_TEMPERATURE], self.C)

    def test_unobserved_sensor_stays_unknown_not_cleared(self):
        states = extract_every_status_effects({"humidity": _linear(LinearState.VALUE_MID)})
        self.assertEqual(states[StatusEffect.JUNGLE_HUMIDITY], self.C)
        self.assertEqual(states[StatusEffect.ROOTS_BURNING], self.U)
        self.assertEqual(states[StatusEffect.ROOTS_ROTTING], self.U)

    def test_disease_bacterial_is_active_and_fungal_is_cleared(self):
        states = extract_every_status_effects({"disease": _disease(DiseaseState.BACTERIAL)})
        self.assertEqual(states[StatusEffect.BACTERIAL_INFECTION], self.A)
        self.assertEqual(states[StatusEffect.FUNGAL_INFECTION], self.C)

    def test_disease_healthy_clears_both_infections(self):
        states = extract_every_status_effects({"disease": _disease(DiseaseState.HEALTHY)})
        self.assertEqual(states[StatusEffect.BACTERIAL_INFECTION], self.C)
        self.assertEqual(states[StatusEffect.FUNGAL_INFECTION], self.C)

    def test_result_without_state_key_counts_as_unobserved(self):
        states = extract_every_status_effects({"temperature": {"raw_event": LinearEvents.NO_EVENTS}})
        self.assertEqual(states[StatusEffect.SCORCHING_TEMPERATURE], self.U)

    def test_existing_active_only_function_is_unchanged(self):
        result = {"temperature": _linear(LinearState.VALUE_CRITICAL_HIGH)}
        self.assertEqual(extract_status_effects(result), [StatusEffect.SCORCHING_TEMPERATURE])

    def test_public_mapping_tables_match_states(self):
        self.assertEqual(DISEASE_STATUS_EFFECTS[DiseaseState.FUNGAL], StatusEffect.FUNGAL_INFECTION)
        self.assertEqual(
            LINEAR_CRITICAL_STATUS_EFFECTS["soil_humidity"][LinearState.VALUE_CRITICAL_LOW],
            StatusEffect.ROOTS_PARCHED,
        )

    def test_persona_update_result_feeds_three_state_report(self):
        from planta_gochi.persona.sheets import build_known_persona

        persona = build_known_persona("상추")
        states = extract_every_status_effects(persona.update({"temperature": 40}))
        self.assertEqual(states[StatusEffect.SCORCHING_TEMPERATURE], self.A)
        self.assertEqual(states[StatusEffect.JUNGLE_HUMIDITY], self.U)


if __name__ == "__main__":
    unittest.main()
