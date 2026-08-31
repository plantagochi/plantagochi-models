import unittest

from planta_gochi.llm.connection import LLMConnection
from planta_gochi.persona.mood import Expression
from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.plant_ai import PlantAI


class _NoOpLLM(LLMConnection):
    """실제 네트워크를 타지 않는 가짜 LLM. 항상 실패로 간주되어 default_dialog로 대체된다."""

    def ask_batch(self, species, situations):
        return None


class PlantAIMoodAndStageTests(unittest.TestCase):
    def setUp(self):
        self.persona = build_known_persona("상추")
        self.ai = PlantAI(self.persona, llm=_NoOpLLM())

    def test_get_mood_reads_last_speak_result(self):
        self.ai.speak({"temperature": 20, "humidity": 60})
        mood = self.ai.get_mood()
        self.assertIsInstance(mood, Expression)
        self.assertEqual(mood, Expression.HAPPY)

    def test_get_growth_stage_reads_last_speak_result(self):
        self.ai.speak({"growth": "sample_easy.jpg"})
        stage = self.ai.get_growth_stage()
        self.assertIn(stage, (1, 2, 3, 4))

    def test_get_growth_stage_without_growth_sensor_is_none(self):
        self.ai.speak({"temperature": 20, "humidity": 60})
        stage = self.ai.get_growth_stage()
        self.assertIsNone(stage)

    def test_calling_without_prior_speak_raises(self):
        fresh_ai = PlantAI(build_known_persona("상추"), llm=_NoOpLLM())
        with self.assertRaises(RuntimeError):
            fresh_ai.get_mood()
        with self.assertRaises(RuntimeError):
            fresh_ai.get_growth_stage()

    def test_get_mood_and_get_growth_stage_never_update_persona_state(self):
        """get_mood()/get_growth_stage()는 순수 read다 — speak() 없이는 절대
        persona.update()를 호출하지 않는다. leaf_count의 debounce streak처럼 호출
        횟수에 비례해 누적되는 값으로, 여러 번 불러도 안 늘어나는지 직접 확인한다."""
        self.ai.speak({"growth": "sample_easy.jpg"})
        streak_after_speak = self.persona.engines["growth"].count_machine.streak

        for _ in range(5):
            self.ai.get_mood()
            self.ai.get_growth_stage()

        streak_after_reads = self.persona.engines["growth"].count_machine.streak
        self.assertEqual(streak_after_speak, streak_after_reads)

    def test_speak_still_returns_list_of_str(self):
        messages = self.ai.speak({"temperature": 20, "humidity": 60})
        self.assertIsInstance(messages, list)
        self.assertTrue(all(isinstance(m, str) for m in messages))


if __name__ == "__main__":
    unittest.main()
