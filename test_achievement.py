import datetime
import unittest

from planta_gochi.persona.achievement import AchievementTracker
from planta_gochi.persona.persona import Persona
from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.persona.state_engine import (
    DiseaseEvents,
    DiseaseState,
    LinearEvents,
    LinearState,
    StateEngine,
)
from planta_gochi.persona.supported_achivements import SupportedAchivements
from planta_gochi.plant_ai import PlantAI


def _linear(raw_event=None, state=None) -> dict:
    """실제 LinearStateMachine.detect_event() 결과 중 AchievementTracker가 실제로 읽는
    raw_event/state만 흉내낸 스텁."""
    result = {}
    if raw_event is not None:
        result["raw_event"] = raw_event
    if state is not None:
        result["state"] = state
    return result


def _disease(raw_event=None, state=None) -> dict:
    result = {}
    if raw_event is not None:
        result["raw_event"] = raw_event
    if state is not None:
        result["state"] = state
    return result


def _growth(stage: int, canopy_ratio: float = 0.1) -> dict:
    return {
        "stage": stage,
        "raw_metrics": {"leaf_count": 10, "canopy_ratio": canopy_ratio, "leaf_size_ratio": 0.05},
    }


class AchievementTrackerUnitTests(unittest.TestCase):
    def setUp(self):
        self.tracker = AchievementTracker()

    def test_starts_with_no_achievements(self):
        self.assertEqual(self.tracker.unlocked(), [])

    def test_fungal_cure_unlocks_on_event(self):
        self.tracker.observe({}, {"disease": _disease(raw_event=DiseaseEvents.FUNGAL_TO_HEALTHY)})
        self.assertIn(SupportedAchivements.FUNGAL_CURE, self.tracker.unlocked())

    def test_bacterial_cure_unlocks_on_event(self):
        self.tracker.observe({}, {"disease": _disease(raw_event=DiseaseEvents.BACTERIAL_TO_HEALTHY)})
        self.assertIn(SupportedAchivements.BACTERIAL_CURE, self.tracker.unlocked())

    def test_disease_no_event_unlocks_nothing(self):
        self.tracker.observe({}, {"disease": _disease(raw_event=DiseaseEvents.NO_EVENTS)})
        self.assertEqual(self.tracker.unlocked(), [])

    def test_growth_stage_five_unlocks(self):
        self.tracker.observe({}, {"growth": _growth(stage=5)})
        self.assertIn(SupportedAchivements.GROWTH_STAGE_FIVE, self.tracker.unlocked())

    def test_growth_stage_below_five_does_not_unlock_stage_five(self):
        self.tracker.observe({}, {"growth": _growth(stage=4)})
        self.assertNotIn(SupportedAchivements.GROWTH_STAGE_FIVE, self.tracker.unlocked())

    def test_growth_stage_skip_unlocks_when_jumping_two_or_more(self):
        """알려진 한계(achievement.py의 GROWTH_STAGE_SKIP 문서 참고): stage는
        TrendStateMachine과 달리 스무딩을 거치지 않는 raw 값이라, 이 테스트처럼 단
        두 틱만으로도(디바운스 없이) 달성된다 — 실제로는 세그멘테이션 노이즈만으로도
        같은 일이 벌어질 수 있다는 뜻이다. 지금은 그대로 두기로 확인했다
        (2026-09-29 대화)."""
        self.tracker.observe({}, {"growth": _growth(stage=1)})
        self.tracker.observe({}, {"growth": _growth(stage=3)})
        self.assertIn(SupportedAchivements.GROWTH_STAGE_SKIP, self.tracker.unlocked())

    def test_growth_stage_single_step_does_not_unlock_skip(self):
        self.tracker.observe({}, {"growth": _growth(stage=1)})
        self.tracker.observe({}, {"growth": _growth(stage=2)})
        self.assertNotIn(SupportedAchivements.GROWTH_STAGE_SKIP, self.tracker.unlocked())

    def test_touch_grass_unlocks_at_half_canopy_ratio(self):
        self.tracker.observe({}, {"growth": _growth(stage=3, canopy_ratio=0.5)})
        self.assertIn(SupportedAchivements.TOUCH_GRASS, self.tracker.unlocked())

    def test_touch_grass_not_triggered_below_threshold(self):
        self.tracker.observe({}, {"growth": _growth(stage=3, canopy_ratio=0.49)})
        self.assertNotIn(SupportedAchivements.TOUCH_GRASS, self.tracker.unlocked())

    def test_sudden_environment_shift_unlocks_on_two_band_jump(self):
        # C_LOW_TO_HIGH: prev=1(critical low) -> curr=4(high), band 차이 3.
        self.tracker.observe({}, {"temperature": _linear(raw_event=LinearEvents.C_LOW_TO_HIGH)})
        self.assertIn(SupportedAchivements.SUDDEN_ENVIRONMENT_SHIFT, self.tracker.unlocked())

    def test_adjacent_band_change_does_not_unlock_sudden_shift(self):
        # LOW_TO_MID: prev=2 -> curr=3, band 차이 1.
        self.tracker.observe({}, {"temperature": _linear(raw_event=LinearEvents.LOW_TO_MID)})
        self.assertNotIn(SupportedAchivements.SUDDEN_ENVIRONMENT_SHIFT, self.tracker.unlocked())

    def test_high_humidity_unlocks_on_critical_high_state(self):
        self.tracker.observe({}, {"humidity": _linear(state=LinearState.VALUE_CRITICAL_HIGH)})
        self.assertIn(SupportedAchivements.HIGH_HUMIDITY, self.tracker.unlocked())

    def test_mild_high_humidity_does_not_unlock(self):
        self.tracker.observe({}, {"humidity": _linear(state=LinearState.VALUE_HIGH)})
        self.assertNotIn(SupportedAchivements.HIGH_HUMIDITY, self.tracker.unlocked())

    def test_status_recovery_trio_needs_three_distinct_effect_types(self):
        # temperature critical-high 복구(SCORCHING) + humidity critical-low 복구(DESERT)
        # 두 종류뿐이면 아직 부족하다.
        self.tracker.observe(
            {},
            {
                "temperature": _linear(raw_event=LinearEvents.C_HIGH_TO_MID),
                "humidity": _linear(raw_event=LinearEvents.C_LOW_TO_MID),
            },
        )
        self.assertNotIn(SupportedAchivements.STATUS_RECOVERY_TRIO, self.tracker.unlocked())

        # 같은 temperature 복구를 반복해도(distinct type 아님) 여전히 2종류뿐.
        self.tracker.observe({}, {"temperature": _linear(raw_event=LinearEvents.C_HIGH_TO_MID)})
        self.assertNotIn(SupportedAchivements.STATUS_RECOVERY_TRIO, self.tracker.unlocked())

        # 세 번째 distinct type(disease 복구)이 나오면 달성.
        self.tracker.observe({}, {"disease": _disease(raw_event=DiseaseEvents.BACTERIAL_TO_HEALTHY)})
        self.assertIn(SupportedAchivements.STATUS_RECOVERY_TRIO, self.tracker.unlocked())

    def test_recovery_into_critical_state_is_not_counted_as_recovery(self):
        """C_LOW_TO_C_HIGH처럼 critical에서 critical로 옮겨간 건 "정상 복구"가 아니다."""
        self.tracker.observe({}, {"temperature": _linear(raw_event=LinearEvents.C_LOW_TO_C_HIGH)})
        self.tracker.observe({}, {"humidity": _linear(raw_event=LinearEvents.C_HIGH_TO_C_LOW)})
        self.tracker.observe({}, {"soil_temp": _linear(raw_event=LinearEvents.C_LOW_TO_C_HIGH)})
        self.assertNotIn(SupportedAchivements.STATUS_RECOVERY_TRIO, self.tracker.unlocked())

    def test_night_owl_farmer_unlocks_when_soil_humidity_rises_within_window(self):
        tracker = AchievementTracker(now_fn=lambda: datetime.datetime(2026, 1, 1, 2, 30))
        tracker.observe({"soil_humidity": 40}, {})
        tracker.observe({"soil_humidity": 55}, {})
        self.assertIn(SupportedAchivements.NIGHT_OWL_FARMER, tracker.unlocked())

    def test_night_owl_farmer_not_triggered_outside_window(self):
        tracker = AchievementTracker(now_fn=lambda: datetime.datetime(2026, 1, 1, 10, 0))
        tracker.observe({"soil_humidity": 40}, {})
        tracker.observe({"soil_humidity": 55}, {})
        self.assertNotIn(SupportedAchivements.NIGHT_OWL_FARMER, tracker.unlocked())

    def test_night_owl_farmer_not_triggered_when_soil_humidity_drops(self):
        tracker = AchievementTracker(now_fn=lambda: datetime.datetime(2026, 1, 1, 2, 30))
        tracker.observe({"soil_humidity": 55}, {})
        tracker.observe({"soil_humidity": 40}, {})
        self.assertNotIn(SupportedAchivements.NIGHT_OWL_FARMER, tracker.unlocked())

    def test_night_owl_farmer_needs_a_previous_reading_to_compare(self):
        """첫 관측치는 "상승"을 판단할 기준이 없으므로 그 자체로는 달성되지 않는다."""
        tracker = AchievementTracker(now_fn=lambda: datetime.datetime(2026, 1, 1, 2, 30))
        tracker.observe({"soil_humidity": 90}, {})
        self.assertNotIn(SupportedAchivements.NIGHT_OWL_FARMER, tracker.unlocked())

    def test_unlocked_achievements_never_disappear(self):
        """한 번 달성한 건 그 뒤 조건이 더 이상 사실이 아니어도 계속 남아있다."""
        self.tracker.observe({}, {"growth": _growth(stage=5)})
        self.assertIn(SupportedAchivements.GROWTH_STAGE_FIVE, self.tracker.unlocked())

        self.tracker.observe({}, {"growth": _growth(stage=1)})
        self.assertIn(SupportedAchivements.GROWTH_STAGE_FIVE, self.tracker.unlocked())

    def test_unlocked_order_is_fixed_by_enum_definition_order(self):
        """관찰 순서와 무관하게 SupportedAchivements 정의 순서로 나온다."""
        self.tracker.observe({}, {"growth": _growth(stage=5)})
        self.tracker.observe({}, {"disease": _disease(raw_event=DiseaseEvents.FUNGAL_TO_HEALTHY)})

        result = self.tracker.unlocked()
        expected_order = [a for a in SupportedAchivements if a in result]
        self.assertEqual(result, expected_order)

    def test_unlocked_is_a_pure_read(self):
        """get 계열답게, 몇 번을 불러도 내부 상태를 바꾸지 않고 항상 같은 값을 돌려준다."""
        self.tracker.observe({}, {"growth": _growth(stage=5)})
        first = self.tracker.unlocked()
        second = self.tracker.unlocked()
        self.assertEqual(first, second)


class PersonaAchievementIntegrationTests(unittest.TestCase):
    def test_update_result_always_includes_achievements_key(self):
        persona = build_known_persona("상추")
        result = persona.update({"temperature": 20, "humidity": 60})
        self.assertIn("achievements", result)
        self.assertEqual(result["achievements"], [])

    def test_get_achievements_matches_update_without_changing_state(self):
        persona = build_known_persona("상추")
        persona.update({"temperature": 40})  # 극단 고온
        persona.update({"temperature": 20})  # 정상으로 복구(critical -> mid): 온도 복구 1종

        via_get = persona.get_achievements()
        via_get_again = persona.get_achievements()
        self.assertEqual(via_get, via_get_again)

        result = persona.update({"temperature": 20})  # 같은 값 반복: 이벤트 없음
        self.assertEqual(result["achievements"], persona.get_achievements())

    def test_achievements_are_not_part_of_get_state_or_load_state(self):
        """휘발성 요구사항: get_state()/load_state()에는 achievement가 전혀 등장하지 않는다."""
        persona = build_known_persona("상추")
        persona.update({"temperature": 40})
        persona.update({"temperature": 20})  # 복구 이벤트 발생시켜 뭔가는 달성되게 함
        self.assertNotEqual(persona.get_achievements(), [])

        state = persona.get_state()
        self.assertNotIn("achievements", state)
        for sensor_state in state.values():
            self.assertNotIn("achievements", sensor_state)

    def test_reloading_persona_state_resets_achievements(self):
        persona = build_known_persona("상추")
        persona.update({"temperature": 40})
        persona.update({"temperature": 20})
        self.assertNotEqual(persona.get_achievements(), [])

        fresh = build_known_persona("상추")
        fresh.load_state(persona.get_state())
        self.assertEqual(fresh.get_achievements(), [])


class _StubLLM:
    def ask_batch(self, species, situations):
        return None


class PlantAIAchievementTests(unittest.TestCase):
    def test_get_achievements_does_not_require_speak_first(self):
        persona = build_known_persona("상추")
        ai = PlantAI(persona, llm=_StubLLM())
        self.assertEqual(ai.get_achievements(), [])

    def test_get_achievements_reflects_speak_driven_updates(self):
        """PlantAI.get_achievements()가 실제로 persona.get_achievements()를 그대로
        위임하는지(speak()가 부른 persona.update()로 쌓인 값이 그대로 보이는지) 확인."""
        persona = build_known_persona("상추")
        ai = PlantAI(persona, llm=_StubLLM())
        ai.speak({"temperature": 40})  # 극단 고온
        ai.speak({"temperature": 20})  # 정상 복구
        self.assertEqual(ai.get_achievements(), persona.get_achievements())
        self.assertNotEqual(ai.get_achievements(), [])

    def test_speak_still_works_with_achievements_key_present(self):
        """achievements 키가 섞여 있어도 speak()의 situations/messages 조립이 깨지지
        않아야 한다(센서가 아닌 키를 걸러내는 로직 회귀 테스트)."""
        persona = build_known_persona("상추")
        ai = PlantAI(persona, llm=_StubLLM())
        messages = ai.speak({"temperature": 20, "humidity": 60})
        self.assertIsInstance(messages, list)
        for message in messages:
            self.assertIsInstance(message, str)


if __name__ == "__main__":
    unittest.main()
