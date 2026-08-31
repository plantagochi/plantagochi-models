import unittest

from planta_gochi.persona.mood import Expression, extract_expression
from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.persona.state_engine import CountEvents, LinearState, TrendState


def _linear(state: LinearState) -> dict:
    """실제 LinearStateMachine.detect_event() 결과 중 mood 추출이 실제로 읽는 부분만 흉내."""
    return {"state": state}


def _growth(count_event=CountEvents.NO_EVENTS, canopy=TrendState.STAGNANT, leaf_size=TrendState.STAGNANT) -> dict:
    """실제 GrowthStateMachine.detect_event() 결과 중 mood 추출이 실제로 읽는 부분만 흉내."""
    return {
        "leaf_count": {"raw_event": count_event},
        "canopy_trend": {"state": canopy},
        "leaf_size_trend": {"state": leaf_size},
    }


class ExtractExpressionTests(unittest.TestCase):
    def test_all_mid_is_happy(self):
        result = {
            "temperature": _linear(LinearState.VALUE_MID),
            "humidity": _linear(LinearState.VALUE_MID),
            "soil_temp": _linear(LinearState.VALUE_MID),
            "soil_humidity": _linear(LinearState.VALUE_MID),
        }
        self.assertEqual(extract_expression(result), Expression.HAPPY)

    def test_single_critical_state_wins_over_positive_others(self):
        """상태가 우선이므로, 다른 센서가 전부 좋아도 하나가 critical low면 DISTRESSED."""
        result = {
            "temperature": _linear(LinearState.VALUE_CRITICAL_LOW),  # freezing
            "humidity": _linear(LinearState.VALUE_MID),
            "soil_temp": _linear(LinearState.VALUE_MID),
            "soil_humidity": _linear(LinearState.VALUE_MID),
        }
        self.assertEqual(extract_expression(result), Expression.DISTRESSED)

    def test_new_leaf_event_is_excited(self):
        """leaf_count는 등급(state)이 없어 이벤트(raw_event)를 mood 판단 기준으로 쓴다."""
        result = {"growth": _growth(count_event=CountEvents.INCREASED)}
        self.assertEqual(extract_expression(result), Expression.EXCITED)

    def test_growth_single_negative_signal_is_not_distressed(self):
        """부정적 신호가 1개뿐이면(캐노피만 DECLINING) DISTRESSED로 확정하지 않는다."""
        result = {"growth": _growth(canopy=TrendState.DECLINING, leaf_size=TrendState.STAGNANT)}
        self.assertNotEqual(extract_expression(result), Expression.DISTRESSED)

    def test_growth_two_negative_signals_forces_distressed(self):
        """부정적 신호가 2개 이상이면(캐노피 + 잎 크기 둘 다 DECLINING) DISTRESSED로 확정."""
        result = {"growth": _growth(canopy=TrendState.DECLINING, leaf_size=TrendState.DECLINING)}
        self.assertEqual(extract_expression(result), Expression.DISTRESSED)

    def test_growth_three_negative_signals_is_distressed(self):
        result = {
            "growth": _growth(
                count_event=CountEvents.DECREASED,
                canopy=TrendState.DECLINING,
                leaf_size=TrendState.DECLINING,
            )
        }
        self.assertEqual(extract_expression(result), Expression.DISTRESSED)

    def test_growth_growing_is_excited(self):
        result = {"growth": _growth(canopy=TrendState.GROWING, leaf_size=TrendState.GROWING)}
        self.assertEqual(extract_expression(result), Expression.EXCITED)

    def test_empty_result_defaults_to_neutral(self):
        self.assertEqual(extract_expression({}), Expression.NEUTRAL)

    def test_unknown_sensor_names_are_ignored(self):
        result = {"어떤_새_센서": _linear(LinearState.VALUE_CRITICAL_LOW)}
        self.assertEqual(extract_expression(result), Expression.NEUTRAL)

    def test_distressed_outranks_excited_across_different_sensors(self):
        result = {
            "temperature": _linear(LinearState.VALUE_CRITICAL_HIGH),  # scorching -> DISTRESSED
            "growth": _growth(count_event=CountEvents.INCREASED),      # new_leaf_joy -> EXCITED
        }
        self.assertEqual(extract_expression(result), Expression.DISTRESSED)


class ExtractExpressionIntegrationTests(unittest.TestCase):
    """synthetic dict가 아니라 실제 Persona/JSON 파이프라인을 통과한 결과로도 확인."""

    def test_real_persona_update_result_is_consumable(self):
        persona = build_known_persona("상추")
        result = persona.update({
            "temperature": 0,       # critical low -> freezing -> DISTRESSED
            "humidity": 60,         # mid -> refreshed -> HAPPY
            "soil_temp": 17,        # mid -> warm_roots -> HAPPY
            "soil_humidity": 55,    # mid -> well_watered -> HAPPY
        })
        self.assertEqual(extract_expression(result), Expression.DISTRESSED)

    def test_real_persona_all_mid_is_happy(self):
        persona = build_known_persona("상추")
        result = persona.update({
            "temperature": 20,
            "humidity": 60,
            "soil_temp": 17,
            "soil_humidity": 55,
        })
        self.assertEqual(extract_expression(result), Expression.HAPPY)


if __name__ == "__main__":
    unittest.main()
