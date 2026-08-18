import pickle
import unittest

from planta_gochi.persona.sheets import build_known_persona

_JSON_SAFE_SCALARS = (int, float, str, bool, type(None))


def _assert_json_safe(value):
    """state dict 안의 모든 값이 int/float/str/bool/None/list/dict인지(= PostgreSQL에
    그대로 넣을 수 있는지) 재귀적으로 확인한다. enum 인스턴스가 남아있으면 여기서 걸린다."""
    if isinstance(value, dict):
        for v in value.values():
            _assert_json_safe(v)
    elif isinstance(value, list):
        for v in value:
            _assert_json_safe(v)
    else:
        assert isinstance(value, _JSON_SAFE_SCALARS), f"JSON-safe하지 않은 값: {value!r} ({type(value)})"


class PersonaStatePickleTests(unittest.TestCase):
    READINGS = [
        {"temperature": 20, "humidity": 60, "soil_temp": 17, "soil_humidity": 55, "growth": "sample_easy.jpg"},
        {"temperature": 40, "humidity": 60, "soil_temp": 30, "soil_humidity": 55, "growth": "sample_hard.jpg"},
        {"temperature": 10, "humidity": 20, "soil_temp": 8, "soil_humidity": 15, "growth": "sample_easy.jpg"},
    ]

    def test_state_is_json_safe(self):
        persona = build_known_persona("상추")
        for reading in self.READINGS:
            persona.update(reading)

        state = persona.get_state()
        _assert_json_safe(state)

    def test_pickle_save_load_roundtrip_restores_values(self):
        persona = build_known_persona("상추")
        for reading in self.READINGS:
            persona.update(reading)

        # save
        state_before = persona.get_state()
        pickled = pickle.dumps(state_before)

        # load (완전히 새 파일 열듯이, 새 bytes에서 다시 읽는다)
        state_after = pickle.loads(pickled)

        self.assertEqual(state_before, state_after)

    def test_load_state_into_fresh_persona_continues_tracking_identically(self):
        """save/load가 '값만 같음'이 아니라 실제로 이어서 추적해도 동일하게 동작하는지 확인.
        같은 이력을 쌓은 persona 두 개를 만들어, 하나는 계속 이어서 쓰고 다른 하나는
        pickle 저장 -> 로드 후 이어서 써서, 같은 다음 입력에 대해 같은 결과가 나오는지 본다."""
        live = build_known_persona("상추")
        for reading in self.READINGS:
            live.update(reading)

        saved_bytes = pickle.dumps(live.get_state())

        restored = build_known_persona("상추")
        restored.load_state(pickle.loads(saved_bytes))

        next_reading = {
            "temperature": 25, "humidity": 65,
            "soil_temp": 20, "soil_humidity": 60,
            "growth": "sample_hard.jpg",
        }
        live_result = live.update(next_reading)
        restored_result = restored.update(next_reading)

        for sensor_name in next_reading:
            self.assertEqual(
                live_result[sensor_name]["prompt"],
                restored_result[sensor_name]["prompt"],
                f"{sensor_name} 센서의 저장/복원 후 동작이 이어지지 않음",
            )
            self.assertEqual(
                live_result[sensor_name]["raw_event"],
                restored_result[sensor_name]["raw_event"],
                f"{sensor_name} 센서의 저장/복원 후 이벤트 판정이 다름",
            )


if __name__ == "__main__":
    unittest.main()
