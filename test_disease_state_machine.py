import unittest

from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.persona.state_engine import (
    DiseaseEvents,
    DiseaseState,
    DiseaseStateMachine,
)

# 텍스트 내용 자체는 known_plant_builder.py/상추.json이 책임지므로, 여기서는 알고리즘만
# 검증한다. enum 이름을 그대로 문구로 써서 최소한의 유효한 dict만 만든다.
_STATE_PROMPTS = {s: s.name for s in DiseaseState}
_EVENT_PROMPTS = {e: e.name for e in DiseaseEvents}
_DEFAULT_DIALOG = {**_STATE_PROMPTS, **_EVENT_PROMPTS}


def _disease_machine(**kwargs) -> DiseaseStateMachine:
    return DiseaseStateMachine(
        state_prompts=_STATE_PROMPTS,
        event_prompts=_EVENT_PROMPTS,
        default_dialog=_DEFAULT_DIALOG,
        **kwargs,
    )


class _StubDiseaseAnalyzer:
    """실제 모델 없이 DiseaseStateMachine을 테스트하기 위한 가짜 DiseaseAnalyzer.
    detect_event()에 넘긴 curr_value(원본 이미지)가 그대로 analyze()로 전달되는지도 기록한다."""

    def __init__(self, labels):
        self._labels = list(labels)
        self.calls = []

    def analyze(self, image):
        self.calls.append(image)
        label = self._labels.pop(0)
        return {"label": label, "confidence": 0.9, "probabilities": {}}


class DiseaseStateMachineTests(unittest.TestCase):
    def test_first_observation_confirms_immediately(self):
        stub = _StubDiseaseAnalyzer(["healthy"])
        machine = _disease_machine(disease_analyzer=stub, confirm_streak=5)

        result = machine.detect_event("frame_001.jpg")

        self.assertEqual(stub.calls, ["frame_001.jpg"])
        self.assertEqual(result["state"], DiseaseState.HEALTHY)
        self.assertFalse(result["is_there_a_event"])
        self.assertEqual(result["raw_event"], DiseaseEvents.NO_EVENTS)

    def test_blip_below_confirm_streak_does_not_change_confirmed_state(self):
        """건강 상태였는데 한두 프레임만 오분류로 튀는 경우, confirm_streak(5)에 못 미치면
        confirmed_state가 그대로 유지되어야 한다."""
        labels = ["healthy"] * 3 + ["fungal"] * 3 + ["healthy"] * 2
        stub = _StubDiseaseAnalyzer(labels)
        machine = _disease_machine(disease_analyzer=stub, confirm_streak=5)

        result = None
        for _ in labels:
            result = machine.detect_event("frame.jpg")

        self.assertEqual(machine.confirmed_state, DiseaseState.HEALTHY)
        self.assertEqual(result["state"], DiseaseState.HEALTHY)

    def test_streak_at_confirm_threshold_confirms_transition(self):
        labels = ["healthy"] + ["bacterial"] * 5
        stub = _StubDiseaseAnalyzer(labels)
        machine = _disease_machine(disease_analyzer=stub, confirm_streak=5)

        results = [machine.detect_event("frame.jpg") for _ in labels]

        # 마지막 5개(bacterial) 중 처음 4개까지는 아직 확정 전, 5번째에 확정.
        for r in results[1:5]:
            self.assertEqual(r["state"], DiseaseState.HEALTHY)
            self.assertFalse(r["is_there_a_event"])

        confirmed = results[5]
        self.assertEqual(confirmed["state"], DiseaseState.BACTERIAL)
        self.assertTrue(confirmed["is_there_a_event"])
        self.assertEqual(confirmed["raw_event"], DiseaseEvents.HEALTHY_TO_BACTERIAL)

    def test_prompt_prefers_event_over_state_on_transition(self):
        labels = ["healthy"] + ["fungal"] * 5
        stub = _StubDiseaseAnalyzer(labels)
        machine = _disease_machine(disease_analyzer=stub, confirm_streak=5)

        for _ in labels[:-1]:
            machine.detect_event("frame.jpg")
        confirmed = machine.detect_event("frame.jpg")

        self.assertEqual(confirmed["prompt"], [confirmed["event_prompt"]])
        self.assertNotEqual(confirmed["prompt"], [confirmed["state_prompt"]])

    def test_prompt_uses_state_when_no_event(self):
        stub = _StubDiseaseAnalyzer(["healthy", "healthy"])
        machine = _disease_machine(disease_analyzer=stub, confirm_streak=5)
        machine.detect_event("frame.jpg")
        result = machine.detect_event("frame.jpg")

        self.assertFalse(result["is_there_a_event"])
        self.assertEqual(result["prompt"], [result["state_prompt"]])

    def test_get_state_and_load_state_roundtrip(self):
        labels = ["healthy"] + ["bacterial"] * 3  # 확정 안 된 candidate 상태로 저장
        stub = _StubDiseaseAnalyzer(labels)
        machine = _disease_machine(disease_analyzer=stub, confirm_streak=5)
        for _ in labels:
            machine.detect_event("frame.jpg")

        saved = machine.get_state()

        restored = _disease_machine(disease_analyzer=_StubDiseaseAnalyzer([]))
        restored.load_state(saved)

        self.assertEqual(restored.confirmed_state, machine.confirmed_state)
        self.assertEqual(restored.candidate_state, machine.candidate_state)
        self.assertEqual(restored.streak, machine.streak)

        # 저장/복원 후 이어서 넣어도 원본과 동일하게 동작해야 한다 (streak 2개 더 넣으면 확정).
        continuation = ["bacterial", "bacterial"]
        machine.disease_analyzer = _StubDiseaseAnalyzer(list(continuation))
        restored.disease_analyzer = _StubDiseaseAnalyzer(list(continuation))

        for _ in continuation:
            live_result = machine.detect_event("frame.jpg")
            restored_result = restored.detect_event("frame.jpg")
            self.assertEqual(live_result["state"], restored_result["state"])
            self.assertEqual(live_result["raw_event"], restored_result["raw_event"])


class DiseaseStateMachinePersonaIntegrationTests(unittest.TestCase):
    def test_real_pipeline_healthy_by_default(self):
        persona = build_known_persona("상추")
        result = persona.update({"disease": "sample_easy.jpg"})
        disease = result["disease"]

        self.assertIn("state", disease)
        self.assertIn(disease["state"], (DiseaseState.HEALTHY, DiseaseState.BACTERIAL, DiseaseState.FUNGAL))
        self.assertEqual(len(disease["prompt"]), 1)
        self.assertEqual(len(disease["default_dialog"]), 1)

    def test_real_pipeline_confirms_transition_after_confirm_streak(self):
        persona = build_known_persona("상추")
        persona.engines["disease"].disease_analyzer = _StubDiseaseAnalyzer(
            ["healthy"] + ["fungal"] * 5
        )

        results = [persona.update({"disease": "frame.jpg"})["disease"] for _ in range(6)]

        self.assertEqual(results[-1]["state"], DiseaseState.FUNGAL)
        self.assertTrue(results[-1]["is_there_a_event"])
        self.assertEqual(results[-1]["raw_event"], DiseaseEvents.HEALTHY_TO_FUNGAL)


if __name__ == "__main__":
    unittest.main()
