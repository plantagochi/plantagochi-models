"""
growth와 disease는 둘 다 "이미지를 curr_value로 받는" state machine(GrowthStateMachine/
DiseaseStateMachine)이라, 같은 사진 하나를 두 센서에 동시에 넣어도 서로 간섭 없이 각자
독립적으로 갱신되어야 한다. 또한 "어떻게 업데이트하는지"(Persona.update()에 한 dict로
넘기는 방식)는 온도/습도 같은 숫자 센서와 완전히 똑같아야 한다 — 이미지 센서라고 해서
별도 메서드나 특별 취급이 필요하면 안 된다.
"""
import io
import unittest

import numpy as np
from PIL import Image

from planta_gochi.persona.persona import Persona
from planta_gochi.persona.sheets import build_known_persona
from planta_gochi.persona.state_engine import DiseaseState


class SharedImageInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("sample_easy.jpg", "rb") as f:
            cls.image_bytes = f.read()

    def _assert_growth_and_disease_agree(self, growth_value, disease_value):
        """숫자 센서(temperature)와 이미지 센서 두 개(growth/disease)를 같은 update() 호출
        하나에 섞어 넣는다 — 이미지 센서라고 별도 호출 방식이 필요하지 않음을 보여준다."""
        persona = build_known_persona("상추")
        result = persona.update({
            "temperature": 20,
            "growth": growth_value,
            "disease": disease_value,
        })

        self.assertEqual(result["growth"]["leaf_count"]["count"], 15)
        self.assertEqual(result["disease"]["state"], DiseaseState.HEALTHY)
        self.assertEqual(result["temperature"]["state"].name, "VALUE_MID")

    def test_same_str_path_for_both_sensors(self):
        self._assert_growth_and_disease_agree("sample_easy.jpg", "sample_easy.jpg")

    def test_same_bytes_for_both_sensors(self):
        self._assert_growth_and_disease_agree(self.image_bytes, self.image_bytes)

    def test_same_numpy_array_for_both_sensors(self):
        arr = np.array(Image.open("sample_easy.jpg").convert("RGB"))
        self._assert_growth_and_disease_agree(arr, arr)

    def test_mixed_input_types_for_same_underlying_photo(self):
        """growth엔 파일 경로, disease엔 bytes로 — 표현 형식이 달라도 같은 사진이면
        결과가 같아야 하고, 두 센서를 같은 update() 호출에 자유롭게 섞을 수 있어야 한다."""
        arr = np.array(Image.open("sample_easy.jpg").convert("RGB"))
        self._assert_growth_and_disease_agree("sample_easy.jpg", self.image_bytes)
        self._assert_growth_and_disease_agree(io.BytesIO(self.image_bytes), arr)

    def test_growth_and_disease_use_independent_analyzer_instances(self):
        """같은 사진을 넣어도 growth(LeafAnalyzer)와 disease(DiseaseAnalyzer)는 서로 다른
        모델/엔진 인스턴스를 쓰므로 서로 간섭하지 않는다."""
        persona = build_known_persona("상추")
        self.assertIsNot(
            persona.engines["growth"].leaf_analyzer,
            persona.engines["disease"].disease_analyzer,
        )

        result = persona.update({"growth": "sample_easy.jpg", "disease": "sample_hard.jpg"})
        # 서로 다른 사진을 넣으면 당연히 결과도 따로 나와야 한다(간섭 없음의 방증).
        self.assertEqual(result["growth"]["leaf_count"]["count"], 15)  # sample_easy.jpg
        # disease 결과는 sample_hard.jpg 기준이라 growth와 별개로 계산된 것.
        self.assertIn(result["disease"]["state"], (DiseaseState.HEALTHY, DiseaseState.BACTERIAL, DiseaseState.FUNGAL))


class ImageAliasFanOutTests(unittest.TestCase):
    """persona.update({"image": img})가 growth/disease에 동시에 img를 넣은 것과 완전히
    같은 동작을 하는지 확인. 기존처럼 센서 이름을 직접 쓰는 방식은 그대로 유지되어야 한다."""

    def test_image_alias_matches_explicit_growth_and_disease_calls(self):
        alias_persona = build_known_persona("상추")
        alias_result = alias_persona.update({"image": "sample_easy.jpg"})

        direct_persona = build_known_persona("상추")
        direct_result = direct_persona.update({
            "growth": "sample_easy.jpg",
            "disease": "sample_easy.jpg",
        })

        self.assertEqual(set(alias_result.keys()), {"growth", "disease"})
        self.assertEqual(set(alias_result.keys()), set(direct_result.keys()))
        self.assertEqual(
            alias_result["growth"]["leaf_count"]["count"],
            direct_result["growth"]["leaf_count"]["count"],
        )
        self.assertEqual(alias_result["disease"]["state"], direct_result["disease"]["state"])

    def test_explicit_sensor_value_overrides_image_alias(self):
        """{"image": a, "growth": b}면 growth는 b를 쓰고, image 별칭은 아직 값을 안 받은
        나머지 이미지 센서(disease)에만 적용되어야 한다."""
        persona = build_known_persona("상추")
        result = persona.update({"image": "sample_easy.jpg", "growth": "sample_hard.jpg"})

        expected_growth = build_known_persona("상추").update({"growth": "sample_hard.jpg"})
        expected_disease = build_known_persona("상추").update({"disease": "sample_easy.jpg"})

        self.assertEqual(
            result["growth"]["leaf_count"]["count"],
            expected_growth["growth"]["leaf_count"]["count"],
        )
        self.assertEqual(result["disease"]["state"], expected_disease["disease"]["state"])

    def test_legacy_direct_sensor_names_still_work_unchanged(self):
        persona = build_known_persona("상추")
        result = persona.update({"temperature": 20, "growth": "sample_easy.jpg"})
        self.assertEqual(set(result.keys()), {"temperature", "growth"})

    def test_image_alias_can_mix_with_numeric_sensors(self):
        persona = build_known_persona("상추")
        result = persona.update({"temperature": 20, "image": "sample_easy.jpg"})
        self.assertEqual(set(result.keys()), {"temperature", "growth", "disease"})

    def test_image_alias_is_noop_when_no_engine_accepts_image(self):
        persona = Persona(name="테스트", engines={})
        result = persona.update({"image": "sample_easy.jpg"})
        self.assertEqual(result, {})

    def test_update_without_image_key_is_untouched(self):
        """"image" 키가 아예 없는 sensor_values는 _expand_image_alias가 손대지 않고
        그대로 돌려줘야 한다(불필요한 dict 복사도 하지 않음)."""
        persona = build_known_persona("상추")
        original = {"temperature": 20}
        expanded = persona._expand_image_alias(original)
        self.assertIs(expanded, original)


if __name__ == "__main__":
    unittest.main()
