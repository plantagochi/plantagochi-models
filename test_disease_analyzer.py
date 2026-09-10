import functools
import http.server
import io
import threading
import unittest

from planta_gochi.sensory import DiseaseAnalyzer

VALID_LABELS = {"bacterial", "fungal", "healthy"}


class DiseaseAnalyzerBackendTests(unittest.TestCase):
    """세 백엔드(onnx/tflite/tflite16)가 서로 합리적으로 일치하는지 확인."""

    @classmethod
    def setUpClass(cls):
        with open("sample_easy.jpg", "rb") as f:
            cls.image_bytes = f.read()

    def test_result_shape(self):
        analyzer = DiseaseAnalyzer("onnx")
        result = analyzer.analyze("sample_easy.jpg")

        self.assertEqual(set(result.keys()), {"label", "confidence", "probabilities"})
        self.assertIn(result["label"], VALID_LABELS)
        self.assertIsInstance(result["confidence"], float)
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)

        self.assertEqual(set(result["probabilities"].keys()), VALID_LABELS)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=3)
        # confidence는 label에 해당하는 확률과 같아야 하고, 셋 중 최댓값이어야 한다.
        self.assertAlmostEqual(result["confidence"], result["probabilities"][result["label"]], places=5)
        self.assertEqual(result["label"], max(result["probabilities"], key=result["probabilities"].get))

    def test_onnx_tflite_tflite16_agree_on_label(self):
        labels = {}
        for backend in ("onnx", "tflite", "tflite16"):
            analyzer = DiseaseAnalyzer(backend)
            labels[backend] = analyzer.analyze("sample_easy.jpg")["label"]
        self.assertEqual(len(set(labels.values())), 1, f"백엔드 간 라벨 불일치: {labels}")

    def test_onnx_tflite_confidence_close(self):
        onnx_result = DiseaseAnalyzer("onnx").analyze("sample_easy.jpg")
        tflite_result = DiseaseAnalyzer("tflite").analyze("sample_easy.jpg")
        self.assertAlmostEqual(onnx_result["confidence"], tflite_result["confidence"], delta=0.01)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            DiseaseAnalyzer("not_a_real_backend")


class DiseaseAnalyzerInputTypeTests(unittest.TestCase):
    """LeafAnalyzer와 같은 입력 형식을 전부 지원하는지 확인 (_image_io 공유 로직)."""

    @classmethod
    def setUpClass(cls):
        cls.analyzer = DiseaseAnalyzer("onnx")
        with open("sample_easy.jpg", "rb") as f:
            cls.image_bytes = f.read()
        cls.baseline_label = cls.analyzer.analyze("sample_easy.jpg")["label"]

        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=".")
        cls._httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
        cls._thread = threading.Thread(target=cls._httpd.serve_forever, daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls):
        cls._httpd.shutdown()

    def test_bytes(self):
        result = self.analyzer.analyze(self.image_bytes)
        self.assertEqual(result["label"], self.baseline_label)

    def test_bytearray(self):
        result = self.analyzer.analyze(bytearray(self.image_bytes))
        self.assertEqual(result["label"], self.baseline_label)

    def test_file_like(self):
        result = self.analyzer.analyze(io.BytesIO(self.image_bytes))
        self.assertEqual(result["label"], self.baseline_label)

    def test_pil_image(self):
        from PIL import Image
        result = self.analyzer.analyze(Image.open("sample_easy.jpg"))
        self.assertEqual(result["label"], self.baseline_label)

    def test_numpy_array(self):
        import numpy as np
        from PIL import Image
        arr = np.array(Image.open("sample_easy.jpg").convert("RGB"))
        result = self.analyzer.analyze(arr)
        self.assertEqual(result["label"], self.baseline_label)

    def test_http_url(self):
        url = f"http://127.0.0.1:{self._httpd.server_port}/sample_easy.jpg"
        result = self.analyzer.analyze(url)
        self.assertEqual(result["label"], self.baseline_label)


if __name__ == "__main__":
    unittest.main()
