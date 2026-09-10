"""
Public API for plant disease classification.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from planta_gochi.sensory._disease_inference import DiseaseOnnxBackend, DiseaseTFLiteBackend
from planta_gochi.sensory._image_io import ImageInput, to_numpy

_ASSETS = Path(__file__).parent / "assets"
_DEFAULT_ONNX = _ASSETS / "disease" / "best.onnx"
_DEFAULT_TFLITE = _ASSETS / "disease" / "best_float32.tflite"
_DEFAULT_TFLITE16 = _ASSETS / "disease" / "best_float16.tflite"

class DiseaseAnalyzer:
    """
    Classify a plant leaf image's disease state.

    LeafAnalyzer와 같은 사용 방식(onnx/tflite/tflite16 백엔드, 같은 입력 형식 전부 지원)을
    따르지만, 이 모델은 bounding box/segmentation이 아니라 이미지 전체에 대한 3-클래스
    분류기(YOLOv8n-cls)라 결과도 그만큼 단순하다 — "bacterial"/"fungal"/"healthy" 중
    하나 + 확신도.

    Parameters
    ----------
    backend : str
        'onnx' (default), 'tflite', or 'tflite16'
    model_path : str or Path, optional
        Custom model file path. Uses bundled model if not specified.

    Examples
    --------
    >>> analyzer = DiseaseAnalyzer()
    >>> result = analyzer.analyze("leaf.jpg")
    >>> print(result["label"])        # "healthy"
    >>> print(result["confidence"])   # 0.9946
    """

    def __init__(
        self,
        backend: str = "onnx",
        model_path: Union[str, Path, None] = None,
    ):
        backend = backend.lower()
        if backend == "onnx":
            path = Path(model_path) if model_path else _DEFAULT_ONNX
            self._backend = DiseaseOnnxBackend(path)
        elif backend == "tflite":
            path = Path(model_path) if model_path else _DEFAULT_TFLITE
            self._backend = DiseaseTFLiteBackend(path)
        elif backend == "tflite16":
            path = Path(model_path) if model_path else _DEFAULT_TFLITE16
            self._backend = DiseaseTFLiteBackend(path)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'onnx' or 'tflite'.")

    def analyze(self, image: ImageInput) -> dict:
        """
        Classify a plant image's disease state.

        Parameters
        ----------
        image : str | Path | bytes | bytearray | file-like | PIL.Image | np.ndarray
            Image to analyze. Accepts a local file path, an http(s) URL (str),
            raw bytes/bytearray (e.g. from S3), any file-like object with a
            .read() method (io.BytesIO, an open file handle, a streaming
            response body, ...), a PIL Image, or a numpy array (HWC, uint8, RGB).

        Returns
        -------
        dict with keys:
            label         : "bacterial" | "fungal" | "healthy"
            confidence    : float (0~1, label의 확률)
            probabilities : {"bacterial": float, "fungal": float, "healthy": float}

        Example
        -------
        {
            "label": "healthy",
            "confidence": 0.9946,
            "probabilities": {"bacterial": 0.0049, "fungal": 0.0004, "healthy": 0.9946}
        }
        """
        arr = to_numpy(image)
        return self._backend.run(arr)
