"""
Public API for leaf analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from planta_gochi.sensory._inference import OnnxBackend, TFLiteBackend

_ASSETS = Path(__file__).parent / "assets"
_DEFAULT_ONNX = _ASSETS / "best.onnx"
_DEFAULT_TFLITE = _ASSETS / "best_float32.tflite"
_DEFAULT_TFLITE16 = _ASSETS / "best_float16.tflite"

# str/Path는 로컬 파일 경로 또는 http(s) URL 둘 다 받는다. 그 외에 .read()를 갖는
# file-like 객체(io.BytesIO, 열린 파일 핸들, S3 스트리밍 응답 등)도 받는다.
ImageInput = Union[str, Path, bytes, bytearray, "PIL.Image.Image", np.ndarray] # type: ignore

class LeafAnalyzer:
    """
    Analyze plant images to count leaves and measure their area.

    Parameters
    ----------
    backend : str
        'onnx' (default) or 'tflite'
    model_path : str or Path, optional
        Custom model file path. Uses bundled model if not specified.
    conf_threshold : float
        Only masks over conf_threshold will be treated as detection output.
    mask_threshold : float
        Only pixels over probability maks_threshold will be treated as part of the mask.

    Examples
    --------
    >>> analyzer = LeafAnalyzer()
    >>> result = analyzer.analyze("plant.jpg")
    >>> print(result["leaf_count"])
    >>> print(result["total_area"]["pixels"])
    """

    def __init__(
        self,
        backend: str = "onnx",
        model_path: Union[str, Path, None] = None,
        conf_threshold: float = 0.25,
        mask_threshold: float = 0.5
    ):
        backend = backend.lower()
        if backend == "onnx":
            path = Path(model_path) if model_path else _DEFAULT_ONNX
            self._backend = OnnxBackend(path, conf_threshold, mask_threshold)
        elif backend == "tflite":
            path = Path(model_path) if model_path else _DEFAULT_TFLITE
            self._backend = TFLiteBackend(path, conf_threshold, mask_threshold)
        elif backend == "tflite16":
            path = Path(model_path) if model_path else _DEFAULT_TFLITE16
            self._backend = TFLiteBackend(path, conf_threshold, mask_threshold)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'onnx' or 'tflite'.")
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold

    def analyze(self, image: ImageInput) -> dict:
        """
        Analyze a plant image and return leaf count and area information.

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
            leaf_count  : int
            leaf_areas  : {"pixels": [...], "ratio": [...]}
            total_area  : {"pixels": int, "ratio": float}

        Example
        -------
        {
            "leaf_count": 5,
            "leaf_areas": {
                "pixels": [1024, 832, 910, 740, 680],
                "ratio":  [0.021, 0.017, 0.019, 0.015, 0.014]
            },
            "total_area": {
                "pixels": 4186,
                "ratio": 0.086
            }
        }
        """
        arr = self._to_numpy(image)
        return self._backend.run(arr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(image: ImageInput) -> np.ndarray:
        """Convert any supported input type to HWC uint8 RGB numpy array."""

        # numpy array — assume already HWC RGB
        if isinstance(image, np.ndarray):
            return _ensure_rgb(image)

        # PIL Image
        try:
            from PIL import Image as PILImage
            if isinstance(image, PILImage.Image):
                return np.array(image.convert("RGB"), dtype=np.uint8)
        except ImportError:
            pass

        # bytes (e.g. S3 response body)
        if isinstance(image, (bytes, bytearray)):
            return _decode_bytes(bytes(image))

        # http(s) URL
        if isinstance(image, str) and _is_url(image):
            return _decode_bytes(_fetch_url(image))

        # file path
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return _decode_bytes(path.read_bytes())

        # file-like object (io.BytesIO, open file handle, streaming response body, ...)
        if hasattr(image, "read"):
            return _decode_bytes(image.read())

        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            "Pass a file path, http(s) URL, bytes/bytearray, a file-like object, "
            "a PIL.Image, or a numpy array."
        )


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _fetch_url(url: str, timeout: float = 10) -> bytes:
    """http(s) URL에서 이미지 바이트를 내려받는다. requests 등 외부 의존성 없이 표준
    라이브러리 urllib만 사용한다."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def _decode_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes → HWC uint8 RGB numpy array."""
    import io
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(data)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    """Ensure numpy array is HWC uint8 RGB."""
    if arr.ndim == 2:
        # grayscale → RGB
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 4:
        # RGBA → RGB
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr