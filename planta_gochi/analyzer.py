"""
Public API for leaf analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from planta_gochi._inference import OnnxBackend, TFLiteBackend

_ASSETS = Path(__file__).parent.parent / "assets"
_DEFAULT_ONNX = _ASSETS / "best.onnx"
_DEFAULT_TFLITE = _ASSETS / "best_float32.tflite"
_DEFAULT_TFLITE16 = _ASSETS / "best_float16.tflite"

ImageInput = Union[str, Path, bytes, "PIL.Image.Image", np.ndarray] # type: ignore

class LeafAnalyzer:
    """
    Analyze plant images to count leaves and measure their area.

    Parameters
    ----------
    backend : str
        'onnx' (default) or 'tflite'
    model_path : str or Path, optional
        Custom model file path. Uses bundled model if not specified.

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
    ):
        backend = backend.lower()
        if backend == "onnx":
            path = Path(model_path) if model_path else _DEFAULT_ONNX
            self._backend = OnnxBackend(path)
        elif backend == "tflite":
            path = Path(model_path) if model_path else _DEFAULT_TFLITE
            self._backend = TFLiteBackend(path)
        elif backend == "tflite16":
            path = Path(model_path) if model_path else _DEFAULT_TFLITE16
            self._backend = TFLiteBackend(path)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Choose 'onnx' or 'tflite'.")

    def analyze(self, image: ImageInput) -> dict:
        """
        Analyze a plant image and return leaf count and area information.

        Parameters
        ----------
        image : str | Path | bytes | PIL.Image | np.ndarray
            Image to analyze. Accepts file path, raw bytes (e.g. from S3),
            PIL Image, or a numpy array (HWC, uint8, RGB).

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
            return _decode_bytes(image)

        # file path
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return _decode_bytes(path.read_bytes())

        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            "Pass a file path (str/Path), bytes, PIL.Image, or numpy array."
        )


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