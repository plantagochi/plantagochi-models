"""
Public API for leaf analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from planta_gochi.sensory._image_io import ImageInput, to_numpy
from planta_gochi.sensory._inference import OnnxBackend, TFLiteBackend

_ASSETS = Path(__file__).parent / "assets"
_DEFAULT_ONNX = _ASSETS / "leaf" / "best.onnx"
_DEFAULT_TFLITE = _ASSETS / "leaf" / "best_float32.tflite"
_DEFAULT_TFLITE16 = _ASSETS / "leaf" / "best_float16.tflite"

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
        arr = to_numpy(image)
        return self._backend.run(arr)
