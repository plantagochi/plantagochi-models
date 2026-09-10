"""
Internal inference logic for the disease classifier (YOLOv8n-cls, ONNX/TFLite).
Not part of the public API.

leaf 모델(YOLOv8-seg, 640 입력, bounding box/mask 출력)과는 아예 다른 아키텍처라
_inference.py와 분리했다 — 이쪽은 224 입력에 3-클래스 확률만 뱉는 단순 분류기.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

INPUT_SIZE = 224

# 모델(ultralytics YOLOv8n-cls) 메타데이터에 박혀있는 클래스 순서 그대로.
# 원본 metadata는 {0: 'Bacterial', 1: 'fungal', 2: 'healthy'}로 대소문자가 섞여
# 있어서 소문자로 통일했다.
CLASS_NAMES = ["bacterial", "fungal", "healthy"]


def _preprocess(image: np.ndarray) -> np.ndarray:
    """Resize/normalize image to model input. Returns CHW blob (1,3,224,224)."""
    import cv2  # type: ignore # optional, fall back to PIL resize if not available

    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    blob = resized.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)
    return blob


def _preprocess_pil(image: np.ndarray) -> np.ndarray:
    """PIL-based resize fallback (no cv2 needed)."""
    from PIL import Image as PILImage

    pil = PILImage.fromarray(image).resize((INPUT_SIZE, INPUT_SIZE), PILImage.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    blob = np.transpose(arr, (2, 0, 1))[np.newaxis]
    return blob


def _safe_preprocess(image: np.ndarray) -> np.ndarray:
    try:
        return _preprocess(image)
    except ImportError:
        return _preprocess_pil(image)


def _postprocess(probs: np.ndarray) -> dict:
    """probs: (1, 3) 모델이 이미 softmax까지 적용해서 내놓는 확률(합=1). 별도 softmax 불필요."""
    row = probs[0]
    index = int(np.argmax(row))
    return {
        "label": CLASS_NAMES[index],
        "confidence": float(row[index]),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, row)},
    }


# ---------------------------------------------------------------------------
# ONNX backend
# ---------------------------------------------------------------------------

class DiseaseOnnxBackend:
    def __init__(self, model_path: Path):
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def run(self, image: np.ndarray) -> dict:
        blob = _safe_preprocess(image)
        outputs = self._session.run(None, {self._input_name: blob})
        return _postprocess(outputs[0])


# ---------------------------------------------------------------------------
# TFLite backend
# ---------------------------------------------------------------------------

class DiseaseTFLiteBackend:
    def __init__(self, model_path: Path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter # type: ignore
            except ImportError:
                raise ImportError(
                    "TFLite backend requires 'ai-edge-litert' or 'tflite-runtime'. "
                    "Install with: pip install planta-gochi[tflite]"
                )
        self._interp = Interpreter(model_path=str(model_path))
        self._interp.allocate_tensors()
        self._input_idx = self._interp.get_input_details()[0]["index"]
        self._output_idx = self._interp.get_output_details()[0]["index"]

    def run(self, image: np.ndarray) -> dict:
        blob = _safe_preprocess(image)               # (1,3,224,224) CHW
        blob_hwc = np.transpose(blob, (0, 2, 3, 1))   # TFLite expects BHWC
        self._interp.set_tensor(self._input_idx, blob_hwc)
        self._interp.invoke()
        output = self._interp.get_tensor(self._output_idx)
        return _postprocess(output)
