"""
Internal inference logic for ONNX and TFLite backends.
Not part of the public API.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

CONF_THRESHOLD = 0.25
MASK_THRESHOLD = 0.5
INPUT_SIZE = 640

#preprocess backend 1(faster)
def _preprocess(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Resize and normalize image to model input.
    Returns (blob, original_h, original_w).
    image: HWC uint8 RGB numpy array
    """
    import cv2  # type: ignore # optional, fall back to PIL resize if not available

    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    blob = resized.astype(np.float32) / 255.0          # 0-1
    blob = np.transpose(blob, (2, 0, 1))               # HWC → CHW
    blob = np.expand_dims(blob, axis=0)                # CHW → BCHW
    return blob, orig_h, orig_w

#preprocess backend 2(safe)
def _preprocess_pil(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    """PIL-based resize fallback (no cv2 needed)."""
    from PIL import Image as PILImage

    orig_h, orig_w = image.shape[:2]
    pil = PILImage.fromarray(image).resize((INPUT_SIZE, INPUT_SIZE), PILImage.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    blob = np.transpose(arr, (2, 0, 1))[np.newaxis]
    return blob, orig_h, orig_w


def _safe_preprocess(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    try:
        return _preprocess(image)
    except ImportError:
        return _preprocess_pil(image)


def _postprocess(
    preds: np.ndarray,       # (1, 37, 8400)
    protos: np.ndarray,      # (1, 32, 160, 160)
    orig_h: int,
    orig_w: int,
) -> dict:
    """
    Parse YOLOv8-seg outputs into leaf count and areas.

    preds layout per anchor: [x, y, w, h, conf, m0..m31]
    """
    pred = preds[0]          # (37, 8400)
    proto = protos[0]        # (32, 160, 160)

    conf = pred[4]           # (8400,)
    mask_coefs = pred[5:]    # (32, 8400)

    keep = conf > CONF_THRESHOLD
    if not keep.any():
        return _empty_result()

    conf_keep = conf[keep]
    coefs_keep = mask_coefs[:, keep].T   # (N, 32)

    # Compute masks: (N, 160*160)
    proto_flat = proto.reshape(32, -1)                       # (32, 160*160)
    masks_flat = 1 / (1 + np.exp(-(coefs_keep @ proto_flat)))  # sigmoid, (N, 160*160)
    masks = masks_flat.reshape(-1, 160, 160)                 # (N, 160, 160)

    # Upscale masks to original image size using nearest-neighbour
    scale_h = orig_h / 160
    scale_w = orig_w / 160
    yi = (np.arange(orig_h) / scale_h).astype(int).clip(0, 159)
    xi = (np.arange(orig_w) / scale_w).astype(int).clip(0, 159)
    masks_full = masks[:, yi][:, :, xi]  # (N, orig_h, orig_w)

    binary = masks_full > MASK_THRESHOLD  # (N, orig_h, orig_w)

    # NMS-lite: remove masks that are >80% overlapping with a higher-conf mask
    binary = _soft_dedup(binary, conf_keep)

    total_pixels = orig_h * orig_w
    pixel_areas = [int(m.sum()) for m in binary]
    ratio_areas = [round(p / total_pixels, 6) for p in pixel_areas]

    total_px = sum(pixel_areas)
    return {
        "leaf_count": len(pixel_areas),
        "leaf_areas": {
            "pixels": pixel_areas,
            "ratio": ratio_areas,
        },
        "total_area": {
            "pixels": total_px,
            "ratio": round(total_px / total_pixels, 6),
        },
    }


def _soft_dedup(binary: np.ndarray, confs: np.ndarray, iou_thresh: float = 0.8) -> np.ndarray:
    """Remove duplicate masks greedily by confidence (highest first)."""
    order = np.argsort(-confs)
    binary = binary[order]
    keep = []
    suppressed = np.zeros(len(binary), dtype=bool)
    for i in range(len(binary)):
        if suppressed[i]:
            continue
        keep.append(i)
        for j in range(i + 1, len(binary)):
            if suppressed[j]:
                continue
            inter = (binary[i] & binary[j]).sum()
            union = (binary[i] | binary[j]).sum()
            if union > 0 and inter / union > iou_thresh:
                suppressed[j] = True
    return binary[keep]


def _empty_result() -> dict:
    return {
        "leaf_count": 0,
        "leaf_areas": {"pixels": [], "ratio": []},
        "total_area": {"pixels": 0, "ratio": 0.0},
    }


# ---------------------------------------------------------------------------
# ONNX backend
# ---------------------------------------------------------------------------

class OnnxBackend:
    def __init__(self, model_path: Path):
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def run(self, image: np.ndarray) -> dict:
        blob, orig_h, orig_w = _safe_preprocess(image)
        outputs = self._session.run(None, {self._input_name: blob})
        preds, protos = outputs[0], outputs[1]
        return _postprocess(preds, protos, orig_h, orig_w)


# ---------------------------------------------------------------------------
# TFLite backend
# ---------------------------------------------------------------------------

class TFLiteBackend:
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
        self._output_details = self._interp.get_output_details()

    def run(self, image: np.ndarray) -> dict:
        blob, orig_h, orig_w = _safe_preprocess(image)
        # TFLite expects BHWC
        blob_hwc = np.transpose(blob, (0, 2, 3, 1))
        self._interp.set_tensor(self._input_idx, blob_hwc)
        self._interp.invoke()

        outputs = [self._interp.get_tensor(d["index"]) for d in self._output_details]
        # outputs[0]: (1, 37, 8400), outputs[1]: (1, 160, 160, 32) → transpose to (1,32,160,160)
        preds = outputs[0]
        protos_raw = outputs[1]
        if protos_raw.ndim == 4 and protos_raw.shape[-1] == 32:
            protos = np.transpose(protos_raw, (0, 3, 1, 2))
        else:
            protos = protos_raw
        return _postprocess(preds, protos, orig_h, orig_w)