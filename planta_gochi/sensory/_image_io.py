"""
"어떤 형식으로 들어오든 HWC uint8 RGB numpy array로 바꾼다"는 로직. 어떤 모델을 쓰든
상관없는 순수 입력 정규화 로직이라 LeafAnalyzer/DiseaseAnalyzer가 공유한다.
"""
from __future__ import annotations

import io
import urllib.request
from pathlib import Path
from typing import Union

import numpy as np

# str/Path는 로컬 파일 경로 또는 http(s) URL 둘 다 받는다. 그 외에 .read()를 갖는
# file-like 객체(io.BytesIO, 열린 파일 핸들, S3 스트리밍 응답 등)도 받는다.
ImageInput = Union[str, Path, bytes, bytearray, "PIL.Image.Image", np.ndarray] # type: ignore


def to_numpy(image: ImageInput) -> np.ndarray:
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
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def _decode_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes → HWC uint8 RGB numpy array."""
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
