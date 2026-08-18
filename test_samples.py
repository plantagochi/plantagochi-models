import functools
import http.server
import io
import threading
import time
from planta_gochi.sensory import LeafAnalyzer
from pathlib import Path
import numpy as np
from PIL import Image

analyzer = LeafAnalyzer("tflite")

for name in ["sample_easy.jpg", "sample_hard.jpg"]:
    start = time.perf_counter()
    result = analyzer.analyze(name)
    elapsed = time.perf_counter() - start
    print(f"[{name}] {elapsed*1000:.1f}ms | leaves: {result['leaf_count']} | total_area: {result['total_area']['ratio']:.4f}")

"""
LeafAnalyzer 입력 타입 테스트
"""

IMAGE = "sample_easy.jpg"

results = {}

# 1. str path
result = analyzer.analyze(IMAGE)
results["str_path"] = result["leaf_count"]
print(f"[str path]   leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 2. Path object
result = analyzer.analyze(Path(IMAGE))
results["path_obj"] = result["leaf_count"]
print(f"[Path obj]   leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 3. bytes
with open(IMAGE, "rb") as f:
    img_bytes = f.read()
result = analyzer.analyze(img_bytes)
results["bytes"] = result["leaf_count"]
print(f"[bytes]      leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 4. PIL Image
pil_img = Image.open(IMAGE)
result = analyzer.analyze(pil_img)
results["pil"] = result["leaf_count"]
print(f"[PIL Image]  leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 5. numpy array
arr = np.array(Image.open(IMAGE).convert("RGB"))
result = analyzer.analyze(arr)
results["numpy"] = result["leaf_count"]
print(f"[numpy]      leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 6. bytearray
result = analyzer.analyze(bytearray(img_bytes))
results["bytearray"] = result["leaf_count"]
print(f"[bytearray]  leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 7. file-like 객체 (io.BytesIO, 열린 파일 핸들 등 .read()를 가진 것 아무거나)
result = analyzer.analyze(io.BytesIO(img_bytes))
results["bytesio"] = result["leaf_count"]
print(f"[BytesIO]    leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

with open(IMAGE, "rb") as f:
    result = analyzer.analyze(f)
    results["file_handle"] = result["leaf_count"]
    print(f"[file 핸들]  leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 8. http(s) URL (외부 네트워크 없이, 로컬 HTTP 서버로 테스트)
_handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=".")
_httpd = http.server.HTTPServer(("127.0.0.1", 0), _handler)
_thread = threading.Thread(target=_httpd.serve_forever, daemon=True)
_thread.start()
try:
    url = f"http://127.0.0.1:{_httpd.server_port}/{IMAGE}"
    result = analyzer.analyze(url)
    results["url"] = result["leaf_count"]
    print(f"[URL]        leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")
finally:
    _httpd.shutdown()

# 결과 일치 확인
counts = list(results.values())
if len(set(counts)) == 1:
    print(f"\n✅ 모든 입력 타입 일치 (leaf_count={counts[0]})")
else:
    print(f"\n⚠️  결과 불일치: {results}")