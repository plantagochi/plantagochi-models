# LeafAnalyzer

식물 이미지에서 잎의 갯수와 면적을 분석하는 기능입니다.

## 사용법

```python
from planta_gochi.sensory import LeafAnalyzer

analyzer = LeafAnalyzer()  # 기본: ONNX 백엔드
#tflite를 쓰려면 :
#analyzer = LeafAnalyzer("tflite")
#tflite 16bit 양자화 버전을 쓰려면: 
#analyzer = LeafAnalyzer("tflite16") 
result = analyzer.analyze("plant.jpg")

print(result["leaf_count"])            # 잎 갯수
print(result["total_area"]["ratio"])   # 전체 잎 면적 비율 (0~1)
```

이미지는 다음 형식을 모두 지원합니다: 로컬 파일 경로(str/Path), http(s) URL(str),
bytes/bytearray, `.read()`를 가진 file-like 객체(`io.BytesIO`, 열린 파일 핸들, 스트리밍
응답 body 등), PIL Image, numpy array.

### 입력 형식 예시
```python
# 1. str path
result = analyzer.analyze(IMAGE)
print(f"[str path]   leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 2. Path object
result = analyzer.analyze(Path(IMAGE))
print(f"[Path obj]   leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 3. http(s) URL
result = analyzer.analyze("https://example.com/plant.jpg")
print(f"[URL]        leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 4. bytes
with open(IMAGE, "rb") as f:
    img_bytes = f.read()
result = analyzer.analyze(img_bytes)
print(f"[bytes]      leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 5. file-like 객체 (io.BytesIO, 열린 파일 핸들, S3/HTTP 스트리밍 응답 등)
import io
result = analyzer.analyze(io.BytesIO(img_bytes))
print(f"[file-like]  leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 6. PIL Image
pil_img = Image.open(IMAGE)
result = analyzer.analyze(pil_img)
print(f"[PIL Image]  leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")

# 7. numpy array
arr = np.array(Image.open(IMAGE).convert("RGB"))
result = analyzer.analyze(arr)
print(f"[numpy]      leaves={result['leaf_count']} ratio={result['total_area']['ratio']}")
```

이 모든 입력 형식은 [`ai.speak()`](plant_ai.md)의 `"growth"` 센서 값으로도 그대로 쓸 수
있습니다 — `Persona`/`GrowthStateMachine`은 값을 가공하지 않고 `LeafAnalyzer.analyze()`에
그대로 넘기기 때문입니다.

## 반환값

```python
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
```
