# planta-gochi-models

식물 이미지에서 잎의 갯수와 면적을 분석하는 라이브러리입니다.

## 설치

```bash
pip install git+https://github.com/your-org/planta-gochi-models.git
```

## 사용법

```python
from planta_gochi import LeafAnalyzer

analyzer = LeafAnalyzer()  # 기본: ONNX 백엔드
#tflite를 쓰려면 :
#analyzer = LeafAnalyzer("tflite")
#tflite 16bit 양자화 버전을 쓰려면: 
#analyzer = LeafAnalyzer("tflite16") 
result = analyzer.analyze("plant.jpg")

print(result["leaf_count"])            # 잎 갯수
print(result["total_area"]["ratio"])   # 전체 잎 면적 비율 (0~1)
```

이미지는 파일 경로(str/Path), bytes, PIL Image, numpy array 모두 지원합니다.

### 입력 형식 예시
```python
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
```

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