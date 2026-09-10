# DiseaseAnalyzer

식물 잎 이미지가 "bacterial"(세균성) / "fungal"(진균성) / "healthy"(건강) 중 어디에
속하는지 분류하는 기능입니다. `LeafAnalyzer`(YOLOv8-seg)와 달리 bounding box/mask를
반환하지 않는 단순 3-클래스 분류기(YOLOv8n-cls)라, 결과도 그만큼 단순합니다.

## 사용법

```python
from planta_gochi.sensory import DiseaseAnalyzer

analyzer = DiseaseAnalyzer()  # 기본: ONNX 백엔드
#tflite를 쓰려면 :
#analyzer = DiseaseAnalyzer("tflite")
#tflite 16bit 양자화 버전을 쓰려면:
#analyzer = DiseaseAnalyzer("tflite16")
result = analyzer.analyze("leaf.jpg")

print(result["label"])         # "healthy"
print(result["confidence"])    # 0.9978 (label의 확률)
```

이미지 입력 형식은 `LeafAnalyzer`와 완전히 동일합니다(내부적으로 같은 입력 정규화
로직을 공유합니다): 로컬 파일 경로(str/Path), http(s) URL(str), bytes/bytearray,
`.read()`를 가진 file-like 객체(`io.BytesIO`, 열린 파일 핸들, 스트리밍 응답 body 등),
PIL Image, numpy array. 자세한 예시는 [LeafAnalyzer 문서](leaf_analyzer.md#입력-형식-예시)를
참고하세요 — `analyzer.analyze(...)`만 `DiseaseAnalyzer`로 바꾸면 그대로 씁니다.

## 반환값

```python
{
    "label": "healthy",
    "confidence": 0.9978,
    "probabilities": {
        "bacterial": 0.0019,
        "fungal": 0.0003,
        "healthy": 0.9978
    }
}
```

`label`은 `probabilities` 중 가장 확률이 높은 것과 항상 같고(`confidence`는 그
확률값), `probabilities`의 세 값은 합이 1입니다. 세 클래스 이름은 모델 메타데이터
(`{0: 'Bacterial', 1: 'fungal', 2: 'healthy'}`)의 대소문자를 소문자로 통일한
`"bacterial"`/`"fungal"`/`"healthy"`입니다.

## LeafAnalyzer와의 차이

|  | LeafAnalyzer | DiseaseAnalyzer |
|---|---|---|
| 모델 | YOLOv8-seg (segmentation) | YOLOv8n-cls (classification) |
| 입력 크기 | 640×640 | 224×224 |
| 출력 | 잎 개수 + 잎마다 bounding box/mask 면적 | 이미지 전체에 대한 단일 라벨 |
| `conf_threshold`/`mask_threshold` | 있음 (검출 개수를 거르는 데 씀) | 없음 (분류기라 항상 결과 하나) |

내부적으로 모델 아키텍처가 완전히 달라서(전처리 크기, 후처리 방식 모두 다름) 두
분석기는 별도의 inference 모듈(`_inference.py` / `_disease_inference.py`)을 쓰지만,
"아무 형식의 이미지를 받아 numpy array로 바꾸는" 입력 정규화 로직(`_image_io.py`)은
공유합니다.
