import time
from planta_gochi import LeafAnalyzer

analyzer = LeafAnalyzer("tflite")

for name in ["sample_easy.jpg", "sample_hard.jpg"]:
    start = time.perf_counter()
    result = analyzer.analyze(name)
    elapsed = time.perf_counter() - start
    print(f"[{name}] {elapsed*1000:.1f}ms | leaves: {result['leaf_count']} | total_area: {result['total_area']['ratio']:.4f}")