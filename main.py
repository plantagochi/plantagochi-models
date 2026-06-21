from planta_gochi import LeafAnalyzer

analyzer = LeafAnalyzer("onnx")
result = analyzer.analyze("sample.jpg")

print(result)