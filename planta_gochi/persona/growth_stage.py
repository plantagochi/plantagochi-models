"""
LeafAnalyzer 원시 지표(leaf_count/canopy_ratio/leaf_size_ratio)로 성장 단계(0~5)를
추정하는 순수 함수. mood.py와 같은 위치의 해석 계층 — StateEngine/Persona는 이 개념을
모르고, GrowthStateMachine이 계산해둔 값을 받아 분류만 한다.

기준값 출처
-----------
external/1~16 폴더(각각 하나의 성장 타임라인)를 파일명 마지막 숫자로 정렬한 뒤,
0/25/50/75/100% 지점 "주변" 5장씩을 LeafAnalyzer에 통과시켜 폴더별로 평균 내고,
그걸 다시 16개 폴더에 걸쳐 평균 낸 값이다 (2026-08-31 측정, scratchpad
estimate_growth_stage_means.py).

0단계: leaf_count == 0
----------------------
잎이 하나도 안 보이면(leaf_count == 0) canopy/leaf_size 값과 무관하게 무조건 0단계로
override한다(거리 기반 분류를 거치지 않음). 1~5단계는 전부 STAGE_REFERENCE_MEANS에
등록된 평균과의 정규화 거리로 결정된다 — 등록된 것보다 훨씬 크거나 작은 값이 들어와도
그중 가장 가까운 단계로 클램프될 뿐, 범위를 벗어난 새로운 값을 반환하지 않는다.
"""

import math
from typing import Dict

# 단계별 평균. 세 지표의 스케일이 서로 달라서(leaf_count는 수십 단위, ratio는 0~1)
# 그대로 거리 비교하면 leaf_count가 지배해버리므로, 아래 _METRIC_RANGES로 정규화한 뒤
# 비교한다.
STAGE_REFERENCE_MEANS: Dict[int, Dict[str, float]] = {
    1: {"leaf_count": 21.65, "canopy_ratio": 0.3194, "leaf_size_ratio": 0.0275},
    2: {"leaf_count": 21.94, "canopy_ratio": 0.3612, "leaf_size_ratio": 0.0269},
    3: {"leaf_count": 29.36, "canopy_ratio": 0.4995, "leaf_size_ratio": 0.0381},
    4: {"leaf_count": 30.43, "canopy_ratio": 0.6177, "leaf_size_ratio": 0.0505},
    5: {"leaf_count": 31.10, "canopy_ratio": 0.6377, "leaf_size_ratio": 0.0582},
}

_METRIC_KEYS = ("leaf_count", "canopy_ratio", "leaf_size_ratio")


def _metric_range(key: str) -> float:
    values = [ref[key] for ref in STAGE_REFERENCE_MEANS.values()]
    span = max(values) - min(values)
    return span or 1.0  # 0으로 나누기 방지(사실상 발생 안 함)


_METRIC_RANGES = {key: _metric_range(key) for key in _METRIC_KEYS}


def classify_growth_stage(leaf_count: float, canopy_ratio: float, leaf_size_ratio: float) -> int:
    """
    현재 관측치를 STAGE_REFERENCE_MEANS에 등록된 모든 단계(현재 1~5)의 평균과 정규화된
    유클리드 거리로 비교해 가장 가까운 단계를 반환한다. leaf_count == 0이면 0을 반환한다
    (아래 참고).
    """
    #만약 leaf_count가 0이라면 override
    if leaf_count == 0:
        return 0

    observed = {
        "leaf_count": leaf_count,
        "canopy_ratio": canopy_ratio,
        "leaf_size_ratio": leaf_size_ratio,
    }

    best_stage = None
    best_distance = math.inf
    for stage, ref in STAGE_REFERENCE_MEANS.items():
        distance = math.sqrt(
            sum(((observed[key] - ref[key]) / _METRIC_RANGES[key]) ** 2 for key in _METRIC_KEYS)
        )
        if distance < best_distance:
            best_distance = distance
            best_stage = stage

    return best_stage
