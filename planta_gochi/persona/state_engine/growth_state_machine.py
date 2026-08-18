from __future__ import annotations

import statistics
from typing import Any, Optional

from planta_gochi.persona.state_engine import DiscreteCountMachine, StateEngine, TrendStateMachine
from planta_gochi.sensory import LeafAnalyzer


class GrowthStateMachine(StateEngine):
    """
    LeafAnalyzer(YOLO 기반 vision model) + TrendStateMachine(캐노피 전체 크기 / 개별 잎 크기
    추세) x2 + DiscreteCountMachine(잎 개수)을 하나로 묶은 합성 state machine.

    다른 dimension(온도/습도)의 LinearStateMachine이 raw 센서 값(숫자)을 detect_event()로
    받아서 그 안에서 알아서 내부 state를 갱신하는 것과 똑같은 사용 패턴을, vision
    파이프라인에도 그대로 적용한다. 즉 LeafAnalyzer의 결과(JSON)를 외부에서 미리 계산해
    넘겨받는 대신, 카메라 프레임(이미지) 자체를 detect_event()의 curr_value로 받아 내부에서
    LeafAnalyzer.analyze()를 직접 돌리고, 그 결과를 다시 세 하위 state machine에 먹여
    상태를 갱신한다.

    세 하위 state machine의 결과는 각각 leaf_count/canopy_trend/leaf_size_trend로 그대로
    노출하고, 최상위 prompt/default_dialog는 셋의 메시지를 이어붙이지 않고 list로 모아
    제공한다(각 하위 엔진의 prompt/default_dialog가 이미 1개짜리 list이므로 그대로 연결).
    """

    def __init__(
        self,
        count_machine: DiscreteCountMachine,
        canopy_machine: TrendStateMachine,
        leaf_size_machine: TrendStateMachine,
        leaf_analyzer: Optional[LeafAnalyzer] = None,
    ):
        # count_machine/canopy_machine/leaf_size_machine은 LinearStateMachine과 마찬가지로
        # 문구를 내장하지 않으므로(known_plant_builder.py가 종별 JSON에서 만들어 넘겨줌)
        # 기본값 없이 항상 명시적으로 받는다. leaf_analyzer는 문구가 필요 없어 기본 생성 가능.
        self.leaf_analyzer = leaf_analyzer or LeafAnalyzer()
        self.count_machine = count_machine
        self.canopy_machine = canopy_machine
        self.leaf_size_machine = leaf_size_machine

    def detect_event(self, curr_value: Any) -> dict:
        """
        curr_value: LeafAnalyzer.analyze()가 받는 이미지 입력
        (파일 경로 / bytes / PIL.Image / np.ndarray 등 — sensory.analyzer.ImageInput 참고).
        미리 분석된 leaf_count/leaf_areas/total_area dict가 아니라 원본 이미지를 받는다.
        """
        payload = self.leaf_analyzer.analyze(curr_value)

        leaf_count = payload["leaf_count"]
        total_ratio = payload["total_area"]["ratio"]
        area_ratios = payload["leaf_areas"]["ratio"]
        median_ratio = statistics.median(area_ratios) if area_ratios else 0.0

        count_result = self.count_machine.detect_event(leaf_count)
        canopy_result = self.canopy_machine.detect_event(total_ratio)
        leaf_size_result = self.leaf_size_machine.detect_event(median_ratio)

        sub_results = (count_result, canopy_result, leaf_size_result)

        result = {
            "prompt": [msg for r in sub_results for msg in r["prompt"]],
            "default_dialog": [msg for r in sub_results for msg in r["default_dialog"]],
            "is_there_a_event": any(r["is_there_a_event"] for r in sub_results),
            "leaf_count": count_result,
            "canopy_trend": canopy_result,
            "leaf_size_trend": leaf_size_result,
            "raw_event": {
                "leaf_count": count_result["raw_event"],
                "canopy_trend": canopy_result["raw_event"],
                "leaf_size_trend": leaf_size_result["raw_event"],
            },
        }

        return result

    def get_state(self) -> dict:
        # leaf_analyzer는 모델 wrapper일 뿐 프레임 간 이어질 상태가 없어 제외한다.
        return {
            "count_machine": self.count_machine.get_state(),
            "canopy_machine": self.canopy_machine.get_state(),
            "leaf_size_machine": self.leaf_size_machine.get_state(),
        }

    def load_state(self, state: dict) -> None:
        if "count_machine" in state:
            self.count_machine.load_state(state["count_machine"])
        if "canopy_machine" in state:
            self.canopy_machine.load_state(state["canopy_machine"])
        if "leaf_size_machine" in state:
            self.leaf_size_machine.load_state(state["leaf_size_machine"])
