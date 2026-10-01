from enum import Enum


class SupportedAchivements(str, Enum):
    """
    ml part(이 라이브러리, 즉 센서/vision 기반 Persona)가 스스로 판단할 수 있는
    achievement만 담는다. "Supported"라는 이름이 붙은 이유가 여기 있다 — 앱 전체
    achievement 기획 중 이 라이브러리가 담당하는 부분집합만 여기 있고, 나머지(앱 레벨
    UI 상호작용 등 센서로 알 수 없는 것)는 앱 쪽에서 별도로 관리한다.

    이 파일은 "무엇을 달성이라고 부르는가"(이름/라벨)만 정의한다. 실제로 언제
    달성됐다고 판단하는지(감지 로직)는 planta_gochi.persona.achievement의
    AchievementTracker가 갖고 있다 — 다른 enum들이 문구 없이 상태/이벤트 이름만
    담당하고 실제 문구는 종별 JSON에 있는 것과 같은 분리다.

    "위기 대응 전문가"(상태이상 알림 후 30분 이내 정상 복구)는 여기 없다 — 다른
    항목들과 달리 "이번 틱에 이벤트가 한 번 감지되면 끝"이 아니라 경과 시간을 따로
    추적해야 해서 성격이 다르다. 지금 단계에서는 보류하기로 사용자와 확인했다
    (2026-09-29 대화) — 나중에 시간 추적 설계가 정해지면 별도로 추가한다.
    """

    STATUS_RECOVERY_TRIO = "살려야한다"
    FUNGAL_CURE = "트러플 바이옴? 아직 하드모드도 아니야!"
    BACTERIAL_CURE = "난 나보다 약한 세균의 명령따위 듣지 않는다."
    GROWTH_STAGE_SKIP = "아무에게도 말하지 마라!"
    GROWTH_STAGE_FIVE = "이제 가망이 없어"
    SUDDEN_ENVIRONMENT_SHIFT = "식물이 죽는다고!!"
    HIGH_HUMIDITY = "Too much water"
    NIGHT_OWL_FARMER = "올빼미 농부"
    TOUCH_GRASS = "Touch grass"
