"""새 상담 요청의 명백한 범위 이탈을 생성 전에 확인한다.

규칙 기반 사전 검사이며 모든 질문의 의도를 판별하는 분류 모델은 아니다.
기존 실험의 생성 함수와 저장된 평가에는 적용하지 않는다.
"""
import re


class OutOfScopeQuestion(ValueError):
    """리포트 대신 서비스 범위를 안내해야 하는 요청."""


SCOPE_MESSAGE = (
    '이 도구는 고객 정보와 상담 지침을 바탕으로 상담 준비를 돕습니다. '
    '실시간 날씨·뉴스 등 상담과 무관한 정보는 제공하지 않습니다. '
    '예: “이 고객의 계약 조건을 요약해줘”, “요금 관련 확인 질문을 만들어줘”. '
    '범위 안내는 프로그램의 사전 검사이며 LLM이 생성한 상담 리포트가 아닙니다.'
)

# 통신 서비스 문제의 배경으로 날씨 등을 언급하는 질문은 허용한다.
_CONSULTATION = re.compile(
    r'고객|상담|계약|약정|해지|이탈|요금|청구|위약금|할인|혜택|가입|'
    r'인터넷|통신|통화|데이터\s*(?:요금|사용)|끊[기겼김]|장애|'
    r'SHAP|RAG|MonthlyCharges|OnlineSecurity|tenure', re.I)
_UNRELATED = re.compile(
    r'날씨|기온|일기예보|미세먼지|우산|비\s*(?:와|오|올)|'
    r'주식|주가|비트코인|뉴스|대통령|맛집|레시피|운세|로또|'
    r'weather|forecast|temperature', re.I)


def validate_question_scope(question):
    """명백한 비상담 주제만 거절하며 모호한 요청은 기존 경로로 전달한다."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('상담 요청을 입력하세요.')
    if _UNRELATED.search(question) and not _CONSULTATION.search(question):
        raise OutOfScopeQuestion(SCOPE_MESSAGE)
