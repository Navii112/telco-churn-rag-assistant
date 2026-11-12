import unittest
from unittest.mock import patch

from runtime.question_scope import OutOfScopeQuestion, validate_question_scope
from runtime.local_reports import generate_local_report
from runtime.quality_v2 import ask_question


class QuestionScopeTests(unittest.TestCase):
    def test_unrelated_requests_are_rejected(self):
        for question in ['오늘 날씨 어때?', '내일 비 와?', '서울 기온 알려줘', '주가 알려줘']:
            with self.subTest(question=question), self.assertRaises(OutOfScopeQuestion):
                validate_question_scope(question)

    def test_consultation_and_weather_context_are_allowed(self):
        for question in ['요약해줘', '할인 가능한가요?', '비가 와서 인터넷이 끊겼는데 상담을 준비해줘',
                         '이 고객의 해지 원인이 날씨라고 단정할 수 있어?']:
            validate_question_scope(question)

    def test_local_report_does_not_generate_or_save_weather(self):
        with patch('runtime.local_reports.generate_v2') as generate:
            with self.assertRaises(OutOfScopeQuestion):
                generate_local_report({'question': '오늘 날씨 어때?'})
            generate.assert_not_called()

    def test_notebook_question_is_checked_before_service(self):
        with patch('runtime.service.ChurnService') as service:
            with self.assertRaises(OutOfScopeQuestion):
                ask_question('오늘 날씨 어때?')
            service.assert_not_called()
