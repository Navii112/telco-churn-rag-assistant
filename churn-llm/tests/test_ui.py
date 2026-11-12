import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import unittest
from pathlib import Path
from unittest.mock import patch
from tempfile import TemporaryDirectory
import json
from streamlit.testing.v1 import AppTest
from runtime.quality_v2 import generate_v2


def fake_request(route,payload=None,**kwargs):
    if route=='/api/tags': return {'models':[{'name':'qwen2.5:1.5b','digest':'test-only'}]}
    if route=='/api/show': return {}
    evidence=json.loads(payload['messages'][1]['content'])['evidence']
    ids=[d['id'] for d in evidence[:2]]
    claim={'text':'계약 조건을 확인하세요.','evidence_ids':ids}
    return {'done':True,'done_reason':'stop','message':{'content':json.dumps({
        'summary':claim,'questions':[claim],'actions':[claim],
        'limitations':['실제 계약 확인 필요'],'abstained':False},ensure_ascii=False)}}


class ScreenTests(unittest.TestCase):
    def test_prediction_and_llm_direct_report(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'), default_timeout=40).run()
        next(x for x in app.selectbox if x.label == '검색 방식').select('TF-IDF (비교 기준)').run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == 'ML 예측 및 설명 계산').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state['prediction']['contributions']),19)
        with TemporaryDirectory() as temporary, patch('runtime.local_reports.ROOT',Path(temporary)), patch(
            'runtime.local_reports.generate_v2',side_effect=lambda baseline:generate_v2(baseline,request=fake_request)):
            next(b for b in app.button if b.label == 'LLM 상담 리포트 생성').click().run()
            self.assertTrue(list(Path(temporary).glob('data/local_reports/*/result.json')))
        self.assertFalse(app.exception)
        result = app.session_state['llm_result']
        self.assertEqual(result['version'],'quality-v2-multi-evidence')
        self.assertEqual(result['status'],'needs_semantic_review')
        self.assertTrue(result['evaluation']['all_links_valid'])
        self.assertIsNone(result['evaluation']['semantic_accuracy'])
        self.assertTrue(result['raw_text'])
        self.assertIn('4조건 비교·평가',[tab.label for tab in app.tabs])
        # 정상 결과가 있던 화면에서도 범위 밖 질문은 기존 리포트를 지운다.
        next(x for x in app.text_area if x.label == '상담 요청').input('오늘 날씨 어때?')
        with patch('runtime.local_reports.generate_v2') as generate, patch('runtime.llm_reports.prepare_context') as retrieve:
            next(b for b in app.button if b.label == 'LLM 상담 리포트 생성').click().run()
            generate.assert_not_called()
            retrieve.assert_not_called()
        self.assertFalse(app.exception)
        self.assertNotIn('llm_result', app.session_state)
        self.assertTrue(any('실시간 날씨' in x.value for x in app.info))
        self.assertFalse(any(x.label == '근거 ID 연결률' for x in app.metric))


if __name__ == '__main__':
    unittest.main()
