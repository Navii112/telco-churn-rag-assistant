import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import copy
import tempfile
import unittest
import numpy as np

from runtime.service import ChurnService
from runtime.guidance import load_guidelines, applicable, prepare_report, validate_report, report_markdown
from runtime.embedding_search import EmbeddingSearch, unit_vectors


class GuidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ChurnService()

    def test_all_sections_loaded_and_marked_fictional(self):
        docs = load_guidelines()
        self.assertEqual(len(docs), 10)
        self.assertIn('guide:payment', [d['id'] for d in docs])
        self.assertTrue(all(d['kind'] == 'fictional_demo_only' and d['notice'] for d in docs))

    def test_applicability_for_internet_and_tenure(self):
        docs = {d['id']: d for d in load_guidelines()}
        pred = {'features': {'InternetService': 'No', 'tenure': 40}}
        self.assertFalse(applicable(docs['guide:technical'], pred))
        self.assertFalse(applicable(docs['guide:onboarding'], pred))
        self.assertTrue(applicable(docs['guide:no-internet'], pred))
        pred['features']['tenure'] = 12
        self.assertTrue(applicable(docs['guide:onboarding'], pred))

    def test_report_source_binding_and_export(self):
        row = self.service.features[self.service.features['InternetService'] == 'No'].iloc[0].to_dict()
        row['tenure'] = 40
        prediction = self.service.predict(row)
        report = prepare_report(self.service, prediction, '인터넷이 없는 고객의 상담을 준비해줘', 'tfidf')
        self.assertTrue(report['questions'])
        self.assertEqual(report['generation_mode'], 'source_template')
        ids = {d['id'] for d in report['references']}
        self.assertNotIn('guide:technical', ids)
        self.assertNotIn('guide:onboarding', ids)
        self.assertIn('guide:benefits', ids)
        self.assertIn('시연용', report_markdown(report))
        broken = copy.deepcopy(report)
        broken['actions'][0]['quote'] = '할인 50% 제공'
        with self.assertRaises(ValueError):
            validate_report(broken)

    def test_report_requires_prediction_and_request(self):
        with self.assertRaises(ValueError):
            prepare_report(self.service, None, '질문', 'tfidf')
        prediction = self.service.predict(self.service.features.iloc[0].to_dict())
        with self.assertRaises(ValueError):
            prepare_report(self.service, prediction, '', 'tfidf')


class FakeEmbed:
    def __init__(self):
        self.calls = 0
        self.digest = 'v1'

    def __call__(self, route, payload=None):
        if route == '/api/tags':
            return {'models': [{'name': 'embeddinggemma:latest', 'digest': self.digest}]}
        self.calls += 1
        return {'embeddings': [[1, 0] if '요금' in text else [0, 1] for text in payload['input']]}


class EmbeddingTests(unittest.TestCase):
    def test_cache_model_digest_and_candidate_filter(self):
        docs = [{'id': 'bill', 'title': '요금', 'text': '요금 문의'},
                {'id': 'tech', 'title': '장애', 'text': '인터넷 문의'}]
        request = FakeEmbed()
        with tempfile.TemporaryDirectory() as folder:
            search = EmbeddingSearch(docs, folder, request)
            self.assertEqual(search.search('요금')[0]['id'], 'bill')
            self.assertEqual(search.search('요금', allowed_ids={'tech'})[0]['id'], 'tech')
            self.assertEqual(request.calls, 3)  # 색인 1회 + 질의 2회
            cached = EmbeddingSearch(docs, folder, request)
            cached.search('요금')
            self.assertEqual(request.calls, 4)  # 문서 재계산 없음
            request.digest = 'v2'
            cached.search('요금')
            self.assertEqual(request.calls, 6)  # 모델 변경으로 문서 재계산

    def test_invalid_vectors_and_missing_model(self):
        for vector in ([[0, 0]], [[np.nan, 1]], [1, 2]):
            with self.assertRaises(ValueError):
                unit_vectors(vector)
        with tempfile.TemporaryDirectory() as folder:
            search = EmbeddingSearch([{'id': 'a', 'title': 'a', 'text': 'a'}], folder,
                                     lambda *args: {'models': []})
            with self.assertRaises(RuntimeError):
                search.ensure_index()


if __name__ == '__main__':
    unittest.main()
