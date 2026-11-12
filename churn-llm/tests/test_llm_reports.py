import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runtime.llm_reports import (CONDITIONS, GenerationConfig, build_messages, prepare_context,
                        automatic_metrics, generate_report, customer_evidence, SCHEMA)
from experiments.run_experiments import run_comparison, summarize
from experiments.review_results import score_reviews
from runtime.service import ChurnService


def fake_response(messages):
    data = json.loads(messages[1]['content'])
    source = data['evidence'][0]
    claim = {'text': '고객 입력을 확인하는 시연 문장입니다.', 'evidence_id': source['id'], 'quote': source['text'][:20]}
    report = {'summary': claim, 'questions': [claim], 'actions': [claim], 'limitations': ['시연용'], 'abstained': False}
    return {'done': True, 'done_reason': 'stop', 'message': {'content': json.dumps(report, ensure_ascii=False)},
            'model': 'test', 'model_digest': 'test'}


class LLMReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ChurnService()
        cls.pred = cls.service.predict(cls.service.features.loc[2280].to_dict())
        cls.context = prepare_context(cls.service, cls.pred, '계약 조건을 확인해줘.', 'tfidf')

    def test_factorial_input_isolation(self):
        built = {c: build_messages(self.context, c) for c in CONDITIONS}
        for condition, (_, docs) in built.items():
            has_guides = any(d['id'].startswith('guide:') for d in docs)
            self.assertEqual(has_guides, CONDITIONS[condition]['rag'])
            self.assertEqual([d for d in docs if d['kind'] == 'customer'], self.context['customer'])
        self.assertEqual(built['rag_basic'][0][1], built['rag_improved'][0][1])
        self.assertEqual(built['no_rag_basic'][0][1], built['no_rag_improved'][0][1])
        self.assertEqual(built['rag_basic'][0][0], built['no_rag_basic'][0][0])

    def test_raw_preserved_before_citation_validation(self):
        raw = fake_response(build_messages(self.context, 'rag_improved')[0])
        report = json.loads(raw['message']['content'])
        report['actions'][0]['quote'] = '모든 고객에게 할인 제공'
        raw['message']['content'] = json.dumps(report)
        result = generate_report(self.context, client=lambda _: raw)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(result['raw_text'], raw['message']['content'])
        self.assertEqual(result['report']['actions'][0]['quote'], '모든 고객에게 할인 제공')
        self.assertFalse(result['automatic']['all_citations_valid'])

    def test_invalid_json_and_transport_error_are_counted(self):
        invalid = generate_report(self.context, client=lambda _: {'done': True, 'message': {'content': 'bad json'}})
        def fail(_):
            raise RuntimeError('transport failure')
        error = generate_report(self.context, client=fail)
        rows = summarize([invalid, error])
        self.assertEqual(rows[0]['attempts'], 2)
        self.assertEqual(rows[0]['json_valid_rate'], 0)
        self.assertEqual(invalid['raw_text'], 'bad json')
        self.assertEqual(error['status'], 'execution_error')

    def test_truncated_output_is_not_repaired(self):
        result = generate_report(self.context, client=lambda _: {'done': True, 'done_reason': 'length',
                                  'message': {'content': '{"summary":'}})
        self.assertEqual(result['status'], 'truncated')
        self.assertEqual(result['raw_text'], '{"summary":')

    def test_no_target_id_or_demographic_leak(self):
        text = json.dumps(customer_evidence(self.pred))
        for key in ('customerID', 'Churn', 'gender', 'SeniorCitizen', 'Dependents'):
            self.assertNotIn(key, text)

    def test_real_schema_validation_not_just_json(self):
        for report in ({}, [], {'summary': {}}):
            self.assertFalse(automatic_metrics(report, [])['structure_valid'])

    def test_experiment_four_attempts_same_customer_and_config(self):
        with tempfile.TemporaryDirectory() as temp:
            folder, rows = run_comparison(self.service, [{'id': 'test', 'row_index': 2280, 'question': '계약 확인'}],
                                          output_root=Path(temp), method='tfidf', client=fake_response)
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({r['customer_hash'] for r in rows}), 1)
            self.assertEqual(len({r['schema_hash'] for r in rows}), 1)
            self.assertEqual(len({json.dumps(r['config']) for r in rows}), 1)
            self.assertEqual(len((folder/'results.jsonl').read_text(encoding='utf-8').splitlines()), 4)
            self.assertTrue((folder/'manual_review.csv').exists())

    def test_pending_manual_reviews_are_not_zero_scores(self):
        row = {'trial_id': 'a', 'condition': 'rag_improved', 'reviewer': ''}
        summary = score_reviews([row])[0]
        self.assertEqual(summary['reviewed'], 0)
        self.assertIsNone(summary['factual_accuracy_0_2'])
        row.update(reviewer='tester', factual_accuracy_0_2='2',policy_invention_0_1='0',
                   causal_overclaim_0_1='0', request_relevance_0_2='2',abstention_appropriate_0_1='1')
        self.assertEqual(score_reviews([row])[0]['factual_accuracy_0_2'], 2)
        row['factual_accuracy_0_2'] = '5'
        with self.assertRaises(ValueError):
            score_reviews([row])


if __name__ == '__main__':
    unittest.main()
