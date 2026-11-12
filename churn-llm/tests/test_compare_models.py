import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import copy
import unittest
from runtime.llm_reports import generate_report, CONDITIONS, GenerationConfig
from tests.test_llm_reports import fake_response
from experiments.compare_models import verify_controls, model_summary


class CompareModelsTests(unittest.TestCase):
    def setUp(self):
        context = {'question': '질문', 'customer_hash': 'customer', 'corpus_hash': 'corpus',
                   'retrieval_method': 'tfidf', 'guides': [], 'customer': [
                       {'id': 'customer:profile','title': '입력','text': '관측된 고객 입력','kind': 'customer','source': 'test'}]}
        self.left = [generate_report(context,c,fake_response,GenerationConfig(model='qwen2.5:1.5b')) for c in CONDITIONS]
        self.right = [generate_report(context,c,fake_response,GenerationConfig(model='qwen2.5:3b')) for c in CONDITIONS]
        for records in (self.left,self.right):
            for r in records:
                r.update(case_id='test',repeat=0)

    def test_only_model_change_accepted(self):
        self.assertTrue(verify_controls(self.left,self.right))

    def test_mismatched_prompt_or_missing_condition_rejected(self):
        changed = copy.deepcopy(self.right)
        changed[0]['messages'][0]['content'] += 'new'
        with self.assertRaises(ValueError):
            verify_controls(self.left,changed)
        with self.assertRaises(ValueError):
            verify_controls(self.left,self.right[:-1])

    def test_failures_stay_in_model_summary(self):
        records = copy.deepcopy(self.right)
        records[0]['automatic']['structure_valid'] = False
        records[0]['automatic']['all_citations_valid'] = False
        records[0]['automatic']['citation_match_rate'] = None
        summary = model_summary(records)
        self.assertEqual(summary['attempts'],4)
        self.assertEqual(summary['structure_valid_rate'],0.75)
        self.assertEqual(summary['all_citations_valid_rate'],0.75)
        self.assertEqual(summary['citation_match_rate_all_attempts'],0.75)


if __name__ == '__main__':
    unittest.main()
