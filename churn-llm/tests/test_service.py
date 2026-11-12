import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import copy
import json
import math
import unittest
import pandas as pd

from runtime.service import ChurnService, validate_answer


class ChurnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ChurnService()

    def test_reproduces_saved_holdout_predictions(self):
        predictions = pd.read_csv(self.service.outputs / 'test_predictions.csv', index_col=0)
        for index in predictions.index[:3]:
            result = self.service.predict(self.service.features.loc[index].to_dict())
            self.assertAlmostEqual(result['score'], predictions.loc[index, 'Selected'], places=6)
            self.assertEqual(result['prediction'], int(result['score'] >= 0.5))

    def test_local_shap_reconstruction(self):
        result = self.service.predict(self.service.features.iloc[0].to_dict())
        margin = result['base_log_odds'] + sum(c['log_odds'] for c in result['contributions'])
        self.assertAlmostEqual(1/(1+math.exp(-margin)), result['score'], places=5)
        self.assertEqual(len(result['contributions']), 19)

    def test_no_label_or_identifier_in_llm_evidence(self):
        result = self.service.predict(self.service.features.iloc[0].to_dict())
        docs = self.service.retrieve('이 고객의 예측을 설명해줘', result)
        self.assertNotIn('customerID', result['features'])
        self.assertNotIn('Churn', result['features'])
        customer_docs = [d for d in docs if d['id'].startswith('customer:')]
        serialized = json.dumps(customer_docs)
        self.assertNotIn('Actual', serialized)
        self.assertNotIn('customerID', serialized)

    def test_metrics_retrieval_uses_selected_model(self):
        docs = self.service.retrieve('최종 모델의 테스트 성능은?')
        metric = next(d for d in docs if d['id'] == 'metrics:Selected')
        self.assertIn('ROC-AUC=0.847267', metric['text'])
        self.assertIn('scope', [d['id'] for d in docs])

    def test_inconsistent_services_and_target_column_rejected(self):
        row = self.service.features.iloc[0].to_dict()
        row['Churn'] = 1
        with self.assertRaises(ValueError):
            self.service.predict(row)
        row.pop('Churn')
        row['PhoneService'], row['MultipleLines'] = 'No', 'Yes'
        with self.assertRaises(ValueError):
            self.service.predict(row)

    def test_citations_checked(self):
        docs = self.service.retrieve('최종 모델 성능')
        good = {'statements': [{'text': '자료의 설명입니다.', 'evidence_id': docs[0]['id'],
                                'quote': docs[0]['text'][:15]}], 'limitation': ''}
        validate_answer(good, docs)
        for key, value in [('evidence_id', 'invented'), ('quote', '없는 인용문')]:
            bad = copy.deepcopy(good)
            bad['statements'][0][key] = value
            with self.assertRaises(ValueError):
                validate_answer(bad, docs)

    def test_missing_charge_imputed_and_unsupported_category_rejected(self):
        row = self.service.features.iloc[0].to_dict()
        row['TotalCharges'] = float('nan')
        result = self.service.predict(row)
        self.assertTrue(any('결측' in w for w in result['warnings']))
        row['Contract'] = 'unknown'
        with self.assertRaises(ValueError):
            self.service.predict(row)


if __name__ == '__main__':
    unittest.main()
