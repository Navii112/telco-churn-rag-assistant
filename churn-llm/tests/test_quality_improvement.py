import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import unittest
from experiments.improve_quality import atomic_sources, link_report, fact_checks


class QualityTests(unittest.TestCase):
    def sources(self):
        return atomic_sources([{'id':'customer:profile','text':'{"tenure":8,"MonthlyCharges":100.15,"TotalCharges":908.55}',
            'source':'test','kind':'customer','title':'profile'}])

    def test_units_and_provenance(self):
        sources=self.sources()
        self.assertEqual(sources[0]['text'],'이용 기간: 8개월')
        self.assertIn('100.15USD',sources[1]['text'])
        self.assertEqual(sources[0]['parent_id'],'customer:profile')

    def test_invalid_id_not_repaired(self):
        original={'summary':{'text':'8개월','evidence_id':'MISSING'},'questions':[],'actions':[]}
        linked=link_report(original,self.sources())
        self.assertEqual(linked['summary']['quote'],'')
        self.assertNotIn('quote',original['summary'])

    def test_wrong_unit_flagged(self):
        self.assertTrue(fact_checks({'text':'8년 이용 고객'},self.sources()))
        self.assertFalse(fact_checks({'text':'8개월 이용 고객, 월 100.15USD'},self.sources()))


if __name__=='__main__': unittest.main()
