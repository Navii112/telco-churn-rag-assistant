import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import unittest
from runtime.quality_v2 import build_sources,normalize_ids,assess,validate_content,valid_structure


class MultiEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.sources=build_sources([{'id':'customer:profile','text':'{"tenure":8,"MonthlyCharges":100.15,"TotalCharges":908.55}',
            'source':'test','kind':'customer','title':'profile'}])

    def report(self,text='이용 기간은 8개월입니다.',ids=None):
        claim={'text':text,'evidence_ids':ids or ['E001']}
        return {'summary':claim,'questions':[{'text':'확인할 내용이 있나요?','evidence_ids':['E001']}],
            'actions':[{'text':'실제 계약을 확인하세요.','evidence_ids':['E001']}],
            'limitations':[],'abstained':False}

    def test_legacy_comma_and_duplicates(self):
        self.assertEqual(normalize_ids({'evidence_id':'E001, E002 E001;E999'}),['E001','E002','E999'])
        self.assertEqual(normalize_ids({'evidence_ids':['E001','E002','E001']}),['E001','E002'])

    def test_invalid_and_empty_ids_fail(self):
        r=self.report(ids=['E001','E999'])
        e=assess(r,self.sources)
        self.assertFalse(e['links'][0]['linked'])
        self.assertEqual(e['links'][0]['missing_ids'],['E999'])
        r['summary']['evidence_ids']=[]
        self.assertFalse(assess(r,self.sources)['links'][0]['linked'])
        self.assertFalse(valid_structure(r))

    def test_multiple_valid_references_link(self):
        self.assertTrue(assess(self.report(ids=['E001','E002']),self.sources)['all_links_valid'])

    def test_valid_but_irrelevant_numeric_source_flagged(self):
        rows=validate_content(self.report('월 요금은 100.15USD입니다.',['E001']),self.sources)
        self.assertEqual(rows[0]['support_status'],'필요 고객 근거 누락')
        self.assertTrue(rows[0]['issues'])

    def test_legacy_atom_does_not_cover_entire_parent(self):
        sources=[dict(d,field=None) for d in self.sources]
        checks=validate_content(self.report('월 요금은 100.15USD입니다.',['E001']),sources)
        self.assertTrue(checks[0]['issues'])

    def test_cutoff_and_uncited_fields(self):
        checks=validate_content(self.report('월 단위 계약으로 광인터넷을 사용합니다.',['E001']),self.sources)
        self.assertTrue(checks[0]['issues'])

    def test_wrong_security_subscription(self):
        sources=build_sources([{'id':'customer:profile','text':'{"OnlineSecurity":"No"}',
            'source':'test','kind':'customer','title':'profile'}])
        checks=validate_content(self.report('온라인 보안 서비스는 가입 상태입니다.'),sources)
        self.assertTrue(any('가입 여부' in x for x in checks[0]['issues']))

    def test_abstention_inconsistency(self):
        report=self.report(); report['limitations']=['혜택 제공 여부는 답변을 보류하고 정책 확인이 필요합니다.']
        result=assess(report,self.sources,'할인율을 알려줘')
        self.assertTrue(any(x['location']=='abstained' for x in result['content_checks']))
        checks=validate_content(self.report('가'*160),self.sources)
        self.assertTrue(checks[0]['issues'])

    def test_wrong_duration_and_contract_confusion(self):
        for text in ('이용 기간은 8년입니다.','계약 기간은 8개월입니다.'):
            self.assertTrue(validate_content(self.report(text),self.sources)[0]['issues'])

    def test_swapped_monthly_total(self):
        self.assertTrue(validate_content(self.report('월 요금은 908.55USD입니다.'),self.sources)[0]['issues'])
        self.assertFalse(validate_content(self.report('월 요금은 100.15USD입니다.', ['E002']),self.sources)[0]['issues'])

    def test_no_rules_is_not_accuracy_pass(self):
        e=assess(self.report('계약을 확인하세요.'),self.sources)
        self.assertIsNone(e['semantic_accuracy'])
        self.assertEqual(e['content_checks'][0]['support_status'],'사람 검토 필요')

    def test_limitation_guess_flagged(self):
        r=self.report(); r['limitations']=['계약 만료일은 자료에 없으므로 추측이 필요합니다.']
        self.assertTrue(any(x['issues'] for x in validate_content(r,self.sources) if x['location'].startswith('limitations')))

    def test_complete_sentences_not_cut(self):
        text='긴 문장이지만 중간에서 잘라서 제공하지 않고 전체 문장의 의미가 유지되어야 합니다.'*2
        docs=build_sources([{'id':'guide:test','text':text,'source':'test','title':'test','kind':'fictional_demo_only'}])
        self.assertEqual(docs[0]['text'],text)


if __name__=='__main__': unittest.main()
