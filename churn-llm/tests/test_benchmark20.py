import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from experiments.benchmark20 import score_reference,summarize,make_review,build_cases
from experiments.benchmark_review import manual_summary
from experiments.benchmark20_scoring import field_mentions


class BenchmarkTests(unittest.TestCase):
    def test_generated_cells_compile(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'tools/create_benchmark20_notebook.py').read_text(encoding='utf-8'))
        count=0
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='code':
                text=eval(compile(ast.Expression(node.args[0]),'<cell generator>','eval'),{'encoded':'test'})
                compile(text,'<notebook cell>','exec'); count+=1
        self.assertGreater(count,5)

    def test_tenure_is_not_contract_duration(self):
        self.assertEqual(field_mentions('이용 기간은 69개월입니다. 계약 유형은 2년 계약입니다.','tenure'),[69])
        self.assertEqual(field_mentions('이용 기간은 69개월(약 6년 3개월)입니다.','tenure'),[69,75])
        self.assertEqual(field_mentions('이용 기간은 5년 9개월입니다.','tenure'),[69])

    def test_approximate_and_formatted_money(self):
        self.assertEqual(field_mentions('월 요금은 약 64.15달러입니다.','MonthlyCharges'),[64.15])
        self.assertEqual(field_mentions('누적 요금: 8,468.2USD','TotalCharges'),[8468.2])
        self.assertEqual(field_mentions('월 요금은 64.15원입니다.','MonthlyCharges'),[])

    def case(self):
        return {'id':'fact_01','question':'월 요금','gold':{'required_fields':{'MonthlyCharges':100.15},'topics':[],
            'expected_abstain':False}}

    def result(self,text):
        c={'text':text,'evidence_ids':['E001']}
        return {'report':{'summary':c,'questions':[c],'actions':[c],'limitations':[],'abstained':False},
            'status':'needs_semantic_review','wall_seconds':1,'references':[{'id':'E001',
            'parent_id':'customer:profile','parent_text':'{"tenure":8,"MonthlyCharges":100.15}',
            'text':'월 요금: 100.15USD','field':'MonthlyCharges','source':'test'}]}

    def test_no_gold_score_from_citations(self):
        result=self.result('월 요금은 확인해야 합니다.')
        result['report']['summary']['quote']='월 요금: 100.15USD'
        score=score_reference(self.case(),result)
        self.assertEqual(score['matched_count'],0)
        self.assertEqual(score['missing_count'],1)

    def test_correct_wrong_and_missing(self):
        self.assertEqual(score_reference(self.case(),self.result('월 요금은 100.15USD입니다.'))['matched_count'],1)
        self.assertEqual(score_reference(self.case(),self.result('월 요금은 908.55USD입니다.'))['mismatch_count'],1)
        self.assertEqual(score_reference(self.case(),self.result('알 수 없습니다.'))['missing_count'],1)

    def test_failure_in_denominator(self):
        case=self.case(); result=self.result('월 요금은 100.15USD입니다.')
        first={'trial_id':'qwen2.5:1.5b:fact_01:original','model':'qwen2.5:1.5b','version':'original',
            'case':case,'result':result,'metrics':score_reference(case,result)}
        result2={**result,'report':None,'status':'execution_error'}
        second={**first,'trial_id':'qwen2.5:1.5b:fact_02:original','result':result2,'metrics':score_reference(case,result2)}
        summary=summarize([first,second])[0]
        self.assertEqual(summary['attempts'],2)
        self.assertEqual(summary['required_fact_expression_rate'],0.5)

    def test_reviews_unrated_and_preserved(self):
        case=self.case(); result=self.result('월 요금은 100.15USD입니다.')
        row={'trial_id':'qwen2.5:1.5b:fact_01:original','case':case,'result':result}
        with TemporaryDirectory() as temporary:
            path=Path(temporary); make_review(path,[row])
            before=(path/'human_review.csv').read_bytes()
            make_review(path,[row])
            self.assertEqual(before,(path/'human_review.csv').read_bytes())
            summary=manual_summary(path)[0]
            self.assertEqual(summary['model'],'qwen2.5:1.5b')
            self.assertEqual(summary['reviewed'],0)
            self.assertIsNone(summary['factual_accuracy_0_2'])


if __name__=='__main__': unittest.main()
