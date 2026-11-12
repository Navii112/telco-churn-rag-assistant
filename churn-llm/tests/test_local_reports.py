import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from runtime.local_reports import generate_local_report,local_report_markdown
from tests.test_ui import fake_request


class LocalReportTests(unittest.TestCase):
    def context(self):
        return {'question':'계약 확인을 준비해줘','customer_hash':'test','corpus_hash':'test',
            'customer':[{'id':'customer:profile','text':'{"tenure":8,"OnlineSecurity":"No"}',
                'source':'test','title':'고객','kind':'customer'}],
            'guides':[{'id':'guide:test','text':'실제 계약 확인 필요.','source':'test','title':'지침','kind':'fictional_demo_only'}]}

    def test_no_rag_isolation_and_unique_record(self):
        with TemporaryDirectory() as temporary:
            first=generate_local_report(self.context(),'no_rag_basic',request=fake_request,output_root=temporary)
            second=generate_local_report(self.context(),'rag_improved',request=fake_request,output_root=temporary)
            self.assertNotEqual(first['record_folder'],second['record_folder'])
            self.assertFalse(any(d['kind']=='fictional_demo_only' for d in first['references']))
            self.assertTrue(any(d['kind']=='fictional_demo_only' for d in second['references']))
            self.assertTrue(Path(first['record_folder'],'result.json').exists())
            self.assertIn('미평가',local_report_markdown(first))

    def test_error_raw_preserved(self):
        def broken(route,payload=None,**kwargs):
            if route=='/api/chat': return {'done':True,'message':{'content':'not json'}}
            return fake_request(route,payload,**kwargs)
        with TemporaryDirectory() as temporary:
            result=generate_local_report(self.context(),request=broken,output_root=temporary)
            saved=json.loads(Path(result['record_folder'],'result.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['raw_text'],'not json')
            self.assertFalse(saved['structure_valid'])
            self.assertEqual(saved['evaluation']['link_rate'],0)


if __name__=='__main__': unittest.main()
