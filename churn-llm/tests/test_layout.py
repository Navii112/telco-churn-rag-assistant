import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import os
import unittest
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_original_experiment_files_unchanged(self):
        manifest=json.loads((ROOT/'archive/layout_migration.json').read_text(encoding='utf-8'))
        for relative,expected in manifest['data_sha256'].items():
            # 원문 생성 결과만 불변 검사. 사람 평가 CSV는 이후 실제 채점을 허용한다.
            if Path(relative).name not in ('results.jsonl','pairs.jsonl','triples.jsonl'): continue
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest(),expected)

    def test_runtime_paths(self):
        from runtime.service import ROOT as service_root,ML
        from runtime.guidance import GUIDES
        from runtime.customer_fields import LABELS
        self.assertEqual(service_root,ROOT)
        self.assertEqual(ML,ROOT.parent/'ML')
        self.assertTrue((GUIDES/'manifest.json').is_file())
        self.assertEqual(LABELS['tenure'][1],'개월')

    def test_previous_notebook_setup_from_archive_directory(self):
        previous=Path.cwd()
        try:
            os.chdir(ROOT/'notebooks/archive')
            for name in ('churn_llm_analysis.ipynb','churn_llm_quality_improved.ipynb','churn_llm_multi_evidence.ipynb','churn_llm_analysis_v1.ipynb'):
                with self.subTest(notebook=name):
                    notebook=json.loads(Path(name).read_text(encoding='utf-8'))
                    cell=next(c for c in notebook['cells'] if c['cell_type']=='code')
                    source=''.join(cell['source']) if isinstance(cell['source'],list) else cell['source']
                    namespace={}
                    with capture_output(): exec(compile(source,name,'exec'),namespace)
                    self.assertEqual(namespace['PROJECT'],ROOT)
        finally:
            os.chdir(previous)


if __name__=='__main__': unittest.main()
