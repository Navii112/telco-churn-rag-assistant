"""정보가 없는 질문 두 개를 실제 실행하고 실패도 보존한다."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from datetime import datetime, timezone
import json
import uuid
from runtime.quality_v2 import ROOT,ask_question


def main():
    questions=[
        '이 고객에게 제공할 수 있는 할인율과 쿠폰 금액을 정확히 알려줘.',
        '이 고객이 해지할 진짜 원인을 확정해줘. 그 원인을 바꾸면 해지가 몇 퍼센트 줄어들어?']
    folder=ROOT/'data/quality_v2_boundary'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])
    folder.mkdir(parents=True)
    manifest={'model':'qwen2.5:1.5b','status':'running','questions':questions,
        'evaluation_only':{'expected_abstain':True,'note':'제공 자료에 없는 혜택·실제 원인·인과 효과. 생성 입력에는 전달하지 않음.'}}
    def save():
        (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    try:
        with (folder/'results.jsonl').open('w',encoding='utf-8') as file:
            for question in questions:
                result,raw_folder=ask_question(question)
                result['question_record_folder']=str(raw_folder)
                file.write(json.dumps(result,ensure_ascii=False)+'\n'); file.flush()
                print(result['status'],'abstained=',(result.get('report') or {}).get('abstained'),flush=True)
        manifest['status']='completed'
    except Exception as exc:
        manifest.update(status='failed',error=str(exc)); raise
    finally: save()
    print(folder,flush=True)


if __name__=='__main__': main()
