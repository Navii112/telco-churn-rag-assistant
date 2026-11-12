"""로컬 화면의 최신 생성 경로. 이전 실험은 별도로 보존한다."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
from runtime.llm_reports import CONDITIONS, GenerationConfig
from runtime.quality_v2 import generate_v2, assess, claims_of
from runtime.question_scope import validate_question_scope

ROOT=Path(__file__).resolve().parents[1]


def generate_local_report(context,condition='rag_improved',config=None,request=None,output_root=None):
    validate_question_scope(context['question'])
    if condition not in CONDITIONS: raise ValueError('지원하지 않는 조건입니다.')
    if not context['question'].strip(): raise ValueError('상담 요청을 입력하세요.')
    config=config or GenerationConfig()
    baseline={'trial_id':'local:0:'+condition,'case_id':'local','repeat':0,'condition':condition,
        'question':context['question'],'config':asdict(config),'customer_hash':context['customer_hash'],
        'corpus_hash':context['corpus_hash'],
        'references':context['customer']+(context['guides'] if CONDITIONS[condition]['rag'] else [])}
    result=generate_v2(baseline,**({'request':request} if request is not None else {}))
    result['evaluation']=assess(result['report'],result['references'],result['question'])
    folder=Path(output_root or ROOT/'data/local_reports')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    result['record_folder']=str(folder)
    (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def local_report_markdown(result):
    lines=['# LLM 상담 준비 초안','', '질문: '+result['question'],
        '모델: '+result['config']['model'],'상태: '+result['status'],
        '주의: 시연용 가상 지침 기반. 의미 정확성은 미평가입니다.','']
    for location,claim in claims_of(result.get('report')):
        lines.extend(['## '+location,claim['text']])
        link=next((x for x in result['evaluation']['links'] if x['location']==location),None)
        if link:
            for citation in link['citations']: lines.append('['+citation['id']+'] '+citation['text'])
            if link['missing_ids']: lines.append('연결 실패: '+', '.join(link['missing_ids']))
    for check in result['evaluation']['content_checks']:
        if check['issues']: lines.append('검토 필요: '+check['text']+' — '+' / '.join(check['issues']))
    report=result.get('report')
    if isinstance(report,dict): lines.extend(['## 한계',*report.get('limitations',[])])
    return '\n'.join(lines)
