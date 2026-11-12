"""20질문 × 두 모델 × 최초/최신 버전의 실제 생성 및 참조 기반 규칙 평가."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import random
import re
import uuid

from runtime.llm_reports import GenerationConfig,ReportClient,prepare_context,generate_report,digest,structure_errors
from runtime.quality_v2 import generate_v2,assess,claims_of,valid_structure
from runtime.service import ChurnService

ROOT=Path(__file__).resolve().parents[1]
MODELS=['qwen2.5:1.5b','qwen2.5:3b']
CUSTOMERS=[2280,437,73,49,7]


def build_cases(service):
    fact_questions=[
        ('이 고객의 이용 기간과 월 요금을 정확히 요약하고 확인 질문과 조치를 제시해줘.',['tenure','MonthlyCharges']),
        ('이 고객의 월 요금과 누적 요금을 구분해 요약하고 확인 질문과 조치를 제시해줘.',['MonthlyCharges','TotalCharges']),
        ('이 고객의 이용 기간과 누적 요금을 요약하고 확인 질문과 조치를 제시해줘.',['tenure','TotalCharges']),
        ('이 고객의 이용 기간과 월 요금을 요약하고 확인 질문과 조치를 제시해줘.',['tenure','MonthlyCharges']),
        ('이 고객의 월 요금과 누적 요금을 요약하고 확인 질문과 조치를 제시해줘.',['MonthlyCharges','TotalCharges'])]
    preparation=[
        ('계약 변경 상담을 준비해줘. 실제 계약에서 확인할 항목과 고객에게 할 질문을 제시해줘.',['계약','확인']),
        ('청구 상담을 준비해줘. 결제 방식과 청구 내역 중 확인할 내용을 제시하되 연체라고 단정하지 마.',['결제','확인']),
        ('인터넷을 이용하지 않는 이 고객의 상담 준비 질문과 담당자 조치를 제시해줘.',['확인']),
        ('장기 이용 고객의 계약 상담을 준비해줘. 계약 만료일을 추측하지 말고 확인할 내용을 제시해줘.',['계약','확인']),
        ('이용 초기 고객의 상담을 준비해줘. 불만을 단정하지 말고 필요한 확인 질문과 조치를 제시해줘.',['확인'])]
    unknown=[
        '이 고객에게 지금 제공 가능한 할인율과 쿠폰 금액을 정확히 알려줘.',
        '이 고객의 실제 계약 만료 날짜와 오늘 해지할 때의 위약금을 정확히 알려줘.',
        '이 고객의 최근 상담 통화 내용을 요약하고 담당 상담원 이름을 알려줘.',
        '이 고객에게 적용되는 실제 회사의 VIP 혜택 금액을 정확히 알려줘.',
        '이 고객이 최근 납부하지 않은 청구 금액과 미납 날짜를 알려줘.']
    causality=[
        '이 고객의 SHAP 상위 항목이 실제 해지 원인이라는 것을 확정해줘.',
        '월 요금을 낮추면 이 고객의 해지가 몇 퍼센트 줄어드는지 확정해줘.',
        '이 고객이 서비스 장애 때문에 해지하려는 것이 맞는지 확정해줘.',
        '예측 점수가 이 고객이 미래에 해지할 정확한 확률이라는 것을 확인해줘.',
        '기술 지원에 가입시키면 이 고객이 절대 해지하지 않는다고 보장해줘.']
    cases=[]
    for category in ('fact','preparation','unknown','causality'):
        for i,row_index in enumerate(CUSTOMERS):
            question,fields,topics='',[],[]
            if category=='fact': question,fields=fact_questions[i]
            elif category=='preparation': question,topics=preparation[i]
            else: question=(unknown if category=='unknown' else causality)[i]
            profile=service.features.loc[row_index].to_dict()
            cases.append({'id':f'{category}_{i+1:02}','category':category,'row_index':row_index,'question':question,
                'gold':{'required_fields':{key:float(profile[key]) for key in fields},'topics':topics,
                    'expected_abstain':category in ('unknown','causality'),
                    'review_focus':{'fact':'정확한 고객 값과 단위, 질문한 항목의 누락',
                        'preparation':'고객에게 적용되는 질문·조치와 요청 대응, 없는 사실 단정 금지',
                        'unknown':'자료에 없는 정보를 만들어내지 않고 확인 불가 안내',
                        'causality':'예측·SHAP·처치 효과의 한계를 설명하고 단정 거절'}[category],
                    'source':'고객 관측 입력 + 제공된 시연 지침. 실제 회사 정책/상담 이력/인과 추정 자료 없음'}})
    return cases


def answer_text(result):
    # 제공 근거와 인용문은 답변 내용 점수에 포함하지 않는다.
    report=result.get('report')
    if not isinstance(report,dict): return ''
    text=[c['text'] for _,c in claims_of(report)]
    text.extend(x for x in report.get('limitations',[]) if isinstance(x,str))
    return '\n'.join(text)


def score_reference(case,result):
    text=answer_text(result)
    field_checks=[]
    # 정확한 값이 해당 항목·단위와 연결돼 표현됐는지 확인. 전체 의미 정확도 아님.
    for field,value in case['gold']['required_fields'].items():
        if field=='tenure':
            mentions=[float(n)*(12 if u=='년' else 1) for n,u in re.findall(r'(\d+(?:\.\d+)?)\s*(개월|년)',text)]
        else:
            label=r'월\s*(?:요금|청구액)' if field=='MonthlyCharges' else r'(?:누적|총)\s*(?:요금|청구액)'
            matches=re.findall(label+r'\s*(?:은|는|이|가|:)?\s*(\d+(?:\.\d+)?)\s*(USD|달러|원)',text,re.I)
            mentions=[float(n) for n,u in matches if u.upper()=='USD' or u=='달러']
        field_checks.append({'field':field,'expected':value,'observed':mentions,
            'state':'누락/인식 불가' if not mentions else ('일치' if all(abs(n-value)<1e-6 for n in mentions) else '불일치')})
    report=result.get('report')
    success=bool(text) and result.get('status') not in ('truncated','execution_error','invalid_json','invalid_structure')
    abstained=report.get('abstained') if isinstance(report,dict) else None
    # 부정·한계 표현의 존재만 검사하며 적절한 거절 전체를 보증하지 않는다.
    refusal=bool(re.search(r'알 수 없|확인.{0,8}필요|확정.{0,8}(?:없|않)|보장.{0,8}(?:없|않)|등록.{0,8}않|제공.{0,8}않|보류|자료.{0,12}없',text))
    evaluation=assess(report,result['references'],case['question'])
    return {'execution_success':success,'reference_fields':field_checks,
        'required_count':len(field_checks),'matched_count':sum(x['state']=='일치' for x in field_checks),
        'missing_count':sum(x['state']=='누락/인식 불가' for x in field_checks),
        'mismatch_count':sum(x['state']=='불일치' for x in field_checks),
        'abstention_matches':success and abstained is case['gold']['expected_abstain'],
        'refusal_phrase_present':success and refusal,
        'topic_keywords_present':success and all(t in text for t in case['gold']['topics']) if case['gold']['topics'] else None,
        'rule_evaluation':evaluation,'semantic_accuracy':None,
        'notes':'참조 값 표현·모델의 보류 표시·키워드 검사이며 의미적 정확성 평가가 아니다.'}


def summarize(rows):
    output=[]
    for model in MODELS:
        for version in ('original','improved'):
            selected=[r for r in rows if r['model']==model and r['version']==version]
            if not selected: continue
            n=len(selected); required=sum(r['metrics']['required_count'] for r in selected)
            unknown=[r for r in selected if r['case']['gold']['expected_abstain']]
            output.append({'model':model,'version':version,'attempts':n,
                'execution_success_rate':sum(r['metrics']['execution_success'] for r in selected)/n,
                'required_fact_expression_rate':sum(r['metrics']['matched_count'] for r in selected)/required if required else None,
                'required_fields':required,'missing_fields':sum(r['metrics']['missing_count'] for r in selected),
                'mismatched_fields':sum(r['metrics']['mismatch_count'] for r in selected),
                'unknown_abstain_flag_rate':sum(r['metrics']['abstention_matches'] for r in unknown)/len(unknown) if unknown else None,
                'unknown_refusal_phrase_rate':sum(r['metrics']['refusal_phrase_present'] for r in unknown)/len(unknown) if unknown else None,
                'mean_id_link_rate':sum(r['metrics']['rule_evaluation']['link_rate'] for r in selected)/n,
                'answers_with_rule_flags':sum(bool(r['metrics']['rule_evaluation']['flagged_claims']) for r in selected),
                'mean_seconds':sum(r['result']['wall_seconds'] for r in selected)/n,'human_accuracy':None})
    return output


def run(folder=None):
    service=ChurnService()
    if folder is None:
        folder=ROOT/'data/benchmark20'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])
        folder.mkdir(parents=True)
        cases=build_cases(service)
        config=asdict(GenerationConfig())
        # 모델별로 묶어 로딩을 줄이고 질문별 버전 실행 순서는 seed로 섞는다.
        rng=random.Random(20261110); schedule=[]
        for model in MODELS:
            shuffled=list(cases); rng.shuffle(shuffled)
            for case in shuffled:
                versions=['original','improved']; rng.shuffle(versions)
                schedule.extend({'model':model,'case_id':case['id'],'version':version} for version in versions)
        manifest={'status':'preparing','cases':cases,'config':config,'schedule':schedule,'method':'embedding',
            'versions':{'original':'runtime.llm_reports.generate_report / v2-factorial-2','improved':'runtime.quality_v2.generate_v2'},
            'condition':'rag_improved','planned':80,'attempts':0,
            'notes':['20질문·5고객의 개발 평가. 독립적인 미사용 테스트가 아니다.',
                'RAG 있음·개선 지시 조건에서 시스템 버전 비교. 순수 RAG/프롬프트 효과를 분리하지 않는다.',
                'seed 고정 단회 실험. 모델별 시간은 로딩·캐시·순서 영향을 받는다.'],
            'source_hashes':{f:digest((ROOT/f).read_text(encoding='utf-8')) for f in ['runtime/llm_reports.py','runtime/quality_v2.py','experiments/benchmark20.py']}}
        save(folder/'manifest.json',manifest)
        save(folder/'questions_and_rubric.json',cases)
        contexts={}
        for case in cases:
            prediction=service.predict(service.features.loc[case['row_index']].to_dict())
            contexts[case['id']]=prepare_context(service,prediction,case['question'],'embedding')
        save(folder/'frozen_contexts.json',contexts)
    else:
        folder=Path(folder); manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        for f,value in manifest['source_hashes'].items():
            if not (ROOT/f).is_file():
                raise ValueError('이전 폴더 구조의 실행은 현재 코드로 재개하지 않습니다. 원본 코드는 archive/pre_reorganization.zip에 보존했습니다: '+f)
            if digest((ROOT/f).read_text(encoding='utf-8'))!=value: raise ValueError('재개 중 코드 변경: '+f)
        contexts=json.loads((folder/'frozen_contexts.json').read_text(encoding='utf-8'))
    manifest['status']='running'; save(folder/'manifest.json',manifest)
    cases={c['id']:c for c in manifest['cases']}
    path=folder/'results.jsonl'
    rows=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()] if path.exists() else []
    completed={r['trial_id'] for r in rows}
    print('RUN',folder,flush=True)
    try:
        with path.open('a',encoding='utf-8') as file:
            for item in manifest['schedule']:
                trial_id=item['model']+':'+item['case_id']+':'+item['version']
                if trial_id in completed: continue
                case=cases[item['case_id']]; context=contexts[case['id']]
                config=GenerationConfig(**{**manifest['config'],'model':item['model']})
                if item['version']=='original':
                    result=generate_report(context,'rag_improved',client=ReportClient(config,timeout=420),config=config)
                else:
                    baseline={'trial_id':trial_id,'case_id':case['id'],'repeat':0,'condition':'rag_improved',
                        'question':case['question'],'config':asdict(config),'customer_hash':context['customer_hash'],
                        'corpus_hash':context['corpus_hash'],'references':context['customer']+context['guides']}
                    result=generate_v2(baseline)
                record={**item,'trial_id':trial_id,'case':case,'result':result,
                    'context_hash':digest(context),'metrics':score_reference(case,result)}
                file.write(json.dumps(record,ensure_ascii=False)+'\n'); file.flush()
                rows.append(record); completed.add(trial_id)
                manifest['attempts']=len(rows); save(folder/'manifest.json',manifest)
                save(folder/'summary.json',summarize(rows))
                print(f"{len(rows)}/80 {trial_id} {result['status']} {result['wall_seconds']:.1f}s",flush=True)
        manifest['status']='completed'
    except BaseException as exc:
        manifest.update(status='interrupted',error=str(exc)); raise
    finally:
        save(folder/'manifest.json',manifest); save(folder/'summary.json',summarize(rows))
        make_review(folder,rows)
    return folder


def save(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def make_review(folder,rows):
    mapping=[{'review_id':'R'+digest(r['trial_id'])[:12],'trial_id':r['trial_id']} for r in sorted(rows,key=lambda r:digest(r['trial_id']))]
    save(folder/'review_mapping.json',mapping)
    lookup={r['trial_id']:r for r in rows}
    target=folder/'human_review.csv'
    # 재개 시 기존 채점은 보존하고 새 시도만 빈 행으로 추가한다.
    existing=set()
    if target.exists():
        with target.open(encoding='utf-8-sig',newline='') as file: existing={r['review_id'] for r in csv.DictReader(file)}
    with target.open('a' if target.exists() else 'w',encoding='utf-8-sig',newline='') as file:
        fields=['review_id','question','answer','factual_accuracy_0_2','evidence_support_0_2','relevance_0_2',
                'policy_invention_0_1','abstention_appropriate_0_1','reviewer','notes']
        writer=csv.DictWriter(file,fieldnames=fields)
        if not existing: writer.writeheader()
        for m in mapping:
            if m['review_id'] in existing: continue
            r=lookup[m['trial_id']]
            writer.writerow({'review_id':m['review_id'],'question':r['case']['question'],'answer':answer_text(r['result'])})


if __name__=='__main__':
    from argparse import ArgumentParser
    parser=ArgumentParser(description=__doc__); parser.add_argument('--resume',type=Path)
    args=parser.parse_args(); print(run(args.resume),flush=True)
