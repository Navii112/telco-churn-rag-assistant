"""LLM 중심 실험 노트북 생성. 기존 v1은 archive에 최초 한 번 보존."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import uuid
ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):
    cells.append(dict(id=uuid.uuid4().hex[:8],cell_type='markdown',metadata={},source=s.strip()))
def code(s):
    cells.append(dict(id=uuid.uuid4().hex[:8],cell_type='code',metadata={},source=s.strip(),execution_count=None,outputs=[]))

md("""# 고객 상담 LLM: 직접 리포트 생성과 RAG·프롬프트 비교

**연구 질문:** 같은 고객·질문·모델에서 RAG와 프롬프트가 LLM 출력에 어떤 차이를 만드는가?
ML은 고객 참고 자료일 뿐이며 재학습·모델 선정을 하지 않습니다.
요약·확인 질문·담당자 조치·한계는 LLM이 직접 생성합니다. 템플릿으로 대체하지 않습니다.
상담 문서는 가상 시연용이며 실제 정책이 아닙니다.

커널은 `ML/.venv/Scripts/python.exe`를 선택하세요. 기본 실행은 저장된 **실제 실험 결과**를 분석합니다.
`RUN_GENERATION` 또는 `RUN_COMPARISON`을 True로 설정하면 새 로컬 모델 호출이 발생합니다.""")
md("""## 1. 실험 환경과 공통 생성 설정
모델·seed·temperature·문맥·출력 길이·JSON schema를 네 조건에서 동일하게 유지합니다.
작은 Qwen 모델을 기본으로 사용합니다. `ollama pull qwen2.5:1.5b`와 `ollama pull embeddinggemma`로 준비합니다.""")
code("""import os,sys,json,copy
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
PROJECT=next((p if (p/'runtime/llm_reports.py').exists() else p/'churn-llm'
              for p in [Path.cwd(),*Path.cwd().parents]
              if (p/'runtime/llm_reports.py').exists() or (p/'churn-llm/runtime/llm_reports.py').exists()),None)
if PROJECT is None: raise FileNotFoundError('churn-llm 또는 상위 폴더에서 실행하세요.')
sys.path.insert(0,str(PROJECT))
import pandas as pd
from IPython.display import display,Markdown,HTML
from html import escape
from runtime.service import ChurnService
from runtime.llm_reports import (CONDITIONS,PROMPTS,SCHEMA,GenerationConfig,prepare_context,
                        build_messages,generate_report,automatic_metrics,llm_report_markdown)
from experiments.run_experiments import run_comparison,summarize
from experiments.review_results import load_review_csv,score_reviews
from runtime.guidance import load_guidelines
def show_answer(record):
    text=llm_report_markdown(record)
    text=text.replace(json.dumps(record['automatic'],ensure_ascii=False),
                      '자동 검사 결과는 평가 표에서 확인하세요. 인용 검증 실패는 내용 검토가 필요하다는 뜻입니다.')
    display(Markdown(text))
    display(HTML('<details><summary>개발 확인용: 전체 JSON 원문 펼치기</summary><pre style="white-space:pre-wrap">'
                 +escape(record['raw_text'])+'</pre></details>'))
RUN_GENERATION=False
RUN_COMPARISON=False
SEARCH_METHOD='embedding'
CONFIG=GenerationConfig(model='qwen2.5:1.5b')
CASES=[json.loads(s) for s in (PROJECT/'data/report_cases.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
CASE=CASES[0]
print('Python:',sys.executable)
print('공통 설정:',CONFIG)
print('실험 사례:',CASE['id'],CASE['question'])""")
md("""## 2. ML 참고 입력 한 번만 계산
모든 조건에 같은 관측 입력·점수·상위 SHAP을 제공합니다. 고객ID·실제 정답·인구통계 특성은 LLM 입력에 넣지 않습니다.
이 셀은 고객 배경 준비이며 이번 프로젝트의 ML 성능 비교가 아닙니다.""")
code("""service=ChurnService()
prediction=service.predict(service.features.loc[CASE['row_index']].to_dict())
context=prepare_context(service,prediction,CASE['question'],SEARCH_METHOD)
print('고정 고객 스냅샷 hash:',context['customer_hash'])
display(pd.DataFrame([{'참고 점수':prediction['score'],'임계값':prediction['threshold'],'모델':prediction['model']}]))
print('생성 입력의 고객 근거:',[d['id'] for d in context['customer']])""")
md("""## 3. RAG 지식 문서와 청킹
시연 지침은 섹션 단위 청크 10개이며 출처·버전·적용 조건을 보존합니다.
RAG 조건에서만 지침을 넣고 no-RAG 조건에는 고객 참고 정보만 전달합니다.""")
code("""guides=load_guidelines()
display(pd.DataFrame([{'id':d['id'],'title':d['title'],'source':d['source'],'적용 조건':d['rule']} for d in guides]))
print('이번 질문의 고정 RAG 근거:',[d['id'] for d in context['guides']])
print(context['guides'][-1]['text'])""")
md("""## 4. 2×2 비교 설계와 누출 검사
- RAG 효과: 기본 프롬프트끼리, 개선 프롬프트끼리 비교합니다.
- 프롬프트 효과: RAG 없는 조건끼리, RAG 있는 조건끼리 비교합니다.
- A와 D만 비교하면 두 요소가 함께 바뀌므로 원인을 구분할 수 없습니다.
JSON 출력 형식 제약은 모든 조건에 공통입니다. no-RAG에 지침이나 템플릿 답안을 몰래 넣지 않습니다.""")
code("""display(pd.DataFrame([{'condition':c,**s} for c,s in CONDITIONS.items()]))
messages={c:build_messages(context,c) for c in CONDITIONS}
assert messages['rag_basic'][0][1]==messages['rag_improved'][0][1]
assert messages['no_rag_basic'][0][1]==messages['no_rag_improved'][0][1]
for c,(_,docs) in messages.items():
    assert any(d['id'].startswith('guide:') for d in docs)==CONDITIONS[c]['rag']
print('조건별 입력 격리 검사 통과')""")
md("""## 5. 기본·개선 프롬프트 확인
기본은 일반 상담 작성 지시, 개선은 사실/추정 구분·미등록 정책 생성 금지·정확한 인용·답변 보류 원칙을 추가합니다.
근거 ID를 schema enum으로 강제하지 않아 검증 실패도 관찰할 수 있습니다.""")
code("""for name,prompt in PROMPTS.items():
    print('---',name,'---')
    print(prompt)
print('공통 출력 필드:',list(SCHEMA['properties']))
print('RAG 개선 조건의 실제 입력:')
print(json.dumps(messages['rag_improved'][0],ensure_ascii=False,indent=2))""")
md("""## 6. 단일 리포트 직접 생성 또는 실제 저장 결과 열기
기본값은 과거 실제 실행을 읽습니다. 새 생성과 혼동하지 않도록 run_id·모델·상태를 표시합니다.
생성 실패/잘린 출력/잘못된 인용도 보존하고 템플릿으로 대체하지 않습니다.""")
code("""runs=sorted([p for p in (PROJECT/'data/experiments').glob('*') if (p/'manifest.json').exists() and (p/'results.jsonl').exists() and json.loads((p/'manifest.json').read_text(encoding='utf-8'))['status']=='completed' and json.loads((p/'manifest.json').read_text(encoding='utf-8'))['config']['model']==CONFIG.model],reverse=True)
if RUN_GENERATION:
    result=generate_report(context,'rag_improved',config=CONFIG)
    print('이번 셀에서 새로 생성했습니다.')
else:
    if not runs: raise FileNotFoundError('먼저 run_experiments.py --limit 1 을 실행하세요.')
    selected_run=runs[0]
    saved=[json.loads(s) for s in (selected_run/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    result=next((r for r in saved if r['condition']=='rag_improved'),saved[0])
    print('과거 실제 실행 결과:',selected_run.name)
print('모델:',result['config']['model'],'조건:',result['condition'],'상태:',result['status'])
show_answer(result)""")
md("""## 7. 네 조건 실제 실행 또는 결과 불러오기
`RUN_COMPARISON=True`는 1사례×4조건을 새로 실행합니다. 전체 사례는 CLI `--limit 4`로 실행합니다.
같은 고객과 문서를 한 번 준비한 후 네 조건에 재사용합니다. 원문은 시도별 JSONL에 즉시 기록됩니다.
실패도 분모에 포함하며 자동 재시도나 답변 교체를 하지 않습니다.""")
code("""if RUN_COMPARISON:
    selected_run,records=run_comparison(service,[CASE],method=SEARCH_METHOD,config=CONFIG,
        on_result=lambda r,n,total: print(n,'/',total,r['condition'],r['status'],flush=True))
    print('새 실험:',selected_run.name)
else:
    if not runs: raise FileNotFoundError('실제 비교 결과가 없습니다.')
    selected_run=runs[0]
    records=[json.loads(s) for s in (selected_run/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    print('불러온 실제 실험:',selected_run.name)
display(pd.DataFrame(summarize(records)).drop(columns=['status_counts'],errors='ignore'))
print('실제 호출 시도:',len(records))""")
md("""## 8. 같은 질문의 생성 답변 나란히 비교
실패한 답변도 생략하지 않습니다. 여기서는 질문 대응·정책 발명·원인 단정·답변 보류를 사람이 확인합니다.
형식/인용 검증은 답변 의미의 정확성을 대신하지 않습니다.""")
code("""comparison=[]
for record in records:
    report=record.get('report') or {}
    summary=report.get('summary',{}) if isinstance(report,dict) else {}
    comparison.append({'case':record['case_id'],'condition':record['condition'],'status':record['status'],
        'LLM 요약':summary.get('text','') if isinstance(summary,dict) else '',
        '원문':record['raw_text'],'인용 일치율':record['automatic']['citation_match_rate'],
        '입력 토큰':record['generation'].get('prompt_eval_count'),
        '출력 토큰':record['generation'].get('eval_count'),'시간(초)':record['wall_seconds']})
display(pd.DataFrame(comparison).drop(columns=['원문']))
# 긴 답변을 DataFrame에만 넣으면 잘리므로 전체 리포트를 별도로 출력합니다.
for record in records:
    display(Markdown('---'))
    print('사례:',record['case_id'],'조건:',record['condition'])
    show_answer(record)""")
md("""## 9. 자동 평가와 내용 수동 평가를 분리
자동: JSON/구조, 질문·조치 존재, 원문 인용 일치, 실패율, 토큰 수·시간.
수동: 사실 정확성·정책 발명·원인 단정·요청 대응·답변 보류 적절성.

`manual_review.csv`의 평가 점수와 reviewer를 입력하면 화면 또는 아래 집계 함수로 평가할 수 있습니다.
미입력은 0점이 아니라 미평가입니다. no-RAG는 지침을 받지 않으므로 지침 인용량만으로 종합 우열을 정하지 않습니다.""")
code("""reviews=load_review_csv(selected_run/'manual_review.csv')
display(Markdown('### 자동 검사: 프로그램이 확인한 결과'))
automatic_rows=[]
for r in records:
    a=r['automatic']
    automatic_rows.append({'조건':CONDITIONS[r['condition']]['label'],
        'JSON 형식':'통과' if a.get('json_valid') else '실패',
        '리포트 구성':'통과' if a.get('report_complete') else '미완성',
        '모든 인용 일치':'통과' if a.get('all_citations_valid') else '검토 필요',
        '인용 일치율':f"{a.get('citation_match_rate',0) or 0:.0%}"})
display(pd.DataFrame(automatic_rows))
display(Markdown('### 내용 평가: 사람이 답변과 근거를 읽고 채점할 항목'))
scored=score_reviews(reviews)
display(pd.DataFrame([{'조건':CONDITIONS[r['condition']]['label'],'전체 답변':r['attempts'],
    '평가 완료':r['reviewed'],'평가 상태':'미평가' if not r['reviewed'] else '평가 있음'} for r in scored]))
rubric=[('사실 정확성','고객 정보와 문서를 올바르게 설명했는가?','0~2점'),
        ('정책 발명','문서에 없는 할인·혜택·위약금을 만들었는가?','0=없음, 1=있음'),
        ('원인·효과 단정','예측이나 SHAP을 실제 원인·효과로 단정했는가?','0=없음, 1=있음'),
        ('요청 대응','사용자 질문에 필요한 답변을 했는가?','0~2점'),
        ('답변 보류','알 수 없는 내용을 확인 필요로 적절히 남겼는가?','0~1점')]
display(pd.DataFrame(rubric,columns=['평가 항목','확인할 내용','점수 기준']))
if any(r['reviewed'] for r in scored):
    labels={'condition':'조건','factual_accuracy_0_2':'사실 정확성',
        'policy_invention_0_1':'정책 발명','causal_overclaim_0_1':'원인·효과 단정',
        'request_relevance_0_2':'요청 대응','abstention_appropriate_0_1':'답변 보류 적절성'}
    display(pd.DataFrame(scored).rename(columns=labels).astype(object).fillna('미평가'))
else:
    display(Markdown('**아직 사람이 채점하지 않아 내용 점수는 없습니다. 기존 None은 오류나 0점이 아니라 미평가를 뜻합니다.**'))
print('평가 입력 파일:',selected_run/'manual_review.csv')
print('8번 절의 답변과 근거를 읽고 5개 점수와 reviewer를 입력한 뒤 이 셀을 재실행하세요.')
print('자동 검사는 내용의 정확성이나 환각률을 뜻하지 않습니다.')""")
md("""## 10. 검증 전 생성 원문 보존 확인
가짜 인용을 넣은 복사본으로 검증합니다. 원래 실험 결과는 변경하지 않습니다.
검증 결과로 답변을 숨겨도 실험에서 그 실패를 제거하지 않습니다.""")
code("""source=records[0]['references'][0]
bad_claim={'text':'모든 고객에게 할인 제공','evidence_id':source['id'],'quote':'존재하지 않는 할인 약속'}
bad_report={'summary':bad_claim,'questions':[bad_claim],'actions':[bad_claim],'limitations':[],'abstained':False}
check=automatic_metrics(bad_report,records[0]['references'])
assert not check['all_citations_valid']
print(check)
assert all('raw_text' in r for r in records)
print('실패·검증 전 원문 보존 확인')""")
md("""## 11. 임베딩 검색은 RAG의 하위 실험
이 프로젝트의 최종 평가 대상은 LLM 출력입니다. 검색 지표만으로 답변 개선을 주장하지 않습니다.
아래는 기존 소규모 검색 세트 결과로, 실제 생성 비교와 구분합니다.""")
code("""path=PROJECT/'data/search_evaluation.json'
if path.exists():
    search_evaluation=json.loads(path.read_text(encoding='utf-8'))
    display(pd.DataFrame([{'검색':k,'Hit@3':v['hit_at_3'],'MRR@3':v['mrr_at_3']} for k,v in search_evaluation['methods'].items()]))
    print(search_evaluation['note'])""")
md("""## 12. 추가 생성 모델 비교
같은 계열의 Qwen 1.5B와 3B를 비교합니다. 크기 외에 학습·양자화 차이도 있을 수 있으므로 파라미터 수의 순수 인과 효과로 해석하지 않습니다.
모델 이외 고객·문서·질문·프롬프트·schema·설정이 일치하는지 검사한 실제 실행 결과입니다.
한 사례의 네 조건 예비 비교이며 자동 지표는 의미적 정확도가 아닙니다.""")
code("""comparison_path=PROJECT/'data/model_comparison/comparison.json'
if comparison_path.exists():
    comparison=json.loads(comparison_path.read_text(encoding='utf-8'))
    assert comparison['controls_verified']
    display(Markdown('같은 질문·근거·프롬프트로 비교한 실제 저장 결과입니다. 1사례의 예비 실험이며 내용 평가는 아직 완료되지 않았습니다.'))
    summary_rows=[]
    for row in comparison['summary']:
        summary_rows.append({'모델':row['model'],'답변 수':row['attempts'],
            '형식 통과':f"{row['structure_valid_rate']:.0%}",
            '모든 인용 일치':f"{row['all_citations_valid_rate']:.0%}",
            '평균 인용 일치':f"{row['citation_match_rate_all_attempts']:.1%}",
            '평균 시간(초)':round(row['mean_wall_seconds'],1)})
    display(pd.DataFrame(summary_rows))
    print('인용 일치는 내용 정확성과 다릅니다. 시간에는 로딩·캐시 영향이 포함됩니다.')
    # 집계표와 별개로 실제 저장된 질문과 두 모델의 전체 답변을 표시합니다.
    model_records={}
    for role,key in [('기존 모델','baseline_run'),('추가 모델','candidate_run')]:
        run_path=Path(comparison[key])
        if not run_path.exists():
            run_path=PROJECT/'data/experiments'/run_path.name
        model_records[role]=[json.loads(line) for line in (run_path/'results.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    for pair in comparison['pairs']:
        display(Markdown('---'))
        print('모델별 답변 비교:',pair['case_id'],CONDITIONS[pair['condition']]['label'])
        for role,rows in model_records.items():
            matches=[r for r in rows if r['case_id']==pair['case_id'] and r['condition']==pair['condition'] and r.get('repeat',0)==pair.get('repeat',0)]
            if not matches:
                print(role,': 일치하는 실행 기록 없음')
                continue
            record=matches[0]
            print(role,record['config']['model'])
            report=record.get('report') or {}
            print('질문:',record['question'])
            answer_rows=[]
            if isinstance(report,dict) and report.get('summary'):
                answer_rows.append({'항목':'고객 요약','답변':report['summary'].get('text','')})
                for key,label in [('questions','확인 질문'),('actions','담당자 조치')]:
                    for claim in report.get(key,[]):
                        answer_rows.append({'항목':label,'답변':claim.get('text','')})
                for limitation in report.get('limitations',[]):
                    answer_rows.append({'항목':'한계','답변':limitation})
                with pd.option_context('display.max_colwidth',None):
                    display(pd.DataFrame(answer_rows))
            else:
                print('정상 리포트 생성 실패. 아래 원문에서 확인하세요.')
            status_labels={'checked':'자동 검사 통과 (내용 검토는 별도)',
                'needs_review':'인용 등 검토 필요','invalid_json':'JSON 해석 실패',
                'truncated':'출력 길이 초과','execution_error':'실행 오류'}
            print('검증 상태:',status_labels.get(record['status'],record['status']))
            details=json.dumps({'raw_text':record['raw_text'],'references':record['references'],
                                'automatic':record['automatic']},ensure_ascii=False,indent=2)
            display(HTML('<details><summary>근거·검증·원문 보기</summary><pre style="white-space:pre-wrap">'+escape(details)+'</pre></details>'))
else:
    print('실제 모델 비교 결과가 없습니다. compare_models.py를 실행하세요.')""")

md("""## 13. 실험 기록 저장과 발표 결론
ML 성능 향상이나 실제 이탈 감소를 주장하지 않습니다.
네 조건의 원문·검증 실패·수동 평가를 근거로 RAG 효과와 프롬프트 효과를 각각 설명합니다.
작은 사례 실행은 통합 동작 확인이며 일반화된 우수성을 입증하지 않습니다.
모델 로딩·캐시·실행 순서에 따라 시간이 달라지므로 load_duration도 확인하세요.""")
code("""output=PROJECT/'data/notebook_outputs'
output.mkdir(parents=True,exist_ok=True)
artifact={'run_id':selected_run.name,'automatic_summary':summarize(records),
          'manual_summary':score_reviews(reviews),'generation_mode':'llm_direct',
          'note':'저장된 실제 실험을 분석한 노트북 결과. 자동 지표는 내용 정확도가 아니다.'}
(output/'llm_experiment_overview.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
(output/'llm_report.md').write_text(llm_report_markdown(result),encoding='utf-8')
print(output/'llm_experiment_overview.json')""")

# 읽는 순서를 안내하고, 비교에 필요하지 않은 개발 출력만 숨긴다.
short_titles={
    1:'환경 설정',2:'고객 참고 정보 준비',3:'상담 지침 준비',4:'비교 조건 검증',
    5:'프롬프트 준비',6:'리포트 불러오기',7:'실험 결과 불러오기',8:'조건별 답변 준비',
    9:'평가 상태 확인',10:'검증 로직 확인',11:'검색 평가 확인',12:'모델 비교 결과',13:'결과 저장'}
explanations={
    1:'**할 일:** 커널을 `ML/.venv/Scripts/python.exe`로 선택하고 위에서부터 모두 실행하세요.\n\n`RUN_GENERATION=False`, `RUN_COMPARISON=False`이면 저장 결과를 읽습니다. True로 바꾸면 Ollama로 새 답변을 생성합니다. 두 모델 비교는 12번에서 기존 비교 실행 기록을 읽습니다.',
    2:'**입력:** 고객의 계약·요금·이용 기간과 기존 ML의 점수·SHAP을 준비합니다. ML은 참고 정보를 제공하며, 이번 비교 대상은 LLM 답변입니다.',
    3:'**RAG란?** 질문과 관련된 상담 지침을 찾아 LLM에 함께 제공하는 방식입니다. 여기서는 시연용 가상 지침을 준비합니다. 긴 문서 목록 출력은 생략합니다.',
    4:'**비교 설계:** RAG 없음/있음 × 기본/개선 프롬프트의 네 조건입니다. 같은 조건끼리 Qwen 1.5B와 3B를 비교합니다. 아래 표에서 조건 이름을 확인하세요.',
    5:'**프롬프트란?** LLM에 주는 작성 지시입니다. 개선 지시에는 근거 인용, 없는 정책 생성 금지, 모르는 내용 확인 필요 표시가 포함됩니다. 전체 지시문 출력은 생략합니다.',
    6:'**실행 준비:** 선택한 모델의 리포트를 생성하거나 저장 결과에서 읽습니다. 답변은 중복을 줄이기 위해 12번에서 모아 봅니다.',
    7:'**결과 준비:** 같은 고객·질문의 네 조건 결과를 불러옵니다. 저장 결과를 읽는 것은 새 LLM 호출과 다릅니다.',
    8:'**답변 확인 위치:** 12번에서 각 조건의 질문과 두 모델의 답변을 읽으세요. 요약·확인 질문·담당자 조치·한계로 구분됩니다.',
    9:'**평가 읽는 법:** 자동 검사는 형식과 인용 일치를 확인합니다. 사실 정확성과 질문 대응은 사람이 채점합니다. 미평가는 점수가 아직 없다는 뜻입니다. 아래 표는 현재 선택 모델의 실험에 대한 평가 상태입니다.',
    10:'**내부 검증:** 존재하지 않는 인용을 검출하는지 확인합니다. 테스트용 복사본만 사용하며 실제 답변은 바꾸지 않습니다. 상세 출력은 생략합니다.',
    11:'**보조 실험:** 검색 성능 기록을 확인합니다. 검색이 잘된다는 사실만으로 답변도 정확하다고 결론 내릴 수 없습니다. 상세 출력은 생략합니다.',
    12:'**핵심 결과:** 먼저 모델별 수치 표를 보고, 아래에서 같은 조건의 두 답변을 비교하세요.\n\n- **형식 통과:** 필요한 리포트 구조를 갖췄는가?\n- **모든 인용 일치:** 한 답변의 인용이 모두 제공 근거와 일치하는가?\n- **평균 인용 일치:** 답변별 인용 일치 비율의 평균입니다.\n- **평균 시간:** 실제 처리 시간이며 로딩·캐시 영향을 받습니다.\n\n모델의 답변은 정답이 아닙니다. 잘못된 단위·추측한 정책·질문 누락이 있는지 직접 읽어보세요. 긴 기록은 펼치기로 남겼습니다.',
    13:'**현재 해석:** 이 노트북은 LLM 호출·RAG·프롬프트·모델 비교가 작동하는지 확인하는 예비 실험입니다. 12번 표의 인용 결과와 실제 답변을 함께 판단하세요. 내용 평가가 끝나기 전에는 어느 모델이 더 정확하다고 결론 내리지 않습니다.\n\n**다음 작업:** 사례와 반복을 늘리고, 답변별 내용 평가를 채운 뒤 같은 조건에서 개선 여부를 비교합니다.'}
section=0
for cell in cells:
    if cell['cell_type']=='markdown':
        text=cell['source']
        if text.startswith('# 고객 상담'):
            cell['source']='# 고객 상담 LLM 비교 실험\n\n**목적:** 같은 고객 질문에 RAG·프롬프트·생성 모델이 답변을 어떻게 바꾸는지 확인합니다.\n\n**읽는 순서:** 1번 실행 설정 → 2~5번 입력·비교 방법 → 9번 평가 기준 → **12번 모델별 답변과 수치** → 13번 해석.\n\n기본 실행은 저장된 실제 결과를 읽습니다. 준비 코드도 순서대로 실행해야 합니다. JSON은 개발 기록이므로 펼치지 않아도 결과를 이해할 수 있습니다.'
        elif text.startswith('## '):
            section=int(text.split('.')[0].replace('## ','').strip())
            cell['source']='## '+str(section)+'. '+short_titles[section]+'\n\n'+explanations[section]
    elif section not in (1,2,4,9,12,13):
        # 별도 전역 네임스페이스를 만들지 않아 뒤 셀에서도 준비 변수를 사용한다.
        cell['source']='from IPython.utils.capture import capture_output\nwith capture_output():\n'+''.join('    '+line+'\n' for line in cell['source'].splitlines())
        cell['metadata']['jupyter']={'source_hidden':True}
    elif section==1:
        cell['source']='# 설정: False=저장 결과 보기, True=새 생성\n'+cell['source']

path=ROOT/'notebooks/archive/churn_llm_analysis.ipynb'
backup=ROOT/'notebooks/archive/churn_llm_analysis_v1.ipynb'
if path.exists() and not backup.exists():
    backup.parent.mkdir(exist_ok=True)
    backup.write_bytes(path.read_bytes())
nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python (ML 가상환경 선택)','language':'python','name':'python3'},
                             'language_info':{'name':'python','version':'3.12.6'}},nbformat=4,nbformat_minor=5)
path.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(path)
