"""다중 근거·내용 검사 개선 전후를 읽기 쉽게 보여주는 별도 노트북."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import uuid

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(text): cells.append(dict(id=uuid.uuid4().hex[:8],cell_type='markdown',metadata={},source=text))
def code(text): cells.append(dict(id=uuid.uuid4().hex[:8],cell_type='code',metadata={},source=text,execution_count=None,outputs=[]))

md('''# 고객 상담 LLM — 다중 근거와 내용 검사 개선

**읽는 순서:** 개선 방법 → 비교 수치 → 실제 답변 → 주장별 경고 → 새 질문 체험.
기존 두 노트북은 보존했습니다. 이 노트북은 저장된 실제 생성 결과를 읽습니다.
커널은 **ML/.venv/Scripts/python.exe**, 실행은 위에서부터 모두 실행하세요.

**답변을 제대로 하는가?** 자료에 있는 정보로 답하도록 개선했지만 정확성을 보장하지는 않습니다.
없는 정보는 확인 필요로 답해야 합니다. 규칙 검사로 잡지 못하는 오류는 사람이 검토해야 합니다.''')
md('''## 1. 이번 개선 내용

| 변경 | 기대 효과 |
|---|---|
| `evidence_ids: ["E001", "E003"]` 배열 | 한 주장에 여러 근거 연결 |
| 이전 쉼표 문자열도 분리·중복 제거 | 이전 출력의 형식 문제와 실제 근거 오류 구분 |
| 문장 중간을 자르지 않고 제공 | 근거 문맥 보존 |
| 이용 기간 ≠ 계약 기간 명시 | 같은 숫자를 다른 의미로 해석하는 오류 억제 |
| 기간·금액·통화·추측 지시·일부 근거 누락 검사 | 내용 오류 후보를 주장별로 표시 |

없는 ID를 추정해서 고치거나 잘못된 답변을 정답으로 교체하지 않습니다. 원문과 실패를 모두 보존합니다.''')
code('''import json,sys
from pathlib import Path
from html import escape
import pandas as pd
from IPython.display import display,Markdown,HTML
PROJECT=next((p if (p/'runtime/quality_v2.py').exists() else p/'churn-llm'
    for p in [Path.cwd(),*Path.cwd().parents]
    if (p/'runtime/quality_v2.py').exists() or (p/'churn-llm/runtime/quality_v2.py').exists()),None)
if PROJECT is None: raise FileNotFoundError('프로젝트 폴더에서 실행하세요.')
sys.path.insert(0,str(PROJECT))
from runtime.quality_v2 import assess,claims_of,normalize_ids,ask_question
from runtime.llm_reports import CONDITIONS,digest
runs=sorted([p for p in (PROJECT/'data/quality_v2_experiments').glob('*')
    if (p/'manifest.json').exists() and (p/'evaluated_triples_rules-v2.2.jsonl').exists()
    and json.loads((p/'manifest.json').read_text(encoding='utf-8'))['status']=='completed'],reverse=True)
if not runs: raise FileNotFoundError('runtime/quality_v2.py 실행과 --evaluate-only 재평가를 먼저 수행하세요.')
RUN=runs[0]
triples=[json.loads(s) for s in (RUN/'evaluated_triples_rules-v2.2.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
VERSIONS={'before':'최초 버전','v1':'1차 개선','v2':'2차 개선 (다중 근거)'}
for triple in triples:
    b,a=triple['before'],triple['v2']
    assert b['config']==a['config'] and b['question']==a['question'] and b['trial_id']==a['trial_id']
    assert digest(b['references'])==a['parent_reference_hash']
print('실제 생성 실험:',RUN.name,'/ 새 생성:',len(triples),'회')
print('노트북 실행 자체는 저장 결과를 읽습니다. 기존 고객·질문·모델·생성 설정 일치 확인 완료.')''')
md('''## 2. 비교 수치를 읽는 방법

세 버전의 입력 표현·지시문·schema가 달라 **전체 시스템 개선 비교**입니다. 순수 모델 학습 효과가 아닙니다.

- **이전 방식 인용 점수:** 당시 검사의 기록입니다. 최초는 직접 쓴 인용문, 1차는 단일 ID 연결을 검사했습니다.
- **같은 검사로 본 ID 연결률:** 모든 버전에 다중 ID 분리 처리를 적용한 재평가입니다. 원문 인용 정확도나 내용 정확도가 아닙니다.
- **내용 경고:** 일부 숫자·단위·근거 오류 후보를 검출한 주장 수입니다. 낮다고 더 정확하다는 뜻은 아닙니다. 숫자를 생략하면 검사 범위도 줄어듭니다.
- **내용 점수:** 아직 미평가입니다. 경고 없음은 정답 판정이 아닙니다.

예전의 3B 50% 중에는 실제로 존재하는 여러 ID를 한 문자열로 써서 실패한 항목이 포함됩니다.
같은 이전 답변을 새 파서로 재해석한 결과와, 새로 생성한 답변을 구분해서 보세요.''')
code('''rows=[]
for model in sorted({t['v2']['config']['model'] for t in triples}):
    selected=[t for t in triples if t['v2']['config']['model']==model]
    for key,label in VERSIONS.items():
        records=[t[key] for t in selected]; n=len(records)
        legacy=sum(r.get('automatic',{}).get('citation_match_rate') or 0 for r in records)/n if key!='v2' else None
        rows.append({'모델':model,'버전':label,'시도':n,
            '이전 방식 인용 점수':'해당 없음' if legacy is None else f'{legacy:.1%}',
            '동일 검사 ID 연결률':f"{sum(r['evaluation']['link_rate'] for r in records)/n:.1%}",
            '규칙 검사 대상 주장':sum(r['evaluation']['checked_claims'] for r in records),
            '내용 경고 주장':sum(r['evaluation']['flagged_claims'] for r in records),
            '평균 시간(초)':round(sum(r['wall_seconds'] for r in records)/n,1),'내용 평가':'미평가'})
display(pd.DataFrame(rows))
print('실패도 분모에 포함합니다. 모델 로딩·캐시·실험 시점이 달라 시간의 일반적 우열을 단정하지 않습니다.')''')
md('''## 3. 실제 질문과 세 버전의 답변

같은 조건에서 요약·확인 질문·담당자 조치를 나란히 읽으세요. 요약이 고객 사실인지,
질문이 사용자 요청에 맞는지, 조치에 추측이 들어가는지 확인합니다. 생성 실패도 표시합니다.''')
code('''def answer_parts(row):
    report=row.get('report')
    if not isinstance(report,dict): return {'오류':row.get('error') or row['status']}
    items={}
    for key,label in [('summary','고객 요약'),('questions','확인 질문'),('actions','담당자 조치')]:
        value=report.get(key,[])
        if isinstance(value,dict): value=[value]
        items[label]=' / '.join(c.get('text','') for c in value if isinstance(c,dict)) if isinstance(value,list) else '잘못된 구조'
    items['한계']=' / '.join(x for x in report.get('limitations',[]) if isinstance(x,str))
    return items
for triple in triples:
    a=triple['v2']
    display(Markdown('### '+a['config']['model']+' — '+CONDITIONS[a['condition']]['label']))
    print('질문:',a['question'])
    parts={key:answer_parts(triple[key]) for key in VERSIONS}
    fields=list(dict.fromkeys(field for group in parts.values() for field in group))
    with pd.option_context('display.max_colwidth',None):
        display(pd.DataFrame([{'항목':field,**{VERSIONS[key]:parts[key].get(field,'') for key in VERSIONS}} for field in fields]))
    print('2차 개선:',a['status'],'/ 내용 경고:',a['evaluation']['flagged_claims'],'건 / 의미 정확성: 미평가')
    details=json.dumps({key:{'원문':triple[key]['raw_text'],'검사':triple[key]['evaluation'],
        '제공 근거':triple[key]['references']} for key in VERSIONS},ensure_ascii=False,indent=2)
    display(HTML('<details><summary>근거·전체 검사·JSON 원문</summary><pre style="white-space:pre-wrap">'+escape(details)+'</pre></details>'))''')
md('''## 4. 내용 검증 — 어떤 문장이 문제인가?

아래는 규칙으로 발견한 **오류 후보**입니다. 먼저 원자료와 대조하세요.
이 검사는 표현에 따라 오탐·누락이 있으므로 판정 결과를 사람이 확인해야 합니다.
숫자 정보에 맞아도 근거의 의미가 적절한지는 미평가로 남깁니다.''')
code('''flags=[]
for triple in triples:
    for key,label in VERSIONS.items():
        row=triple[key]
        for check in row['evaluation']['content_checks']:
            if check['issues']:
                flags.append({'모델':row['config']['model'],'조건':CONDITIONS[row['condition']]['label'],
                    '버전':label,'위치':check['location'],'문장':check['text'],'경고':' / '.join(check['issues'])})
if flags:
    with pd.option_context('display.max_colwidth',None): display(pd.DataFrame(flags))
else: print('현재 규칙에서 경고가 감지되지 않았습니다. 내용이 검증됐다는 뜻은 아닙니다.')
print('주장별 수동 평가 양식:',RUN/'claim_review.csv')
display(pd.DataFrame([
    {'평가':'사실 정확성','기준':'0=틀림, 1=부분, 2=적절'},
    {'평가':'근거 충족','기준':'0=뒷받침 안 됨, 1=일부만, 2=충분'},
    {'평가':'질문 대응','기준':'0=무관, 1=일부, 2=적절'}]))
print('평가자가 답변과 실제 연결 근거를 함께 보고 점수와 reviewer를 입력합니다. 빈 점수는 미평가입니다.')''')
md('''## 5. 정보가 없는 질문의 실제 응답

할인·쿠폰·실제 이탈 원인은 자료에 없는 정보입니다. 해당 질문에는 모른다는 이유와 확인할 내용을 제시해야 합니다.
아래 응답의 `abstained`는 모델의 자기 표시일 뿐, 적절한 보류인지 확인한 정답 판정이 아닙니다.''')
code('''boundary=PROJECT/'data/quality_v2_boundary'
files=sorted(boundary.glob('*/results.jsonl'))
if not files:
    print('별도 경계 질문 실행 기록 없음. 아래 새 질문 기능으로 확인할 수 있습니다.')
else:
    records=[json.loads(s) for s in files[-1].read_text(encoding='utf-8').splitlines() if s.strip()]
    for row in records:
        print('질문:',row['question'])
        with pd.option_context('display.max_colwidth',None):
            display(pd.DataFrame([{'항목':k,'답변':v} for k,v in answer_parts(row).items()]))
        print('모델의 답변 보류 표시:',(row.get('report') or {}).get('abstained'),
              '/ 사람 평가: 미평가')
        current=assess(row['report'],row['references'],row['question'])
        warnings=[{'문장':c['text'],'경고':' / '.join(c['issues'])} for c in current['content_checks'] if c['issues']]
        if warnings:
            with pd.option_context('display.max_colwidth',None): display(pd.DataFrame(warnings))
        else: print('규칙 경고 감지 없음. 정답 판정이 아닙니다.')''')
md('''## 6. 내가 직접 질문하기 (선택)

`RUN_CUSTOM=True`로 바꾸고 질문을 수정하면 로컬 Ollama를 실제 호출합니다.
RAG와 개선 지시를 사용하며 새 고객 정보·관련 문서를 검색합니다. 결과와 검사는 별도 폴더에 저장합니다.
로컬 Streamlit의 주 리포트 화면도 같은 최신 생성·검사를 사용합니다. 화면의 이전 비교 실험은 별도로 보존합니다.
여러 질문을 이어가더라도 대화 기억은 없으며 각 질문을 독립적으로 처리합니다.''')
code('''RUN_CUSTOM=False
CUSTOM_QUESTION='이 고객의 계약 상담 전에 무엇을 확인해야 하나요?'
CUSTOM_MODEL='qwen2.5:1.5b'
if RUN_CUSTOM:
    custom,location=ask_question(CUSTOM_QUESTION,row_index=2280,model=CUSTOM_MODEL)
    print('질문:',custom['question'])
    with pd.option_context('display.max_colwidth',None):
        display(pd.DataFrame([{'항목':k,'답변':v} for k,v in answer_parts(custom).items()]))
    print('상태:',custom['status'],'/ 내용 경고:',custom['evaluation']['flagged_claims'])
    print('기록:',location)
    display(pd.DataFrame(custom['evaluation']['content_checks']))
else:
    print('새 질문 실행 꺼짐. 저장 결과만 표시했습니다.')''')
md('''## 7. 발표에서 설명할 결론

다중 근거와 기존 형식 호환을 구현했고, 연결 검사와 내용 검사를 분리했습니다.
규칙 검사는 일부 오류를 발견하는 도구이며 LLM의 전체 내용 정확성이나 환각률이 아닙니다.
높은 연결률만으로 모델 우열·상담 효과를 주장하지 않습니다.

**다음 단계:** 여러 고객·질문으로 반복 실험하고, 주장별 근거 충족·사실 정확성·질문 대응을 사람이 평가합니다.
평가에 사용한 질문은 개발 사례이므로 향후 별도의 미사용 질문에서도 확인해야 합니다.''')
path=ROOT/'notebooks/archive/churn_llm_multi_evidence.ipynb'
path.write_text(json.dumps({'cells':cells,'metadata':{'kernelspec':{
    'display_name':'Python (ML 가상환경)','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12.6'}},'nbformat':4,'nbformat_minor':5},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(path)
