"""개선 전후 비교 전용 노트북. 기존 노트북은 수정하지 않는다."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import uuid

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(text):
    cells.append(dict(id=uuid.uuid4().hex[:8],cell_type='markdown',metadata={},source=text))
def code(text):
    cells.append(dict(id=uuid.uuid4().hex[:8],cell_type='code',metadata={},source=text,execution_count=None,outputs=[]))

md('''# 고객 상담 LLM — 품질 개선 전후 비교

**목적:** 기존의 낮은 인용 일치 문제를 개선하고, 실제 답변이 어떻게 달라졌는지 확인합니다.

이 파일은 새 개선 버전입니다. 기존 `churn_llm_analysis.ipynb`와 실험 원본은 그대로 보존합니다.
커널은 **ML/.venv/Scripts/python.exe**를 선택하고 위에서부터 실행하세요. 기본 실행은 실제 저장 결과를 읽으며 LLM을 새로 호출하지 않습니다.''')
md('''## 1. 무엇을 개선했나?

| 기존 | 개선 |
|---|---|
| 고객 정보가 영어 필드의 JSON으로 전달됨 | 한국어 항목과 개월·USD 단위를 명시 |
| 긴 근거 문서에서 직접 인용문 생성 | 짧은 근거 단위와 간단한 ID 제공 |
| LLM이 답변·근거 ID·인용문을 모두 작성 | LLM은 답변·ID를 작성하고 코드는 해당 원문 연결 |
| 주로 형식·인용 검사 | 제한적인 기간·금액 오류 검사 추가 |

**중요:** 인용문 연결은 코드의 개선입니다. 인용 일치가 높아져도 답변이 근거에 맞는다는 뜻은 아닙니다.
유효 ID를 강제하는 enum은 사용하지 않아 잘못된 ID 선택도 실패로 남깁니다. 재시도·정답 대체는 하지 않습니다.''')
code('''import json,sys
from pathlib import Path
import pandas as pd
from IPython.display import display,Markdown,HTML
from html import escape
PROJECT=next((p if (p/'experiments/improve_quality.py').exists() else p/'churn-llm'
    for p in [Path.cwd(),*Path.cwd().parents]
    if (p/'experiments/improve_quality.py').exists() or (p/'churn-llm/experiments/improve_quality.py').exists()),None)
if PROJECT is None: raise FileNotFoundError('LLM 프로젝트 폴더에서 실행하세요.')
sys.path.insert(0,str(PROJECT))
from runtime.llm_reports import CONDITIONS
from experiments.improve_quality import fact_checks
runs=sorted([p for p in (PROJECT/'data/quality_experiments').glob('*')
    if (p/'manifest.json').exists() and json.loads((p/'manifest.json').read_text(encoding='utf-8'))['status']=='completed'],reverse=True)
if not runs: raise FileNotFoundError('먼저 improve_quality.py를 실행하세요.')
RUN=runs[0]
pairs=[json.loads(s) for s in (RUN/'pairs.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
for pair in pairs:
    # 기존 출력에도 같은 검사를 적용하되 저장 원본은 변경하지 않습니다.
    pair['before']['fact_flags']=fact_checks(pair['before'].get('report'),pair['after']['references'])
print('실제 개선 실험:',RUN.name,'/ 개선 후 호출 수:',len(pairs))
print('이번 노트북 실행에서는 새 모델 호출을 하지 않습니다.')''')
md('''## 2. 비교 조건과 읽는 법

기존과 동일한 고객·질문·검색 문서·모델·생성 설정을 사용합니다. 개선 후에는 동일 문서를 짧게 변환해 전달합니다.
두 모델 × RAG 유무 × 기본/개선 지시의 기존 8개 실행을 각각 한 번 재실행했습니다.

입력 표현·schema·인용 처리·프롬프트가 함께 바뀌므로 **전체 시스템 개선 전후 비교**입니다.
여러 개선 중 어떤 요소가 효과를 냈는지 분리해서 증명하는 실험은 아닙니다. 단일 사례의 예비 비교입니다.''')
code('''controls=[]
from runtime.llm_reports import digest
for pair in pairs:
    b,a=pair['before'],pair['after']
    assert b['config']==a['config'] and b['question']==a['question']
    assert b['condition']==a['condition'] and b['trial_id']==a['trial_id']
    assert digest(b['references'])==a['baseline_reference_hash']
    controls.append({'모델':a['config']['model'],'조건':CONDITIONS[a['condition']]['label'],'고객 질문':a['question']})
display(pd.DataFrame(controls))
print('모델·질문·생성 설정·원본 근거 일치 확인 완료')''')
md('''## 3. 수치로 보는 개선 전후

- **형식 통과:** 필요한 구조를 갖춘 답변 비율.
- **인용 연결 일치:** 선택 ID가 존재하고 표시 인용문이 근거 원문과 일치하는 정도. 개선 후에는 코드 연결을 포함한 지표입니다.
- **기간·금액 경고:** 일부 숫자와 단위 오류만 검사합니다. 경고 없음이 정확성 통과를 뜻하지 않습니다.
- **내용 정확성:** 사람의 평가가 필요하며 현재 자동 수치로 대체하지 않습니다.

실패한 실행도 분모에 포함합니다. 두 버전의 인용 작성 방식이 달라 같은 능력의 점수로 해석하면 안 됩니다.''')
code('''summary=[]
for model in sorted({p['before']['config']['model'] for p in pairs}):
    selected=[p for p in pairs if p['before']['config']['model']==model]
    for key,label in [('before','기존: LLM 직접 인용'),('after','개선: 코드 인용 연결')]:
        rows=[p[key] for p in selected]
        summary.append({'모델':model,'버전':label,'시도':len(rows),
            '형식 통과율':sum(r['automatic']['structure_valid'] for r in rows)/len(rows),
            '평균 인용 연결 일치율':sum(r['automatic']['citation_match_rate'] or 0 for r in rows)/len(rows),
            '모든 인용 연결 통과':sum(r['automatic']['all_citations_valid'] for r in rows)/len(rows),
            '평균 시간(초)':sum(r['wall_seconds'] for r in rows)/len(rows),
            '기간·금액 경고':sum(bool(r.get('fact_flags')) for r in rows)})
display(pd.DataFrame(summary).style.format({'형식 통과율':'{:.0%}','평균 인용 연결 일치율':'{:.1%}',
    '모든 인용 연결 통과':'{:.0%}','평균 시간(초)':'{:.1f}'}))
detail=[]
for p in pairs:
    b,a=p['before'],p['after']
    detail.append({'모델':a['config']['model'],'조건':CONDITIONS[a['condition']]['label'],
        '기존 인용 연결':f"{b['automatic']['citation_match_rate'] or 0:.0%}",
        '개선 인용 연결':f"{a['automatic']['citation_match_rate'] or 0:.0%}",
        '개선 실행 상태':a['status'],'숫자 경고':' / '.join(a['fact_flags']) or '감지 없음 (정확성 보장 아님)'})
display(pd.DataFrame(detail))''')
md('''## 4. 실제 질문과 답변 비교 — 핵심 결과

각 조건에서 기존과 개선 답변을 나란히 읽으세요. 같은 뜻을 반복하는지, 고객 기간을 잘못 바꾸는지,
없는 계약 조건이나 혜택을 만드는지 확인합니다. **인용 연결 통과와 답변 내용 적절성은 별개**입니다.
근거·생성 원문은 펼치기로 확인할 수 있습니다.''')
code('''def answer_items(record):
    report=record.get('report') or {}
    if not isinstance(report,dict): return {'생성 오류':record.get('error') or record['status']}
    items={'고객 요약':(report.get('summary') or {}).get('text','생성 실패')}
    for key,label in [('questions','확인 질문'),('actions','담당자 조치')]:
        items[label]=' / '.join(c.get('text','') for c in report.get(key,[]))
    items['한계']=' / '.join(report.get('limitations',[]))
    return items
for pair in pairs:
    b,a=pair['before'],pair['after']
    display(Markdown('### '+a['config']['model']+' — '+CONDITIONS[a['condition']]['label']))
    print('질문:',a['question'])
    before,after=answer_items(b),answer_items(a)
    with pd.option_context('display.max_colwidth',None):
        display(pd.DataFrame([{'항목':key,'기존 답변':before.get(key,''),'개선 답변':after.get(key,'')}
                             for key in dict.fromkeys([*before,*after])]))
    print('기존 기간·금액 경고:',b['fact_flags'] or '감지 없음')
    print('개선 후 상태:',a['status'],'/ 기간·금액 경고:',a['fact_flags'] or '감지 없음')
    details=json.dumps({'기존 원문':b['raw_text'],'개선 원문':a['raw_text'],
        '개선 연결 리포트':a['report'],'개선 근거':a['references'],'자동 검사':a['automatic']},ensure_ascii=False,indent=2)
    display(HTML('<details><summary>근거와 원문 보기</summary><pre style="white-space:pre-wrap">'+escape(details)+'</pre></details>'))''')
md('''## 5. 이 결과로 무엇을 말할 수 있나?

**확인 가능한 것:** ID 선택과 인용문 연결을 분리했을 때 인용 연결 오류가 얼마나 줄었는지, 실제 답변과 시간의 변화.

**아직 결론 내릴 수 없는 것:** 답변 정확성 향상, 최적 모델, 순수 RAG 효과, 실제 상담 효과.
이를 확인하려면 여러 질문·고객과 반복 실험, 근거가 주장을 뒷받침하는지에 대한 내용 평가가 필요합니다.
숫자 검사는 모든 사실 오류를 찾지 못하며, 긴 문장의 기계적 분할이 문맥을 끊을 수도 있습니다.

실행 시간이 달라도 로딩·캐시와 실험 시점이 다르므로 모델 속도의 일반적 우열로 단정하지 않습니다.''')
code('''print('내용 평가 양식:',RUN/'manual_review.csv')
print('현재 점수는 미평가입니다. 평가자가 고객 정보·근거와 대조해 채점하고 reviewer를 기록하세요.')
print('기존 노트북과 실험 원본은 수정하지 않았습니다.')''')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python (ML 가상환경)','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12.6'}},'nbformat':4,'nbformat_minor':5}
path=ROOT/'notebooks/archive/churn_llm_quality_improved.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(path)
