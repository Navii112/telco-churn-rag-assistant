"""20질문 평가의 실제 사람 채점만 집계한다. 빈 값은 미평가로 남긴다."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import csv
import json
from html import escape

FIELDS={'factual_accuracy_0_2':2,'evidence_support_0_2':2,'relevance_0_2':2,
        'policy_invention_0_1':1,'abstention_appropriate_0_1':1}


def manual_summary(folder):
    mapping={r['review_id']:r['trial_id'] for r in json.loads((folder/'review_mapping.json').read_text(encoding='utf-8'))}
    with (folder/'human_review.csv').open(encoding='utf-8-sig',newline='') as file: reviews=list(csv.DictReader(file))
    seen=set(); groups={}
    for row in reviews:
        key=row['review_id']
        if key in seen or key not in mapping: raise ValueError('중복 또는 미등록 평가 ID: '+key)
        seen.add(key)
        trial=mapping[key]; model,case,version=trial.rsplit(':',2)
        group=groups.setdefault((model,version),{'attempts':0,'complete':[]})
        group['attempts']+=1; values={}
        for field,high in FIELDS.items():
            raw=str(row.get(field,'')).strip()
            if not raw: continue
            value=float(raw)
            if not value.is_integer() or not 0<=value<=high: raise ValueError(field+' 범위 오류')
            values[field]=int(value)
        if len(values)==len(FIELDS) and row.get('reviewer','').strip(): group['complete'].append(values)
    output=[]
    for (model,version),group in groups.items():
        n=len(group['complete'])
        output.append({'model':model,'version':version,'attempts':group['attempts'],'reviewed':n,
            'completion_rate':n/group['attempts'],
            **{field:sum(r[field] for r in group['complete'])/n if n else None for field in FIELDS}})
    return output


def export_review_packet(folder):
    """평가자에게 모델 이름 대신 질문·답변·원자료·기준을 함께 제공한다."""
    mapping=json.loads((folder/'review_mapping.json').read_text(encoding='utf-8'))
    records={r['trial_id']:r for r in [json.loads(s) for s in (folder/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]}
    packet=[]
    for item in mapping:
        row=records[item['trial_id']]
        from experiments.benchmark20 import answer_text
        packet.append({'review_id':item['review_id'],'question':row['case']['question'],
            'criteria':row['case']['gold'],'answer':answer_text(row['result']),
            'execution_status':row['result']['status'],'execution_error':row['result'].get('error'),
            'references':row['result']['references'],'report':row['result']['report']})
    path=folder/'blind_review_packet.json'
    path.write_text(json.dumps(packet,ensure_ascii=False,indent=2),encoding='utf-8')
    html=['<!doctype html><meta charset="utf-8"><title>내용 평가 자료</title>',
        '<style>body{font-family:system-ui;max-width:1000px;margin:30px auto}pre{white-space:pre-wrap}details{padding:12px;border:1px solid #ddd;margin:8px 0}</style>',
        '<h1>내용 평가 자료</h1><p>모델·버전 이름을 숨겼습니다. 출력 형식으로 버전을 짐작할 수 있어 완전한 맹검은 아닙니다.</p>',
        '<p>고객 tenure는 이용 개월 수이며 계약 기간과 다릅니다. 월·누적 요금 통화는 USD입니다. 예측 점수는 보정된 미래 확률이 아니고 SHAP은 실제 원인을 증명하지 않습니다. 지침은 가상 시연용입니다.</p>',
        '<p>human_review.csv에 같은 review_id로 점수와 reviewer를 기록하세요. 빈 점수는 미평가입니다.</p>']
    for item in packet:
        html.append('<details><summary>'+escape(item['review_id']+' — '+item['question'])+'</summary><p>실행 상태: '+escape(item['execution_status'])+'</p><h3>답변</h3><pre>'+escape(item['answer'] or item['execution_error'] or '(응답 없음)')+'</pre><h3>평가 기준</h3><pre>'+escape(json.dumps(item['criteria'],ensure_ascii=False,indent=2))+'</pre><h3>제공 근거</h3><pre>'+escape(json.dumps(item['references'],ensure_ascii=False,indent=2))+'</pre></details>')
    (folder/'blind_review_packet.html').write_text('\n'.join(html),encoding='utf-8')
    return path


if __name__=='__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    parser=ArgumentParser(); parser.add_argument('folder',type=Path)
    args=parser.parse_args(); print(export_review_packet(args.folder))
