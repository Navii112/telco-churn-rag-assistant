"""생성 기록 보존, 고객 필드 표현 인식 개선을 동일하게 적용한 사후 평가 v1.1."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from copy import deepcopy
import json
import re
from pathlib import Path
from experiments.benchmark20 import score_reference,summarize,save
from runtime.llm_reports import digest


def duration_months(text):
    return sum(float(n)*(12 if unit=='년' else 1) for n,unit in re.findall(r'(\d+(?:\.\d+)?)\s*(년|개월)',text))


def field_mentions(text,field):
    if field=='tenure':
        # 계약 기간·향후 계획은 고객 이용 기간으로 간주하지 않는다.
        pattern=r'(?:이용|사용|가입|tenure)[^\d\n.!?]{0,12}((?:\d+(?:\.\d+)?\s*(?:년|개월)\s*)+)(?:\s*\(([^)]{0,40})\))?'
        found=[]
        for match in re.finditer(pattern,text,re.I):
            found.append(duration_months(match.group(1)))
            if match.group(2) and re.search(r'\d+\s*(?:년|개월)',match.group(2)):
                found.append(duration_months(match.group(2)))
        # “69개월 동안 이용”처럼 값 뒤에 항목이 나오는 표현.
        for match in re.finditer(r'((?:\d+(?:\.\d+)?\s*(?:년|개월)\s*)+)(?:동안|째|간)?\s*(?:서비스를?\s*)?(?:이용|사용)',text):
            found.append(duration_months(match.group(1)))
        return found
    label=r'(?:월\s*(?:요금|청구액)|MonthlyCharges)' if field=='MonthlyCharges' else r'(?:(?:누적|총)\s*(?:요금|청구액)|TotalCharges)'
    pattern=label+r'\s*(?:은|는|이|가|:)?\s*(?:약|대략|정확히)?\s*(?P<prefix>USD|달러|\$)?\s*(?P<number>\d[\d,]*(?:\.\d+)?)\s*(?P<suffix>USD|달러|원)?'
    found=[]
    for match in re.finditer(pattern,text,re.I):
        currency=match.group('suffix') or match.group('prefix')
        if currency and currency.upper() in ('USD','달러','$'):
            found.append(float(match.group('number').replace(',','')))
    return found


def rescore(case,result):
    from experiments.benchmark20 import answer_text
    score=score_reference(case,result)
    text=answer_text(result); checks=[]
    for field,value in case['gold']['required_fields'].items():
        observed=field_mentions(text,field)
        checks.append({'field':field,'expected':value,'observed':observed,
            'state':'누락/인식 불가' if not observed else ('일치' if all(abs(x-value)<1e-6 for x in observed) else '불일치')})
    score.update(reference_fields=checks,matched_count=sum(c['state']=='일치' for c in checks),
        missing_count=sum(c['state']=='누락/인식 불가' for c in checks),
        mismatch_count=sum(c['state']=='불일치' for c in checks),evaluation_version='reference-v1.1')
    return score


def evaluate(folder):
    folder=Path(folder)
    rows=[json.loads(s) for s in (folder/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    output=[]
    for record in rows:
        row=deepcopy(record); row['metrics']=rescore(row['case'],row['result']); output.append(row)
    target=folder/'evaluated_results_v1_1.jsonl'
    if target.exists(): raise FileExistsError('재평가 원본 보존: '+str(target))
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output),encoding='utf-8')
    save(folder/'summary_v1_1.json',summarize(output))
    save(folder/'evaluation_manifest_v1_1.json',{'version':'reference-v1.1',
        'input_hash':digest((folder/'results.jsonl').read_text(encoding='utf-8')),
        'scoring_code_hash':digest(Path(__file__).read_text(encoding='utf-8')),
        'notes':['질문·정답 기준·생성 원문은 수정하지 않음.',
            '계약 기간과 이용 기간 분리, 괄호 안 년·개월 환산 확인, 약/쉼표/접두 통화 표현 인식.',
            '출력을 본 뒤 보완한 검사로 모든 모델·버전에 동일 적용. 전체 의미 정확성 검사가 아님.']})
    print(target)


if __name__=='__main__':
    from argparse import ArgumentParser
    parser=ArgumentParser(); parser.add_argument('folder',type=Path)
    evaluate(parser.parse_args().folder)
