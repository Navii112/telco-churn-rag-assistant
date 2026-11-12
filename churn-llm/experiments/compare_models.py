"""기존 실험을 기준으로 생성 모델만 바꿔 실행하고 통제 조건을 확인합니다."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from dataclasses import replace
import json
from pathlib import Path

from runtime.llm_reports import GenerationConfig, CONDITIONS
from experiments.run_experiments import run_comparison, save_json
from runtime.service import ROOT, ChurnService


def read_records(folder):
    return [json.loads(s) for s in (Path(folder)/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]


def verify_controls(baseline, candidate):
    def keyed(records):
        result = {(r['case_id'], r['repeat'], r['condition']): r for r in records}
        if len(result) != len(records):
            raise ValueError('중복 실험 키')
        return result
    left, right = keyed(baseline), keyed(candidate)
    if not left or left.keys() != right.keys():
        raise ValueError('사례·반복·조건이 일치하지 않습니다. 일부 성공 결과만 골라 비교하지 마세요.')
    for key, a in left.items():
        b = right[key]
        for field in ('question', 'customer_hash', 'corpus_hash', 'schema_hash', 'prompt_version', 'messages', 'retrieval_method'):
            if a[field] != b[field]:
                raise ValueError(f'통제 조건 불일치: {key} {field}')
        if {k:v for k,v in a['config'].items() if k != 'model'} != {k:v for k,v in b['config'].items() if k != 'model'}:
            raise ValueError('생성 설정이 모델 외에도 다릅니다.')
        if a['config']['model'] == b['config']['model']:
            raise ValueError('서로 다른 모델을 선택하세요.')
    return True


def model_summary(records):
    n = len(records)
    def seconds(field):
        values = [r['generation'].get(field) for r in records if isinstance(r['generation'].get(field), (int, float))]
        return sum(values)/len(values)/1e9 if values else None
    rows = [{'model': records[0]['config']['model'], 'attempts': n,
             'json_valid_rate': sum(r['automatic']['json_valid'] for r in records)/n,
             'structure_valid_rate': sum(r['automatic']['structure_valid'] for r in records)/n,
             'complete_report_rate': sum(r['automatic']['report_complete'] for r in records)/n,
             'all_citations_valid_rate': sum(r['automatic']['all_citations_valid'] for r in records)/n,
             'citation_match_rate_all_attempts': sum(r['automatic']['citation_match_rate'] or 0 for r in records)/n,
             'mean_wall_seconds': sum(r['wall_seconds'] for r in records)/n,
             'mean_load_seconds': seconds('load_duration'),
             'mean_prompt_eval_seconds': seconds('prompt_eval_duration'),
             'mean_generation_seconds': seconds('eval_duration')}]
    return rows[0]


def export_comparison(baseline_dir, candidate_dir, output):
    baseline, candidate = read_records(baseline_dir), read_records(candidate_dir)
    verify_controls(baseline, candidate)
    summary = [model_summary(baseline), model_summary(candidate)]
    pair = []
    for a in baseline:
        b = next(r for r in candidate if (r['case_id'],r['repeat'],r['condition']) == (a['case_id'],a['repeat'],a['condition']))
        pair.append({'case_id': a['case_id'], 'repeat': a['repeat'], 'condition': a['condition'],
                     'baseline_status': a['status'], 'candidate_status': b['status'],
                     'baseline_citation_rate': a['automatic']['citation_match_rate'],
                     'candidate_citation_rate': b['automatic']['citation_match_rate'],
                     'baseline_seconds': a['wall_seconds'], 'candidate_seconds': b['wall_seconds'],
                     'baseline_raw_text': a['raw_text'], 'candidate_raw_text': b['raw_text']})
    result = {'controls_verified': True, 'baseline_run': str(Path(baseline_dir).resolve()),
              'candidate_run': str(Path(candidate_dir).resolve()), 'summary': summary, 'pairs': pair,
              'notes': ['동일 사례·조건·고객·문서·프롬프트·schema·생성 설정인지 검사했습니다. 모델만 다릅니다.',
                        '자동 인용 일치는 내용 정확성이나 환각률이 아닙니다. 원문과 수동 평가를 함께 확인하세요.',
                        '실패도 전체 시도 분모에 포함합니다. 인용 비율이 없는 실패는 전체 평균에서 0으로 집계합니다.',
                        '시간은 냉간/웜 로딩 및 캐시 영향을 받습니다. 실제 generation.load_duration도 확인하세요.',
                        '작은 개발 사례 예비 비교이며 일반화 성능이나 최종 모델 우수성의 증명이 아닙니다.']}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'comparison.json', result)
    lines = ['# 생성 모델 예비 비교', '', *result['notes'], '',
             '| 모델 | 시도 | 구조 통과 | 인용 전체 통과 | 평균 인용 일치 | 평균 시간(초) |',
             '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['model']} | {r['attempts']} | {r['structure_valid_rate']:.1%} | {r['all_citations_valid_rate']:.1%} | {r['citation_match_rate_all_attempts']:.1%} | {r['mean_wall_seconds']:.1f} |")
    for p in pair:
        lines += ['', '## '+p['case_id']+' / '+p['condition'], '', '### 기존 모델 원문', '',
                  '```json', p['baseline_raw_text'], '```', '', '### 추가 모델 원문', '', '```json', p['candidate_raw_text'], '```']
    (output/'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate-model', default='qwen2.5:3b')
    parser.add_argument('--candidate-run', type=Path, help='이미 실행한 추가 모델 결과가 있으면 재사용')
    parser.add_argument('--output', type=Path, default=ROOT/'data/model_comparison')
    args = parser.parse_args()
    manifest = json.loads((args.baseline/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed':
        raise ValueError('완료된 기준 실험이 필요합니다.')
    if args.candidate_run:
        candidate_dir = args.candidate_run
    else:
        config = replace(GenerationConfig(**manifest['config']), model=args.candidate_model)
        candidate_dir, _ = run_comparison(ChurnService(), manifest['cases'], method=manifest['method'],
            config=config, repeats=manifest['repeats'],
            on_result=lambda r,n,total: print(f"{n}/{total}: {r['condition']} -> {r['status']}", flush=True))
    result = export_comparison(args.baseline, candidate_dir, args.output)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2), flush=True)
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
