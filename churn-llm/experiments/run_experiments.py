"""RAG×프롬프트 4조건 실제 실행. 원문/실패 포함 JSONL + 자동지표 + 수동평가 CSV."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import uuid

from runtime.llm_reports import CONDITIONS, SCHEMA, PROMPTS, COMMON, GenerationConfig, ReportClient, generate_report, prepare_context, digest
from runtime.service import ChurnService, ROOT


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def summarize(records):
    summary = []
    for condition in CONDITIONS:
        rows = [r for r in records if r['condition'] == condition]
        if not rows:
            continue
        def rate(key):
            return sum(bool(r['automatic'][key]) for r in rows) / len(rows)
        summary.append({'condition': condition, 'attempts': len(rows),
                        'json_valid_rate': rate('json_valid'), 'structure_valid_rate': rate('structure_valid'),
                        'complete_report_rate': rate('report_complete'), 'all_citations_valid_rate': rate('all_citations_valid'),
                        'mean_verified_citations_per_attempt': sum(r['automatic']['verified_citation_count'] for r in rows)/len(rows),
                        'mean_wall_seconds': sum(r['wall_seconds'] for r in rows)/len(rows),
                        'status_counts': dict(Counter(r['status'] for r in rows)),
                        'manual_semantic_accuracy': None})
    return summary


def run_comparison(service, cases, output_root=ROOT / 'data/experiments', method='embedding', config=None,
                   repeats=1, client=None, on_result=None):
    config = config or GenerationConfig()
    if repeats < 1 or not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('중복 없는 사례와 양수 반복 횟수가 필요합니다.')
    client = client or ReportClient(config)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    folder = Path(output_root) / run_id
    folder.mkdir(parents=True)
    manifest = {'run_id': run_id, 'status': 'running', 'method': method, 'conditions': CONDITIONS,
                'config': config.__dict__, 'output_schema': SCHEMA, 'schema_hash': digest(SCHEMA),
                'system_prompts': PROMPTS, 'common_format_instruction': COMMON, 'case_count': len(cases), 'repeats': repeats,
                'cases_hash': digest(cases), 'cases': cases,
                'design': '2x2 RAG on/off × basic/improved. 동일 고객/질문/모델/스키마/생성설정. 조건 실행순서는 seed로 섞음.',
                'notes': ['실패·잘린 출력·검증 실패도 시도 분모에 포함. 재시도/정제/템플릿 대체 없음.',
                          '자동 지표는 형식·인용 일치이며 의미적 정확성/환각률이 아님.',
                          'no-RAG에는 상담 문서를 주지 않으므로 지침 인용 지표를 단독 종합 성능으로 사용하지 말 것.',
                          '실행 시간은 모델 로딩·캐시·실행 순서의 영향을 받음. load_duration도 원본에 기록.',
                          '출력 schema 제약은 네 조건 공통. 자유 텍스트 출력과의 비교가 아님.']}
    save_json(folder / 'manifest.json', manifest)
    rows = []
    rng = random.Random(config.seed)
    try:
        with (folder / 'results.jsonl').open('w', encoding='utf-8') as file:
            for case in cases:
                prediction = service.predict(service.features.loc[case['row_index']].to_dict())
                context = prepare_context(service, prediction, case['question'], method)
                for repeat in range(repeats):
                    order = list(CONDITIONS)
                    rng.shuffle(order)
                    for condition in order:
                        result = generate_report(context, condition, client, config)
                        result.update(trial_id=f"{case['id']}:{repeat}:{condition}", case_id=case['id'],
                                      repeat=repeat, row_index=case['row_index'], review_focus=case.get('review_focus', ''),
                                      expected_abstain=case.get('expected_abstain'))
                        rows.append(result)
                        file.write(json.dumps(result, ensure_ascii=False) + '\n')
                        file.flush()
                        if on_result:
                            on_result(result, len(rows), len(cases)*repeats*4)
        manifest['status'] = 'completed'
    except Exception as exc:
        manifest['status'], manifest['error'] = 'failed', str(exc)
        raise
    finally:
        manifest['attempt_count'] = len(rows)
        save_json(folder / 'manifest.json', manifest)
        save_json(folder / 'summary.json', summarize(rows))
        fields = ['trial_id', 'condition', 'case_id', 'review_focus', 'expected_abstain',
                  'factual_accuracy_0_2', 'policy_invention_0_1', 'causal_overclaim_0_1',
                  'request_relevance_0_2', 'abstention_appropriate_0_1', 'reviewer', 'notes']
        with (folder / 'manual_review.csv').open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, '') for k in fields})
    return folder, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT / 'data/report_cases.jsonl')
    parser.add_argument('--limit', type=int, default=1, help='기본 1사례×4조건. 전체는 --limit 4')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--method', choices=['embedding', 'tfidf'], default='embedding')
    parser.add_argument('--model', default='qwen2.5:1.5b')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/experiments')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('limit은 양수여야 합니다.')
    cases = [json.loads(s) for s in args.cases.read_text(encoding='utf-8').splitlines() if s.strip()][:args.limit]
    folder, _ = run_comparison(ChurnService(), cases, args.output, args.method, GenerationConfig(model=args.model), args.repeats,
        on_result=lambda r, n, total: print(f"{n}/{total}: {r['case_id']} {r['condition']} -> {r['status']}", flush=True))
    print(folder, flush=True)


if __name__ == '__main__':
    main()
