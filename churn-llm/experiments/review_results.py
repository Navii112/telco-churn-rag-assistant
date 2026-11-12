"""수동 내용 평가 집계. 미입력은 0점으로 처리하지 않습니다."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import csv
from collections import defaultdict

RANGES = {'factual_accuracy_0_2': (0, 2), 'policy_invention_0_1': (0, 1),
          'causal_overclaim_0_1': (0, 1), 'request_relevance_0_2': (0, 2),
          'abstention_appropriate_0_1': (0, 1)}


def score_reviews(rows):
    groups = defaultdict(list)
    counts = defaultdict(int)
    ids = set()
    for row in rows:
        if row['trial_id'] in ids:
            raise ValueError('중복 평가 trial_id')
        ids.add(row['trial_id'])
        counts[row['condition']] += 1
        values = {}
        for key, (low, high) in RANGES.items():
            raw = str(row.get(key, '')).strip()
            if raw in ('', 'nan', 'None'):
                continue
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(key + ': 숫자만 입력하세요.') from exc
            if not low <= value <= high or not value.is_integer():
                raise ValueError(f'{key}: {low}~{high} 정수만 입력하세요.')
            values[key] = value
        if len(values) == len(RANGES) and str(row.get('reviewer', '')).strip() not in ('', 'nan', 'None'):
            groups[row['condition']].append(values)
    return [{'condition': key, 'attempts': n, 'reviewed': len(groups[key]),
             'review_completion_rate': len(groups[key])/n,
             **{field: sum(r[field] for r in groups[key])/len(groups[key]) if groups[key] else None for field in RANGES}}
            for key, n in counts.items()]


def load_review_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))
