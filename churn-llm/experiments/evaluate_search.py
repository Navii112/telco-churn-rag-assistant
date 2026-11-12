"""작은 개발용 질문 세트의 순수 검색 비교. 고정 근거 추가/고객 조건 필터 이전 평가."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import argparse
import json
from pathlib import Path
import time
from runtime.service import ChurnService, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, default=ROOT / 'data/search_questions.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/search_evaluation.json')
    args = parser.parse_args()
    service = ChurnService()
    questions = [json.loads(s) for s in args.questions.read_text(encoding='utf-8').splitlines() if s.strip()]
    result = {'dataset': str(args.questions), 'note': '작성한 지침에 대한 개발용 6문항 점검. 일반화 성능이나 LLM 답변 품질 평가가 아님.',
              'corpus_size': len(service.documents), 'methods': {}}
    for method in ('tfidf', 'embedding'):
        rows = []
        for q in questions:
            start = time.perf_counter()
            docs = service.search_documents(q['question'], method, top_k=3)
            ids = [d['id'] for d in docs]
            expected = set(q['expected_ids'])
            rows.append({**q, 'retrieved_ids': ids, 'hit_at_3': bool(expected.intersection(ids)),
                         'recall_at_3': len(expected.intersection(ids))/len(expected),
                         'reciprocal_rank_at_3': next((1/(i+1) for i, key in enumerate(ids) if key in expected), 0),
                         'seconds': time.perf_counter()-start})
        result['methods'][method] = {'hit_at_3': sum(r['hit_at_3'] for r in rows)/len(rows),
                                     'mrr_at_3': sum(r['reciprocal_rank_at_3'] for r in rows)/len(rows), 'rows': rows}
        print(method, result['methods'][method]['hit_at_3'], flush=True)
    result['embedding'] = service.embedding_search.metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
