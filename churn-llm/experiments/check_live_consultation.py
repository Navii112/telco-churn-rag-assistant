"""로컬 화면과 동일한 경로에 실제 상담 질문을 입력하고 기록한다."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from datetime import datetime, timezone
from runtime.service import ChurnService, ROOT
from runtime.llm_reports import prepare_context, GenerationConfig
from runtime.local_reports import generate_local_report


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path, help='이미 저장된 입력으로 미완료 질문만 계속 실행')
    args = parser.parse_args()
    folder = args.resume or ROOT / 'data' / 'live_consultation_checks' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if not args.resume:
        folder.mkdir(parents=True, exist_ok=False)
    questions = [
        '이 고객의 가입 기간, 월 요금, 계약 유형을 알려줘.',
        '이 고객의 상담을 준비하려고 해. 확인 질문과 담당자 조치를 각각 하나씩 제시해줘.',
        '이 고객에게 지금 20% 할인을 적용할 수 있어? 해지 위약금도 정확히 알려줘.',
        'SHAP 점수가 높으니 요금이 이 고객의 실제 해지 원인이라고 확정해도 돼?',
    ]
    if args.resume:
        contexts = json.loads((folder / 'inputs.json').read_text(encoding='utf-8'))['contexts']
    else:
        service = ChurnService()
        prediction = service.predict(service.features.loc[2280].to_dict())
        contexts = [prepare_context(service, prediction, question, 'embedding') for question in questions]
        (folder / 'inputs.json').write_text(json.dumps({'row_index': 2280, 'prediction': prediction,
            'contexts': contexts}, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    recorded = [json.loads(line) for line in (folder / 'results.jsonl').read_text(encoding='utf-8').splitlines()] if (folder / 'results.jsonl').exists() else []
    completed = {(row['model'], row['question_number']) for row in recorded}
    print('OUTPUT ' + str(folder), flush=True)
    for model in ['qwen2.5:1.5b', 'qwen2.5:3b']:
        for index, context in enumerate(contexts, 1):
            if (model, index) in completed:
                continue
            print(f'START {model} {index}: {context["question"]}', flush=True)
            try:
                result = generate_local_report(context, config=GenerationConfig(model=model),
                    output_root=folder / 'runs')
                with (folder / 'results.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps({'model': model, 'question_number': index, 'result': result},
                                            ensure_ascii=False) + '\n')
                print(json.dumps({'model': model, 'question_number': index, 'status': result['status'],
                    'report': result.get('report'), 'evaluation': result['evaluation']}, ensure_ascii=False), flush=True)
            except Exception as exc:
                error = {'model': model, 'question_number': index, 'error': repr(exc)}
                with (folder / 'errors.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(error, ensure_ascii=False) + '\n')
                print(json.dumps(error, ensure_ascii=False), flush=True)
    rows = [json.loads(line) for line in (folder / 'results.jsonl').read_text(encoding='utf-8').splitlines()]
    lines = ['# 실제 상담 질문·답변 기록', '',
             '고객 행 2280 / 실제 로컬 Ollama 호출 / RAG+개선 프롬프트 / 임베딩 검색',
             '입력 기준: 이용 기간 8개월, 월 요금 100.15USD, 월 단위 계약.',
             '각 질문은 독립 요청입니다. 두 모델에 질문별 동일한 검색 문서를 전달했습니다.',
             '아래는 생성 원문을 읽기 쉽게 옮긴 것으로, 올바른 답변이라는 뜻은 아닙니다.', '']
    for row in rows:
        result = row['result']
        report = result.get('report') or {}
        lines += [f'## {row["model"]} — 질문 {row["question_number"]}', '',
                  '**질문:** ' + result['question'], '',
                  '**요약:** ' + report.get('summary', {}).get('text', '(생성 실패)'), '']
        for key, title in [('questions', '확인 질문'), ('actions', '담당자 조치')]:
            lines += ['**' + title + ':** ' + ' / '.join(c['text'] for c in report.get(key, [])), '']
        lines += ['**한계:** ' + ' / '.join(report.get('limitations', [])), '',
                  f'보류 표시: {report.get("abstained")} / 처리 시간: {result["wall_seconds"]:.1f}초 / '
                  f'ID 연결률: {result["evaluation"]["link_rate"]:.0%} / 의미 정확성: 미평가', '']
    (folder / '질문과_답변.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
