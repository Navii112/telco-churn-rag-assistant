"""시연용 Markdown 지침 로드·조건 적용·출처가 있는 상담 준비 리포트."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
import re
from pathlib import Path

GUIDES = Path(__file__).resolve().parents[1] / 'guidelines'


def load_guidelines(folder=GUIDES):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    documents = []
    for spec in manifest['documents']:
        source = folder / spec['file']
        body = source.read_text(encoding='utf-8')
        for match in re.finditer(r'^## ([\w-]+) \| ([^\n]+)\n(.*?)(?=^## |\Z)', body, re.M | re.S):
            key, title, text = match.groups()
            questions = re.findall(r'^질문: (.+)$', text, re.M)
            actions = re.findall(r'^조치: (.+)$', text, re.M)
            documents.append({'id': 'guide:' + key, 'title': title.strip(), 'text': text.strip(),
                              'source': 'churn-llm/guidelines/' + spec['file'],
                              'kind': 'fictional_demo_only', 'version': manifest['version'],
                              'notice': manifest['notice'], 'rule': spec.get('rules', {}).get(key),
                              'questions': questions, 'actions': actions})
    if len({d['id'] for d in documents}) != len(documents):
        raise ValueError('상담 지침 ID 중복')
    return documents


def applicable(document, prediction):
    rule = document.get('rule')
    if not rule:
        return True
    value = prediction['features'].get(rule['feature'])
    if value is None:
        return False
    if 'values' in rule:
        return value in rule['values']
    return rule.get('min', float('-inf')) <= float(value) <= rule.get('max', float('inf'))


def select_report_evidence(service, prediction, question, method='embedding'):
    """LLM/실험 공통 근거 선택. 질문·조치를 템플릿으로 생성하지 않음."""
    if not prediction:
        raise ValueError('상담 준비 리포트는 먼저 ML 예측을 실행해야 합니다.')
    if not question.strip() or len(question) > 1000:
        raise ValueError('상담 준비 요청은 1~1000자로 입력하세요.')
    eligible = [d for d in service.documents if d.get('kind') == 'fictional_demo_only'
                and d.get('questions') and applicable(d, prediction)]
    features = prediction['features']
    actionable = {'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges', 'InternetService', 'TechSupport', 'PaymentMethod'}
    focused = any(word in question for word in ('청구', '요금', '결제', '계약', '인터넷', '기술', '가입', '혜택', '할인', '위약'))
    context = ' '.join(f"{service.labels.get(c['feature'],c['feature'])} {features[c['feature']]}"
                       for c in prediction['contributions'] if c['feature'] in actionable)
    query = question if focused else question + ' ' + context
    ranked = service.search_documents(query, method, top_k=3,
                                      allowed_ids={d['id'] for d in eligible})
    fixed = [next(d for d in service.documents if d['id'] == key)
             for key in ('guide:principles', 'guide:benefits')]
    documents = list({d['id']: d for d in fixed + ranked}.values())
    snapshot = service.retrieve('이 고객의 예측 판단 근거', prediction, top_k=0, method='tfidf')
    documents = snapshot + documents
    return documents, ranked


def prepare_report(service, prediction, question, method='embedding'):
    """이전 원문 템플릿 비교 기준. 주 서비스는 llm_reports.generate_report 사용."""
    documents, ranked = select_report_evidence(service, prediction, question, method)
    questions, actions = [], []
    for guide in ranked:
        for text in guide.get('questions', []):
            questions.append({'text': text, 'evidence_id': guide['id'], 'quote': text})
        for text in guide.get('actions', []):
            actions.append({'text': text, 'evidence_id': guide['id'], 'quote': text})
    if not questions:
        raise ValueError('조건에 맞는 상담 지침을 검색하지 못했습니다.')
    report = {
        'report_type': 'consultation_preparation', 'generation_mode': 'source_template',
        'notice': '시연용 가상 상담 지침 기반입니다. 실제 정책·혜택 안내나 상담 효과를 보장하지 않습니다.',
        'question': question, 'retrieval_method': method,
        'customer_summary': {'score': prediction['score'], 'threshold': prediction['threshold'],
                             'prediction': prediction['prediction'], 'model': prediction['model']},
        'model_findings': prediction['contributions'][:3],
        'questions': questions, 'actions': actions,
        'limitations': [*prediction['warnings'],
                        '실제 할인·위약금·혜택 조건은 미등록 상태입니다. 실제 약관 확인이 필요합니다.',
                        '고객의 실제 불만·해지 의사·상담 효과는 이 데이터로 확인할 수 없습니다.'],
        'references': documents, 'llm_explanation': None,
    }
    validate_report(report)
    return report


def validate_report(report):
    evidence = {d['id']: d for d in report['references']}
    for key in ('questions', 'actions'):
        for item in report[key]:
            doc = evidence.get(item['evidence_id'])
            if not doc or doc.get('kind') != 'fictional_demo_only' or item['quote'] not in doc['text']:
                raise ValueError('상담 리포트의 근거 검증 실패')
    return report


def report_markdown(report):
    summary = report['customer_summary']
    lines = ['# 상담 준비 리포트', '', '> ' + report['notice'], '',
             f"- 모델: {summary['model']}", f"- 이탈 예측 점수: {summary['score']:.2%}",
             f"- 판정: {'이탈 위험' if summary['prediction'] else '유지 예측'} (임계값 {summary['threshold']})",
             f"- 검색 방식: {report['retrieval_method']}", '', '## 고객별 모델 기여도', '']
    for item in report['model_findings']:
        lines.append(f"- {item['label']}: {item['value']} / {item['log_odds']:+.6f} log-odds ({item['direction']})")
    for title, key in (('상담 전 확인 질문', 'questions'), ('담당자 확인 조치', 'actions')):
        lines.extend(['', '## ' + title, ''])
        lines.extend(f"- {item['text']} [{item['evidence_id']}]" for item in report[key])
    if report.get('llm_explanation'):
        lines.extend(['', '## LLM 설명', ''])
        for statement in report['llm_explanation']['statements']:
            lines.append(f"- {statement['text']} [{statement['evidence_id']}]")
    lines.extend(['', '## 해석 한계', ''])
    lines.extend('- ' + text for text in report['limitations'])
    lines.extend(['', '## 근거 출처', ''])
    lines.extend(f"- [{d['id']}] {d['title']} — {d['source']}" for d in report['references'])
    return '\n'.join(lines)
