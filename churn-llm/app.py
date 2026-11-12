import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
"""LLM 직접 상담 리포트 생성과 2×2 비교 실험 화면."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import csv
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from runtime.service import ChurnService, ROOT
from runtime.llm_reports import CONDITIONS, GenerationConfig, prepare_context
from runtime.local_reports import generate_local_report, local_report_markdown
from runtime.quality_v2 import VERSION, claims_of
from runtime.question_scope import OutOfScopeQuestion, validate_question_scope
from experiments.run_experiments import run_comparison
from experiments.review_results import score_reviews, load_review_csv, RANGES

st.set_page_config(page_title='고객 상담 LLM 실험실', page_icon='💬', layout='wide')

@st.cache_resource
def load_service():
    return ChurnService()

service = load_service()
st.title('고객 상담 LLM 실험실')
st.caption('LLM 리포트 직접 생성 · RAG 적용 전후 · 프롬프트 개선 전후 비교')
st.info('상담 지침은 시연용 가상 문서입니다. ML은 고객 배경 참고 자료이며 이번 프로젝트의 평가 대상은 LLM 출력입니다.')
with st.sidebar:
    model = st.selectbox('생성 모델', ['qwen2.5:1.5b', 'qwen2.5:3b', 'gemma2:latest'])
    search_label = st.selectbox('검색 방식', ['임베딩 (EmbeddingGemma)', 'TF-IDF (비교 기준)'])
    method = 'embedding' if search_label.startswith('임베딩') else 'tfidf'
    st.header('분석할 고객 입력')
    subset = st.radio('예시 데이터 범위', ['테스트 고객', '학습 고객'])
    split = 'Test' if subset == '테스트 고객' else 'Train'
    choices = service.splits[service.splits == split].index.tolist()
    sample = st.selectbox('예시 행 번호 (고객ID 아님)', choices)
    st.caption('기존 데이터 행을 입력 예시로 사용합니다. 실제 이탈 정답은 LLM에 전달하지 않습니다.')
    seed = service.features.loc[sample]
    row = {}
    with st.form(f'customer_{sample}'):
        for column in service.bundle['input_columns']:
            label = f'{service.labels.get(column, column)} · {column}'
            if column in ('tenure', 'MonthlyCharges', 'TotalCharges'):
                default = seed[column]
                default = 0.0 if pd.isna(default) else float(default)
                row[column] = st.number_input(label, min_value=0.0, value=default,
                                              step=1.0 if column == 'tenure' else 0.1)
            else:
                values = sorted(service.features[column].astype(str).unique())
                row[column] = st.selectbox(label, values, index=values.index(str(seed[column])))
        predict_clicked = st.form_submit_button('ML 예측 및 설명 계산', type='primary', width='stretch')
    st.caption('예측 버튼을 누른 시점의 입력을 고정합니다. 값을 바꾼 뒤에는 다시 예측하세요.')

if predict_clicked:
    try:
        prediction = service.predict(row)
        st.session_state.prediction = prediction
        st.session_state.pop('llm_result', None)
        st.session_state.pop('report_error', None)
        st.session_state.pop('scope_notice', None)
    except (ValueError, TypeError) as exc:
        st.session_state.pop('prediction', None)
        st.session_state.pop('llm_result', None)
        st.error(str(exc))
        st.stop()

prediction = st.session_state.get('prediction')
config = GenerationConfig(model=model)
with st.expander('참고 고객 정보 (ML은 재학습하지 않음)', expanded=prediction is None):
    if prediction:
        a,b,c = st.columns(3)
        a.metric('ML 참고 점수', f"{prediction['score']:.2%}")
        b.metric('참고 판정', '이탈 위험' if prediction['prediction'] else '유지 예측')
        c.metric('임계값', prediction['threshold'])
        table = pd.DataFrame(prediction['contributions']).head(5)
        table['value'] = table['value'].astype(str)
        st.dataframe(table, hide_index=True)
        st.json(prediction['features'], expanded=False)
    else:
        st.write('왼쪽에서 예시 입력을 선택하고 ML 예측을 계산하세요. 비교 실험은 미리 정한 사례로 별도 실행할 수 있습니다.')
    st.caption('점수는 보정된 미래 확률이 아니며 SHAP은 실제 해지 원인을 증명하지 않습니다.')

report_tab, experiment_tab, model_tab, guide_tab = st.tabs(['LLM 상담 리포트', '4조건 비교·평가', '생성 모델 비교', '지식 문서'])
with report_tab:
    st.write('요약·확인 질문·담당자 조치·한계는 모두 LLM이 생성합니다. 원문 템플릿으로 대체하지 않습니다.')
    st.caption('최신 다중 근거·내용 검사 적용. 질문은 매번 독립적으로 처리하며 이전 대화를 기억하지 않습니다.')
    with st.form('llm_report'):
        condition = st.selectbox('생성 조건', list(CONDITIONS), index=3, format_func=lambda c: CONDITIONS[c]['label'])
        question = st.text_area('상담 요청', value='계약 조건을 확인하기 위한 상담 리포트를 작성해줘. 질문과 조치를 각각 한 개씩 제시해줘.', max_chars=1000)
        make_report = st.form_submit_button('LLM 상담 리포트 생성', type='primary', disabled=prediction is None)
    if make_report:
        st.session_state.pop('llm_result', None)
        st.session_state.pop('report_error', None)
        st.session_state.pop('scope_notice', None)
        try:
            validate_question_scope(question)
            with st.spinner('동일 고객 스냅샷과 관련 문서를 준비합니다.'):
                context = prepare_context(service, prediction, question, method)
            with st.spinner('로컬 LLM이 리포트를 직접 작성합니다. 모델에 따라 수 분이 걸릴 수 있습니다.'):
                st.session_state.llm_result = generate_local_report(context, condition, config=config)
        except OutOfScopeQuestion as exc:
            st.session_state.scope_notice = str(exc)
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            st.session_state.report_error = str(exc)
    if st.session_state.get('scope_notice'):
        st.info(st.session_state.scope_notice)
    if st.session_state.get('report_error'):
        st.error(st.session_state.report_error)
    result = st.session_state.get('llm_result')
    if result and result.get('version') != VERSION:
        st.info('이전 실행 결과입니다. 최신 방식으로 리포트를 다시 생성하세요.')
        result = None
    if result:
        st.caption(f"실행 조건: {result['condition']} / 모델: {result['config']['model']} / {result['wall_seconds']:.1f}초")
        st.write('상태:', result['status'])
        evaluation=result['evaluation']
        st.warning('검토용 LLM 초안입니다. 근거 연결 통과나 경고 없음은 내용 정확성을 보장하지 않습니다.')
        a,b,c=st.columns(3)
        a.metric('근거 ID 연결률',f"{evaluation['link_rate']:.0%}")
        b.metric('내용 경고 항목',evaluation['flagged_claims'])
        c.metric('의미 정확성','미평가')
        st.caption('근거 ID 연결률은 ID의 존재 여부만 검사합니다. 연결된 문서가 답변 내용을 뒷받침하는지는 별도 검토가 필요합니다.')
        if result.get('structure_valid'):
            generated = result['report']
            st.subheader('LLM 고객 상황 요약')
            st.text(generated['summary']['text'])
            for title,key in [('LLM 확인 질문','questions'),('LLM 담당자 조치','actions')]:
                st.subheader(title)
                for item in generated[key]:
                    st.text(item['text'])
            for text in generated['limitations']:
                st.info(text)
            st.caption('모델의 답변 보류 표시: '+str(generated['abstained']))
        else:
            st.error('리포트 생성 또는 구조 검사가 실패했습니다. 아래에서 원문을 확인하세요.')
        for location,claim in claims_of(result.get('report')):
            with st.expander('근거 보기 · '+location):
                st.text(claim['text'])
                link=next((x for x in evaluation['links'] if x['location']==location),None)
                if link:
                    for citation in link['citations']:
                        st.caption('['+citation['id']+'] '+citation['source'])
                        st.text(citation['text'])
                    if not link['linked']:
                        st.warning('연결되지 않은 근거: '+(', '.join(link['missing_ids']) or '근거 ID 없음'))
        for check in evaluation['content_checks']:
            if check['issues']:
                st.warning(check['text']+' — '+' / '.join(check['issues']))
        with st.expander('생성 원문·프롬프트·자동 검사 (검증 전 출력 보존)'):
            st.text(result['raw_text'])
            st.json(evaluation)
            st.json(result['messages'])
            if result['error']:
                st.error(result['error'])
        with st.expander('이 조건에서 LLM에 제공한 근거'):
            for doc in result['references']:
                st.text('[' + doc['id'] + '] ' + doc['title'])
                st.text(doc['text'])
                st.caption(doc['source'])
        st.download_button('LLM 결과 JSON 다운로드', json.dumps(result,ensure_ascii=False,indent=2), file_name='llm_report.json',mime='application/json')
        st.download_button('LLM 리포트 Markdown 다운로드',local_report_markdown(result),file_name='llm_report.md',mime='text/markdown')
        st.caption('실행 기록: '+result['record_folder'])

with experiment_tab:
    st.info('이 탭은 기존 v2-factorial-2 방식의 비교 실험입니다. 최신 로컬 리포트 결과와 같은 버전으로 합산하지 않습니다.')
    st.write('RAG 없음/있음 × 기본/개선 프롬프트. 모델·고객·질문·출력 스키마·생성 설정을 동일하게 유지합니다.')
    cases = [json.loads(s) for s in (ROOT/'data/report_cases.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    with st.form('experiment'):
        chosen = st.multiselect('평가 사례', [c['id'] for c in cases], default=[cases[0]['id']])
        repeats = st.number_input('사례별 반복 횟수', min_value=1,max_value=3,value=1,step=1)
        st.caption('1사례 × 1회 = LLM 4회 호출. 실행 실패도 저장·집계합니다.')
        run = st.form_submit_button('4조건 실제 비교 실행', type='primary')
    if run:
        selected = [c for c in cases if c['id'] in chosen]
        if not selected:
            st.error('평가 사례를 선택하세요.')
        else:
            progress = st.progress(0)
            message = st.empty()
            def update(record, n, total):
                progress.progress(n/total)
                message.text(f"{n}/{total}: {record['condition']} → {record['status']}")
            try:
                folder, records = run_comparison(service, selected, method=method, config=config,repeats=int(repeats),on_result=update)
                st.session_state.last_run = folder.name
            except (RuntimeError,ValueError,OSError,KeyError) as exc:
                st.error(str(exc))
    runs = sorted([p for p in (ROOT/'data/experiments').glob('*') if (p/'summary.json').exists()],reverse=True)
    if runs:
        names = [p.name for p in runs]
        default = names.index(st.session_state.last_run) if st.session_state.get('last_run') in names else 0
        name = st.selectbox('저장된 실험', names,index=default)
        folder = ROOT/'data/experiments'/name
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        st.dataframe(pd.DataFrame(summary).drop(columns=['status_counts'], errors='ignore'),hide_index=True)
        st.caption('자동 지표는 의미적 정확성·환각률이 아닙니다. 실패를 포함한 전체 시도가 분모입니다. 의미 품질은 아래 수동 평가에서 기록합니다.')
        records = [json.loads(s) for s in (folder/'results.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        for record in records:
            with st.expander(record['trial_id'] + ' · ' + record['status']):
                st.text(record['question'])
                st.text(record['raw_text'])
                st.json(record['automatic'])
        st.markdown('#### 내용 수동 평가')
        st.caption('사실 정확성·요청 대응: 0=틀림/실패, 1=부분, 2=적절. 정책 발명·원인 단정: 0=없음, 1=있음. 보류 적절성: 0=부적절, 1=적절. 평가자와 모든 점수 입력 시 집계합니다.')
        review = load_review_csv(folder/'manual_review.csv')
        edited = st.data_editor(pd.DataFrame(review),hide_index=True,key='review_'+name,
            disabled=[c for c in review[0] if c not in {*RANGES,'reviewer','notes'}]) if review else pd.DataFrame()
        if st.button('수동 평가 저장·집계',disabled=not review):
            try:
                edited_rows = edited.fillna('').to_dict('records')
                manual_summary = score_reviews(edited_rows)
                with (folder/'manual_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(review[0]));writer.writeheader();writer.writerows(edited_rows)
                (folder/'manual_summary.json').write_text(json.dumps(manual_summary,ensure_ascii=False,indent=2),encoding='utf-8')
                st.success('수동 평가를 저장했습니다.')
                st.dataframe(pd.DataFrame(manual_summary),hide_index=True)
            except (ValueError,TypeError) as exc:
                st.error(str(exc))
        st.download_button('수동 평가 CSV 다운로드',(folder/'manual_review.csv').read_bytes(),file_name='manual_review.csv',mime='text/csv')
        with st.expander('실험 설계·실행 설정'):
            st.json(json.loads((folder/'manifest.json').read_text(encoding='utf-8')))
    else:
        st.info('저장된 실제 실험이 없습니다. 비교 실행 또는 run_experiments.py를 실행하세요.')

with model_tab:
    st.caption('아래는 보존된 초기 모델 비교입니다. 최신 개선 전후 비교는 churn_llm_multi_evidence.ipynb에서 확인하세요.')
    comparison_path = ROOT/'data/model_comparison/comparison.json'
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
        st.dataframe(pd.DataFrame(comparison['summary']), hide_index=True)
        for note in comparison['notes']:
            st.caption(note)
        st.write('모델 외 통제 조건 검사:', comparison['controls_verified'])
        for pair in comparison['pairs']:
            with st.expander(pair['case_id']+' / '+pair['condition']):
                left,right = st.columns(2)
                with left:
                    st.write(comparison['summary'][0]['model'],pair['baseline_status'])
                    st.text(pair['baseline_raw_text'])
                with right:
                    st.write(comparison['summary'][1]['model'],pair['candidate_status'])
                    st.text(pair['candidate_raw_text'])
        st.download_button('모델 비교 JSON 다운로드', comparison_path.read_bytes(),file_name='model_comparison.json',mime='application/json')
    else:
        st.info('모델 비교 실행 결과가 없습니다. compare_models.py로 동일 조건 비교를 실행하세요.')

with guide_tab:
    for doc in service.documents:
        if doc.get('kind')=='fictional_demo_only':
            with st.expander(doc['id']+' · '+doc['title']):
                st.text(doc['text']);st.caption(doc['source'])
    st.caption('이전 템플릿 리포트 함수는 회귀 비교용으로만 보존하며 주 화면과 비교 실험에서는 사용하지 않습니다.')
