# 고객 상담 LLM: 직접 리포트 생성과 RAG·프롬프트 비교

주요 연구 대상은 LLM 출력입니다. 기존 ML은 고객 관측 정보·예측 점수·SHAP을 제공하는 참고 엔진이며 재학습하지 않습니다.
상담 리포트의 요약·확인 질문·담당자 조치·한계는 모두 LLM이 직접 생성합니다.
이전 원문 템플릿은 주 화면·실험에서 사용하지 않습니다.

## 실행

LLM 워크스페이스 루트에서:

```powershell
ollama pull qwen2.5:1.5b
ollama pull embeddinggemma
& .\ML\.venv\Scripts\python.exe -m streamlit run .\churn-llm\app.py --server.port 8502 --server.address 127.0.0.1 --browser.gatherUsageStats false
```

브라우저: http://127.0.0.1:8502
현재 PC에는 Qwen 1.5B, Qwen 3B, Gemma 2, EmbeddingGemma가 설치돼 있습니다.
Qwen은 반복 실험의 자원 부담을 낮추려고 추가한 약 986MB 로컬 모델이며, 더 정확하다는 전제로 선정한 것은 아닙니다.
Python 환경은 기존 `ML/.venv`를 사용합니다. 추가 유료 API가 필요하지 않고 CPU·RAM·전력·저장 공간을 사용합니다.

## 화면

1. 왼쪽에서 생성 모델과 검색 방식을 선택합니다.
2. 예시 고객 입력으로 ML 참고 점수를 계산합니다. ML 결과는 접힌 참고 영역에 표시합니다.
3. **LLM 상담 리포트** 탭에서 조건과 질문을 선택하고 생성합니다.
4. 구조가 유효하면 LLM 초안을 표시하며 인용 검증 실패 시 미검증 상태를 명시합니다.
5. 원문·프롬프트·제공 근거·검증 결과와 JSON/Markdown 다운로드를 제공합니다.
6. **4조건 비교·평가** 탭에서 동일 사례를 4조건으로 실제 실행합니다.
7. 자동 지표와 조건별 생성 원문을 비교하고 수동 평가 점수를 입력해 저장합니다.

모델 실패·잘린 출력·검증 실패를 템플릿으로 대체하거나 성공처럼 표시하지 않습니다.
원문은 검증 전에 저장하며 정상/실패 여부와 관계없이 실험 분모에 포함합니다.
가상 지침은 실제 회사 정책이 아니고, 예측·SHAP은 실제 해지 원인이나 상담 효과를 입증하지 않습니다.

## 2×2 통제 실험

| 조건 | RAG | 프롬프트 |
|---|---|---|
| no_rag_basic | 없음 | 기본 |
| rag_basic | 있음 | 기본 |
| no_rag_improved | 없음 | 개선 |
| rag_improved | 있음 | 개선 |

고객·질문·모델·생성 seed/temperature/문맥/출력 길이·JSON schema는 한 실험 안에서 동일합니다.
RAG 후보는 사례별 한 번 동결하고 두 RAG 조건에 그대로 사용합니다.
no-RAG에는 고객 관측 입력·점수·SHAP만 주며 지침·이전 템플릿 답안·범위 문서를 넣지 않습니다.
모든 조건에 JSON 출력 형식 제약을 공통 적용합니다. 자유 텍스트 생성과 구조화 생성의 비교는 아닙니다.
실행 순서는 seed로 섞습니다. LLM 가중치를 학습하거나 토크나이저를 변경하지 않습니다.

- RAG 효과: `rag_basic - no_rag_basic`, `rag_improved - no_rag_improved`
- 프롬프트 효과: `no_rag_improved - no_rag_basic`, `rag_improved - rag_basic`
- A와 D만 비교하면 두 요소가 동시에 바뀌므로 원인을 구분할 수 없습니다.

## 실제 비교 실행

```powershell
& .\ML\.venv\Scripts\python.exe churn-llm/experiments/run_experiments.py --limit 1
& .\ML\.venv\Scripts\python.exe churn-llm/experiments/run_experiments.py --limit 4 --repeats 2 --model qwen2.5:1.5b
```

기본은 1사례×4조건×1회입니다. 4사례×2회는 32번 모델을 호출합니다.
`--method tfidf`로 검색 기준을 바꿀 수 있고 `--model gemma2:latest`도 지원합니다.
모델을 바꾼 결과는 별도 실험으로 비교해야 합니다. 같은 통제 실험 안에서 모델을 바꾸지 않습니다.
예시 사례는 계약·요금/결제·미등록 혜택·인과관계 단정 질문입니다.
`data/report_cases.jsonl`의 expected_abstain/review_focus는 평가자 참고이며 생성 입력에는 전달하지 않습니다.

`data/experiments/<run_id>/` 파일:

- `manifest.json`: 조건·사례·공통 생성 설정·실행 상태
- `results.jsonl`: 시도별 생성 원문, 파싱 결과, 프롬프트, 제공 근거, 모델 digest, 토큰·시간·실패
- `summary.json`: 자동 지표 집계 (전체 시도가 분모)
- `manual_review.csv`: 내용 평가 입력 양식. 초기 점수는 비어 있음
- `manual_summary.json`: 화면에서 수동 평가 저장 시 생성

자동 재시도·답변 정제·템플릿 대체는 하지 않습니다. 같은 run_id를 덮어쓰지 않습니다.
오류가 발생해도 이미 실행한 결과를 보존합니다.
초기 실제 실험에는 검증 실패와 출력 길이 초과가 있었으며 원본을 삭제하지 않았습니다.
짧은 출력 길이 제약을 공통으로 보완한 실행도 별도 run_id와 prompt_version으로 기록합니다.

## 평가를 구분하기

자동: JSON 유효성, 구조, 질문·조치 존재, 정확한 ID/원문 인용 일치, 토큰 수, 시간, 실행 실패.
이 지표는 의미적 정확성이나 환각률이 아닙니다. 근거를 인용해도 잘못 해석할 수 있습니다.
no-RAG는 상담 문서를 받지 않으므로 지침 인용량만으로 종합 우열을 정하지 마세요.
검증에 통과한 답변만 분모에 넣으면 선택 편향이 생기므로 실패도 전체 시도에 포함합니다.

수동 평가 rubric:

| 항목 | 값 |
|---|---|
| 사실 정확성 | 0=틀림/실패, 1=부분적으로 맞음, 2=적절 |
| 정책 발명 | 0=없음, 1=있음 |
| 실제 원인·효과 단정 | 0=없음, 1=있음 |
| 요청 대응 | 0=실패, 1=부분, 2=적절 |
| 답변 보류의 적절성 | 0=부적절, 1=적절 |

모든 점수와 reviewer가 입력된 행만 내용 평균을 계산하고 평가 완료율을 같이 표시합니다.
미평가를 0점이나 정답으로 간주하지 않습니다. 평가자는 같은 지침을 읽고, 가능하면 조건명을 가려 평가하세요.
모델 로딩/캐시/순서가 실행 시간에 영향을 주므로 `load_duration`도 확인하세요.
작은 자체 개발 세트에서의 실행 결과로 일반적 우수성이나 실제 이탈 감소를 주장하지 마세요.

## 생성 모델 추가 비교

기존 4조건 실험의 생성 모델은 `qwen2.5:1.5b` 하나였습니다.
`embeddinggemma`는 검색용 임베딩 모델이고, `gemma2`는 초기 연결 점검에 사용한 모델입니다.
추가 비교 모델은 같은 계열의 `qwen2.5:3b`입니다. 한국어와 JSON 출력 용도로 사용하며,
큰 모델이 더 좋다는 결론을 미리 정하지 않고 결과를 기록합니다.
1.5B와 3B는 학습·가중치도 다르므로 파라미터 수의 순수 인과 효과로 해석하지 않습니다.

```powershell
ollama pull qwen2.5:3b
& .\\ML\\.venv\\Scripts\\python.exe churn-llm/experiments/compare_models.py --baseline churn-llm/data/experiments/20261109T062056Z-c7e977 --candidate-model qwen2.5:3b
```

`experiments/compare_models.py`는 기준 실행의 사례·반복·설정을 가져와 모델만 바꿉니다.
완료 후 고객 hash·문서 hash·프롬프트·schema·생성 설정과 조건 집합이 같은지 검사합니다.
하나라도 다르면 모델 비교 표를 만들지 않습니다. 일부 성공 조건만 선택하는 것도 거절합니다.
이미 실행한 추가 모델 결과는 `--candidate-run <실험폴더>`로 지정해 비교 보고서만 다시 만들 수 있습니다.

결과: `data/model_comparison/comparison.json`, `comparison.md`.
화면의 **생성 모델 비교** 탭과 노트북의 **추가 생성 모델 비교** 절에서도 확인합니다.
단일 사례×4조건의 작은 예비 비교입니다. 인용 일치와 내용 정확성은 구분하고,
최종 모델 선정 전 나머지 사례·반복·내용 평가를 추가하세요.

## 노트북

### 품질 개선 전후 비교 (별도 버전)

#### 2차 개선: 다중 근거와 내용 검사

새 노트북은 `notebooks/archive/churn_llm_multi_evidence.ipynb`입니다. 최초·1차·2차 답변을 비교합니다.
`runtime/quality_v2.py`는 다중 `evidence_ids` 배열, 이전 쉼표 문자열의 분리·중복 제거,
완전한 문장 제공, 이용 기간과 계약 기간 구분 지시, 일부 고객 숫자·통화·근거 누락 검사를 제공합니다.
숫자가 일치해도 의미적 뒷받침은 사람이 검토해야 합니다. 규칙에 걸리지 않는다고 정답은 아닙니다.

```powershell
# 두 모델 × 네 조건의 실제 새 생성 (기존 원본 보존)
& .\ML\.venv\Scripts\python.exe .\churn-llm\runtime\quality_v2.py
# 위 명령이 출력한 실험 폴더를 지정해 세 버전에 동일한 검사 적용
& .\ML\.venv\Scripts\python.exe .\churn-llm\runtime\quality_v2.py --evaluate-only <실험폴더>
# 정보 없는 질문의 별도 실제 실행
& .\ML\.venv\Scripts\python.exe .\churn-llm\experiments\run_quality_boundary.py
& .\ML\.venv\Scripts\python.exe .\churn-llm\tools\create_quality_v2_notebook.py
& .\ML\.venv\Scripts\python.exe .\churn-llm\tools\execute_notebook.py churn_llm_multi_evidence.ipynb
```

기록은 `data/quality_v2_experiments/`의 새 실행 폴더에 저장합니다.
`triples.jsonl`은 생성 당시 기록, `evaluated_triples_rules-v2.2.jsonl`은 세 버전에 동일한 최신 검사 버전을 적용한 별도 기록입니다.
`evaluation_manifest_rules-v2.2.json`은 검사 코드·입력 hash를 보존합니다. 이전 재평가 파일도 보존하며 검사 규칙 보완은 생성 결과와 분리합니다.
최신 검사는 가입 여부 오류와 답변 보류 표시의 불일치 가능성도 표시합니다.
`claim_review.csv`는 주장별 사람 평가 양식이며 점수는 실제 평가자가 입력합니다.
노트북 6절의 `RUN_CUSTOM=True` 또는 `runtime.quality_v2.ask_question`으로 새 질문을 실제 생성할 수 있습니다.
로컬 Streamlit의 **LLM 상담 리포트** 탭도 `runtime/local_reports.py`를 통해 같은 최신 생성·검사를 사용합니다.
새 화면 실행은 `data/local_reports/<run_id>/result.json`에 원문·입력·다중 근거·검사를 보존합니다.
근거 연결률과 내용 경고를 분리해서 표시하고, 의미 정확성은 미평가로 남깁니다.
기존 **4조건 비교·평가**와 초기 모델 비교는 이전 방식임을 화면에 명시하며 최신 결과와 합산하지 않습니다.
외부 배포 설정이나 호스팅은 추가하지 않았습니다.
새 질문도 독립 실행이며 대화 기억 기능이나 모든 질문의 정확성을 보장하지 않습니다.

`notebooks/archive/churn_llm_quality_improved.ipynb`는 기존 노트북과 별도로 만든 개선 실험입니다.
기존 두 모델의 8개 실행을 기준으로 고객·질문·원본 검색 근거·생성 설정을 유지하고,
한국어 필드와 단위 명시, 짧은 근거 ID, LLM의 ID 선택과 코드의 원문 연결 분리를 적용합니다.
유효하지 않은 ID는 보정하지 않습니다. 생성 실패와 잘린 출력도 저장합니다.
기간·금액 검사는 일부 표현만 다루며 경고 없음은 내용 정확성을 보장하지 않습니다.

```powershell
# 실제 8회 생성. 기존 결과를 수정하지 않고 새 실험 폴더에 저장합니다.
& .\ML\.venv\Scripts\python.exe .\churn-llm\experiments\improve_quality.py
# 저장된 개선 실험을 읽는 노트북 생성·실행
& .\ML\.venv\Scripts\python.exe .\churn-llm\tools\create_quality_notebook.py
& .\ML\.venv\Scripts\python.exe .\churn-llm\tools\execute_notebook.py churn_llm_quality_improved.ipynb
```

결과는 `data/quality_experiments/<run_id>/`에 별도로 저장됩니다.
`pairs.jsonl`은 기존/개선 원문·입력·근거·검사를 보존하고 `manual_review.csv`는 빈 내용 평가 양식입니다.
새 노트북의 3절은 수치, 4절은 질문과 기존/개선 답변을 나란히 보여줍니다.
입력·프롬프트·schema·인용 처리를 함께 바꾼 **시스템 개선 비교**입니다.
인용 연결 지표의 상승을 LLM의 정확성 상승이나 순수 RAG 효과로 해석하지 않습니다.

`notebooks/archive/churn_llm_analysis.ipynb`를 ML 가상환경 커널로 엽니다.
기존 v1 노트북은 `notebooks/archive/churn_llm_analysis_v1.ipynb`에 보존했습니다.
새 노트북은 실험 설계 → 입력 격리 → 프롬프트 → 직접 생성 → 네 조건 원문 비교 → 자동/수동 평가 순서입니다.
기본은 저장된 실제 결과를 읽습니다. `RUN_GENERATION=True` / `RUN_COMPARISON=True`로 새 모델 호출을 실행합니다.
`tools/create_notebook.py`는 노트북을 재생성하며 셀 출력은 초기화합니다.

## 파일 역할

- `runtime/llm_reports.py`: LLM 직접 생성·네 조건·공통 schema·검증 전 원문·자동 검사
- `experiments/run_experiments.py`: 실제 4조건 실행·실패 보존·집계·수동 평가 양식
- `experiments/review_results.py`: 수동 평가 점수 검사·완료율·집계
- `experiments/compare_models.py`: 모델 외 통제 조건 검사와 두 생성 모델 비교 보고서
- `app.py`: LLM 리포트와 비교 실험 UI
- `runtime/service.py`: ML 참고 입력과 검색·기존 보조 질의응답
- `runtime/embedding_search.py`: 임베딩 캐시와 코사인 검색
- `runtime/guidance.py`: 적용 조건·근거 선택 및 이전 템플릿 비교 기준
- `guidelines/`: 가상 상담 지침 6개, 섹션 청크 10개

`data/consultation_example.*`와 이전 `data/notebook_outputs/consultation_report.*`는 v1 템플릿 예시입니다.
현재 LLM 직접 생성 결과는 `data/experiments/`와 새 노트북의 `llm_report.md`를 확인하세요.

## 20질문·80응답 평가

`experiments/benchmark20.py`는 고객 5명에 대해 사실 확인·상담 준비·정보 없음·잘못된 단정 유도 질문을 5개씩 준비합니다.
생성 전에 질문·평가 기준과 검색 문맥을 저장하며 gold 기준은 모델 입력에 전달하지 않습니다.
Qwen 1.5B와 3B, 최초 `llm_reports.generate_report`와 최신 `quality_v2.generate_v2`를 비교합니다.
두 버전 모두 RAG 있음·개선 지시 조건을 사용합니다. 순수 RAG·프롬프트 효과를 분리하는 실험은 아닙니다.

```powershell
& .\ML\.venv\Scripts\python.exe .\churn-llm\experiments\benchmark20.py
# 중단된 실행은 동일 코드·설정으로 이어갑니다. 저장된 실패도 재시도하지 않습니다.
& .\ML\.venv\Scripts\python.exe .\churn-llm\experiments\benchmark20.py --resume <실험폴더>
& .\ML\.venv\Scripts\python.exe .\churn-llm\experiments\benchmark20_scoring.py <실험폴더>
& .\ML\.venv\Scripts\python.exe .\churn-llm\tools\create_benchmark20_notebook.py
& .\ML\.venv\Scripts\python.exe .\churn-llm\tools\execute_notebook.py churn_llm_benchmark20.ipynb
```

최종 결과는 `notebooks/churn_llm_benchmark20.ipynb`에서 봅니다. 실행 결과·설정·질문을 내장하므로
분석 재실행에 Ollama나 다른 프로젝트 파일이 필요하지 않습니다. 단독 환경 의존성은 `requirements-analysis.txt`입니다.
기본 실행은 저장된 실제 결과만 분석합니다. 8절의 `RUN_DEMO=True`를 선택하면 내장한 질문 한 개를
동결된 근거·설정으로 로컬 Ollama에 다시 보낼 수 있습니다. 별도 `.py` 파일은 필요 없지만 Ollama와 해당 Qwen이 필요합니다.
이 시연은 새 문서 검색을 하지 않으며, 새 응답을 기존 80회 결과와 합산하지 않습니다.

자동 지표는 필수 고객 값 표현, 보류 표시, 확인 불가 표현, ID 연결, 일부 규칙 경고와 시간입니다.
표에는 이용 기간·계약 기간 혼동과 약/쉼표 통화 표현의 인식 오탐을 보완한 reference-v1.1을 동일 적용합니다.
생성 시점의 `results.jsonl`과 새 `evaluated_results_v1_1.jsonl`은 별도로 보존합니다.
`evaluation_manifest_v1_1.json`에 사후 검사 보완과 코드 hash를 남깁니다. 검사 보완을 모델 개선으로 주장하지 않습니다.
사실 정확성·근거 적절성·질문 대응 등 의미 평가는 `human_review.csv`에 실제 평가자가 채점합니다.
`review_mapping.json`에 모델·버전 대조표를 분리했고 미평가는 0점으로 처리하지 않습니다.
`benchmark_review.py <실험폴더>`는 모델 이름 대신 평가 ID로 질문·답변·근거·정답 기준을 묶은
`blind_review_packet.html`과 JSON을 만듭니다. 출력 형식으로 버전을 짐작할 수 있어 완전한 맹검은 아닙니다.
사람 채점 후 노트북 생성·실행 명령을 다시 실행하면 해당 점수도 내장됩니다.
이 세트는 개발 평가이며 별도의 미사용 테스트·반복 실험을 대신하지 않습니다.

## 테스트 실행

```powershell
& .\ML\.venv\Scripts\python.exe -m unittest discover -s churn-llm/tests -t churn-llm -p 'test_*.py' -v
```

단위 테스트의 가짜 모델 응답은 통제 로직 검증용이며 임시 폴더에만 저장합니다.
실제 모델 실험과 섞지 않습니다.

구현 참고: https://docs.ollama.com/api/chat
모델: https://ollama.com/library/qwen2.5:1.5b
추가 모델: https://ollama.com/library/qwen2.5:3b
