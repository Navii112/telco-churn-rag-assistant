"""Ollama EmbeddingGemma 로컬 의미 검색. corpus/model digest별 캐시."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import hashlib
import json
from pathlib import Path
import threading
from urllib.request import Request, ProxyHandler, build_opener
from urllib.error import URLError, HTTPError
import numpy as np


def local_request(route, payload=None, timeout=120):
    request = Request('http://127.0.0.1:11434' + route,
                      data=json.dumps(payload).encode('utf-8') if payload is not None else None,
                      headers={'Content-Type': 'application/json'})
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError('로컬 Ollama 요청 실패. 실행 상태와 설치 모델을 확인하세요. '
                           '임베딩은 ollama pull embeddinggemma 로 설치합니다.') from exc


def unit_vectors(values):
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError('임베딩 차원/값 오류')
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError('0 벡터는 검색에 사용할 수 없습니다.')
    return matrix / norms


class EmbeddingSearch:
    MODEL = 'embeddinggemma:latest'

    def __init__(self, documents, cache_dir, request=local_request):
        self.documents, self.cache_dir, self.request = documents, Path(cache_dir), request
        self.vectors = None
        self.lock = threading.RLock()
        self.metadata = None

    def embed(self, texts):
        result = self.request('/api/embed', {'model': self.MODEL, 'input': texts,
                                           'truncate': False, 'keep_alive': 0,
                                           'options': {'num_thread': 2}})
        matrix = unit_vectors(result['embeddings'])
        if matrix.shape[0] != len(texts):
            raise ValueError('임베딩 응답 개수가 입력과 다릅니다.')
        return matrix

    def ensure_index(self):
        with self.lock:
            models = self.request('/api/tags').get('models', [])
            entry = next((m for m in models if m.get('name') == self.MODEL), None)
            if not entry or entry.get('remote_model'):
                raise RuntimeError('로컬 embeddinggemma가 필요합니다. ollama pull embeddinggemma 를 실행하세요.')
            corpus = [{'id': d['id'], 'title': d['title'], 'text': d['text']} for d in self.documents]
            fingerprint = hashlib.sha256(json.dumps({'model': self.MODEL, 'digest': entry['digest'],
                'prefix_version': 1, 'corpus': corpus}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if self.metadata and self.metadata['fingerprint'] == fingerprint:
                return
            path = self.cache_dir / f'{fingerprint}.npz'
            if path.exists():
                try:
                    with np.load(path, allow_pickle=False) as saved:
                        vectors = unit_vectors(saved['vectors'])
                        if vectors.shape[0] != len(corpus):
                            raise ValueError('캐시 행 수 오류')
                    self.vectors = vectors
                except (OSError, ValueError):
                    self.vectors = None
            else:
                self.vectors = None
            if self.vectors is None:
                batches = []
                for start in range(0, len(corpus), 8):
                    batches.append(self.embed([f"title: {d['title']} | text: {d['text']}" for d in corpus[start:start+8]]))
                self.vectors = np.concatenate(batches)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.tmp')
                with temporary.open('wb') as file:
                    np.savez_compressed(file, vectors=self.vectors)
                temporary.replace(path)
            self.metadata = {'model': self.MODEL, 'digest': entry['digest'], 'fingerprint': fingerprint,
                             'count': len(corpus), 'dimensions': self.vectors.shape[1], 'cache': str(path)}

    def search(self, query, top_k=3, allowed_ids=None):
        with self.lock:
            self.ensure_index()
            query_vector = self.embed(['task: search result | query: ' + query])[0]
            if self.vectors.shape[1] != len(query_vector):
                raise ValueError('문서/질의 임베딩 차원이 다릅니다.')
            scores = self.vectors @ query_vector
            indices = [int(i) for i in np.argsort(-scores)
                       if allowed_ids is None or self.documents[int(i)]['id'] in allowed_ids][:top_k]
            return [{**self.documents[i], 'retrieval_score': float(scores[i]), 'retrieval_method': 'embedding'} for i in indices]
