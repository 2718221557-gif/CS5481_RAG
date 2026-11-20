"""Dense retrieval using explicitly injected, already validated dependencies."""
import hashlib
import time
import uuid

import numpy as np

from .contracts import citation_fields, validate_response
from .utils import sha256_file


class Retriever:
    """Reuse the model and collections loaded by notebook sections 5 and 6."""

    def __init__(self, model, collections, manifests, manifest_paths):
        self.model = model
        self.collections = dict(collections)
        self.manifests = dict(manifests)
        self.manifest_paths = dict(manifest_paths)
        keys = set(self.manifests)
        if not keys or keys != set(self.collections) or keys != set(self.manifest_paths):
            raise ValueError("Collection, manifest, and manifest-path keys must match")

    def retrieve(self, query, strategy='B', top_k=5, filters=None, candidate_k=30):
        started = time.perf_counter()
        response = {'schema_version': 'retrieval_response.v1', 'request_id': uuid.uuid4().hex, 'query': query if isinstance(query, str) else '', 'retrieval_query': query.strip() if isinstance(query, str) else '', 'status': 'ERROR', 'index': None, 'config': {'mode': 'dense'}, 'filters': None, 'results': [], 'timing_ms': {'encode': 0.0, 'search': 0.0, 'rerank': 0.0, 'total': 0.0}, 'warnings': [], 'error': None}
        error_code = 'INVALID_REQUEST'
        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError('query must be a nonempty string')
            if not isinstance(strategy, str) or strategy not in self.manifests:
                raise ValueError('strategy must be A, B, or C')
            if type(top_k) is not int or type(candidate_k) is not int or (not 1 <= top_k <= candidate_k <= 1000):
                raise ValueError('Require integer values satisfying 1 <= top_k <= candidate_k <= 1000')
            if filters is not None and (not isinstance(filters, dict)):
                raise ValueError('filters must be a dict or None')
            filters = filters or {}
            if set(filters) - {'language', 'section'}:
                raise ValueError('Only language and section filters are supported; date ranges are not implemented')
            if any((not isinstance(v, str) or not v.strip() for v in filters.values())):
                raise ValueError('Filter values must be nonempty strings')
            response['filters'] = dict(filters) or None
            response['config'].update(top_k=top_k, candidate_k=candidate_k)
            token_count = len(self.model.tokenizer(query.strip(), add_special_tokens=True, truncation=False)['input_ids'])
            if token_count > self.model.max_seq_length:
                raise ValueError(f'Query length of {token_count} tokens exceeds the model limit of {self.model.max_seq_length}; shorten the query')
            m, collection = (self.manifests[strategy], self.collections[strategy])
            response['index'] = {'strategy': strategy, 'collection': collection.name, 'manifest_sha256': sha256_file(self.manifest_paths[strategy])}
            conditions = []
            mapping = {'language': 'document_language', 'section': 'document_section_id'}
            for key, value in filters.items():
                conditions.append({mapping[key]: {'$eq': value}})
            kwargs = {}
            if conditions:
                kwargs['where'] = conditions[0] if len(conditions) == 1 else {'$and': conditions}
            error_code = 'ENCODING_FAILED'
            t = time.perf_counter()
            v = self.model.encode([query.strip()], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            response['timing_ms']['encode'] = (time.perf_counter() - t) * 1000
            if v.shape != (1, m['embedding']['dimension']) or not np.isfinite(v).all():
                raise RuntimeError('Invalid query vector dimensions or values')
            if not np.isclose(np.linalg.norm(v[0]), 1, atol=0.0001):
                raise RuntimeError('Query vector is not correctly normalized')
            error_code = 'SEARCH_FAILED'
            t = time.perf_counter()
            raw = collection.query(query_embeddings=v.tolist(), n_results=min(candidate_k, m['chunk_count']), include=['documents', 'metadatas', 'distances'], **kwargs)
            response['timing_ms']['search'] = (time.perf_counter() - t) * 1000
            error_code = 'INVALID_EVIDENCE'
            rows, seen = ([], set())
            for i, cid in enumerate(raw['ids'][0]):
                if cid in seen:
                    response['warnings'].append(f'Ignoring duplicate chunk_id: {cid}')
                    continue
                seen.add(cid)
                meta = raw['metadatas'][0][i]
                text = raw['documents'][0][i]
                distance = float(raw['distances'][0][i])
                if not isinstance(text, str) or not text.strip() or (not np.isfinite(distance)):
                    raise ValueError(f'Invalid candidate text or distance: {cid}')
                rows.append({'rank': len(rows) + 1, 'chunk_id': cid, 'document_id': meta['document_id'], 'chunk_index': meta['chunk_index'], 'text': text, 'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(), 'scores': {'distance': distance, 'similarity': 1 - distance, 'bm25': None, 'rrf': None, 'rerank': None}, 'retrieved_by': ['dense'], 'citation': {key: meta['citation_' + key] for key in citation_fields}})
            rows.sort(key=lambda r: r['scores']['distance'])
            response['results'] = rows[:top_k]
            for rank, row in enumerate(response['results'], 1):
                row['rank'] = rank
            response['status'] = 'OK' if response['results'] else 'NO_EVIDENCE'
            response['timing_ms']['total'] = (time.perf_counter() - started) * 1000
            validate_response(response)
        except Exception as exc:
            response['status'] = 'ERROR'
            response['results'] = []
            response['error'] = {'code': error_code, 'message': f'{type(exc).__name__}: {exc}'}
        response['timing_ms']['total'] = (time.perf_counter() - started) * 1000
        return validate_response(response)
