"""Fast v2 tests without model downloads or changes to the real database."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from .workshop import BM25, Workshop, fuse, timestamp, validate


class Model:
    max_seq_length = 256

    def tokenizer(self, text, **kwargs):
        return {"input_ids": text.split()}

    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0, 0.0]])

    def get_embedding_dimension(self):
        return 3

    def _first_module(self):
        return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash="a" * 40)))


class Collection:
    name = "fixture"
    metadata = {"model_name": "fixture", "model_revision": "a" * 40, "embedding_dimension": 3}

    def count(self):
        return 1

    def query(self, **kwargs):
        self.last_query = kwargs
        return deepcopy(self.raw)


class WorkshopTests(unittest.TestCase):
    schema = None

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "manifest.json"
        path.write_text("{}", encoding="utf-8")
        self.w = Workshop.__new__(Workshop)
        self.w.schema = self.schema
        self.w.model = Model()
        self.collection = Collection()
        meta = {"document_id": "fixture-doc", "chunk_index": 0,
                "citation_url": "https://example.com/news", "citation_title": "Fixture",
                "citation_published_at_utc": "2026-01-01T00:00:00Z",
                "citation_source_name": "Example", "citation_source_domain": "example.com",
                "document_language": "en", "document_section_id": "world"}
        self.collection.raw = {"ids": [["a" * 64]], "documents": [["news evidence"]],
                               "metadatas": [[meta]], "distances": [[0.1]]}
        self.w.collections = {s: self.collection for s in "ABC"}
        self.w.manifests = {s: {"chunk_count": 1, "embedding": {
            "model": "fixture", "revision": "a" * 40, "dimension": 3}} for s in "ABC"}
        self.w.manifest_paths = {s: path for s in "ABC"}
        self.w.dates_ready = False
        self.w.rows, self.w.lexical = {}, {}

    def test_fewer_than_k(self):
        result = self.w.retrieve("news", top_k=5)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(len(result["results"]), 1)

    def test_empty_and_errors(self):
        for kwargs in [{"query": ""}, {"query": "news", "top_k": 0},
                       {"query": "news", "filters": {"unknown": "x"}},
                       {"query": "news", "filters": {"date_from_ts": 1}}]:
            self.assertEqual(self.w.retrieve(**kwargs)["status"], "ERROR")
        self.collection.raw = {k: [[]] for k in self.collection.raw}
        self.assertEqual(self.w.retrieve("news")["status"], "NO_EVIDENCE")

    def test_vector_dimension(self):
        self.w.model.encode = lambda *a, **k: np.array([[1.0, 0.0]])
        self.assertEqual(self.w.retrieve("news")["error"]["code"], "ENCODING_FAILED")

    def test_metadata_identity(self):
        self.w.validate_startup()
        self.w.manifests["B"]["embedding"]["revision"] = "b" * 40
        with self.assertRaises(ValueError):
            self.w.validate_startup()

    def test_wrong_citation(self):
        self.collection.raw["metadatas"][0][0]["citation_url"] = "invalid"
        self.assertEqual(self.w.retrieve("news")["error"]["code"], "INVALID_EVIDENCE")

    def test_filter_logic(self):
        result = self.w.retrieve("news", filters={"language": "en", "section": "world"})
        self.assertEqual(result["status"], "OK")
        self.assertEqual(len(self.collection.last_query["where"]["$and"]), 2)
        self.assertEqual(self.w.retrieve("news", filters={"language": "fr"})["status"], "ERROR")

    def test_date_boundaries(self):
        self.assertTrue(Workshop.matches({"document_published_at_ts": 10}, {"date_from_ts": 10, "date_to_ts": 11}))
        self.assertFalse(Workshop.matches({"document_published_at_ts": 11}, {"date_from_ts": 10, "date_to_ts": 11}))
        self.assertEqual(timestamp("2026-01-01T08:00:00+08:00"), timestamp("2026-01-01T00:00:00Z"))
        with self.assertRaises(ValueError):
            timestamp("2026-01-01")

    def test_rrf_and_bm25(self):
        ranked, _ = fuse([["a", "b"], ["b", "c"]])
        self.assertEqual(ranked[0], "b")
        self.assertEqual(len(ranked), 3)
        index = BM25(["alpha beta", "gamma"])
        self.assertEqual(index.search("alpha", {1}, 5), [])
        self.assertEqual(index.search("unknown", {0, 1}, 5), [])

    def test_schema_rejects_invalid_config(self):
        response = self.w.retrieve("news")
        response["config"]["top_k"] = "five"
        with self.assertRaises(Exception):
            validate(response, self.schema)

    def test_batch_keeps_error(self):
        records = [self.w.retrieve(query) for query in ("news", "", "news")]
        self.assertEqual([r["status"] for r in records], ["OK", "ERROR", "OK"])
        text = "\n".join(json.dumps(r, allow_nan=False) for r in records)
        self.assertEqual(len([json.loads(line) for line in text.splitlines()]), 3)


def run_checks(schema):
    WorkshopTests.schema = schema
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(WorkshopTests))
    if not result.wasSuccessful():
        raise RuntimeError("Workshop unit tests failed")
    print("UNIT CHECKS PASSED:", result.testsRun)
    return {"status": "PASS", "tests": result.testsRun}
