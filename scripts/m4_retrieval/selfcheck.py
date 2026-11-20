"""Offline tests; no model downloads or database changes."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .retriever import Retriever


class FakeModel:
    max_seq_length = 256

    def tokenizer(self, text, **kwargs):
        return {"input_ids": text.split()}

    def encode(self, texts, **kwargs):
        return np.array([[1.0, 0.0, 0.0]])


class FakeCollection:
    name = "fixture_collection"

    def query(self, **kwargs):
        self.last_query = kwargs
        return copy.deepcopy(self.raw)


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        path = Path(self.temp.name) / "manifest.json"
        path.write_text("{}", encoding="utf-8")
        self.model = FakeModel()
        self.collection = FakeCollection()
        meta = {
            "document_id": "doc_1", "chunk_index": 0,
            "citation_url": "https://example.com/news",
            "citation_title": "Fixture article",
            "citation_published_at_utc": "2026-01-01T00:00:00Z",
            "citation_source_name": "Example", "citation_source_domain": "example.com",
        }
        self.collection.raw = {
            "ids": [["a" * 64, "b" * 64]],
            "documents": [["First evidence", "Second evidence"]],
            "metadatas": [[copy.deepcopy(meta), copy.deepcopy(meta)]],
            "distances": [[0.3, 0.1]],
        }
        self.retriever = Retriever(
            self.model, {"B": self.collection},
            {"B": {"embedding": {"dimension": 3}, "chunk_count": 2}},
            {"B": path},
        )

    def test_valid_sorted_fewer_than_k(self):
        result = self.retriever.retrieve("news", top_k=5)
        self.assertEqual(result["status"], "OK")
        self.assertEqual([row["chunk_id"] for row in result["results"]], ["b" * 64, "a" * 64])
        self.assertEqual(result["results"][0]["citation"]["url"], "https://example.com/news")
        json.dumps(result, allow_nan=False)

    def test_invalid_requests(self):
        for kwargs in [
            {"query": ""}, {"query": None}, {"query": "x", "top_k": 0},
            {"query": "x", "filters": {"date_from": "2026-01-01"}},
            {"query": "x " * 257},
        ]:
            with self.subTest(kwargs=kwargs):
                result = self.retriever.retrieve(**kwargs)
                self.assertEqual(result["status"], "ERROR")
                self.assertEqual(result["error"]["code"], "INVALID_REQUEST")

    def test_empty_results(self):
        self.collection.raw = {key: [[]] for key in self.collection.raw}
        self.assertEqual(self.retriever.retrieve("news")["status"], "NO_EVIDENCE")

    def test_invalid_evidence(self):
        self.collection.raw["metadatas"][0][0]["citation_url"] = "invalid-url"
        result = self.retriever.retrieve("news")
        self.assertEqual(result["error"]["code"], "INVALID_EVIDENCE")

    def test_wrong_vector_dimension(self):
        self.model.encode = lambda *args, **kwargs: np.array([[1.0, 0.0]])
        result = self.retriever.retrieve("news")
        self.assertEqual(result["error"]["code"], "ENCODING_FAILED")

    def test_filter_translation(self):
        self.retriever.retrieve("news", filters={"language": "en", "section": "world"})
        self.assertEqual(self.collection.last_query["where"], {
            "$and": [
                {"document_language": {"$eq": "en"}},
                {"document_section_id": {"$eq": "world"}},
            ]
        })

    def test_duplicate_ids(self):
        self.collection.raw["ids"][0][1] = "a" * 64
        result = self.retriever.retrieve("news")
        self.assertEqual(result["status"], "OK")
        self.assertEqual(len(result["results"]), 1)
        self.assertTrue(result["warnings"])


def run_checks():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RetrieverTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Offline module checks failed")
    print("PASS: 7 offline module tests")
