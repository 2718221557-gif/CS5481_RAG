"""Executable M4 steps 2-7. Code and comments are English only."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import time
import uuid

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(text + "\n", encoding="utf-8")
    temp.replace(path)


def normalize(text):
    return " ".join(text.split())


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Date must include an explicit timezone")
    return int(parsed.astimezone(timezone.utc).timestamp())


def make_schema(v1):
    schema = deepcopy(v1)
    props = schema["properties"]
    props["schema_version"] = {"const": "retrieval_response.v2"}
    props["index"] = {
        "type": ["object", "null"], "additionalProperties": False,
        "required": ["strategy", "collection", "manifest_sha256"],
        "properties": {
            "strategy": {"enum": ["A", "B", "C"]},
            "collection": {"type": "string", "minLength": 1},
            "manifest_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }
    props["config"] = {
        "type": "object", "additionalProperties": False,
        "required": ["mode"],
        "properties": {
            "mode": {"enum": ["dense", "bm25", "hybrid"]},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 1000},
            "candidate_k": {"type": "integer", "minimum": 1, "maximum": 1000},
            "rrf_constant": {"type": "integer", "minimum": 1},
        },
    }
    props["filters"] = {
        "type": ["object", "null"], "additionalProperties": False,
        "properties": {
            "language": {"type": "string", "pattern": "\\S"},
            "section": {"type": "string", "pattern": "\\S"},
            "date_from_ts": {"type": "integer"},
            "date_to_ts": {"type": "integer"},
        },
    }
    row = props["results"]["items"]["properties"]
    row["retrieved_by"] = {
        "type": "array", "minItems": 1, "uniqueItems": True,
        "items": {"enum": ["dense", "bm25"]},
    }
    for name in ("bm25", "rrf"):
        row["scores"]["properties"][name] = {"type": ["number", "null"], "minimum": 0}
    schema["allOf"].append({
        "if": {"properties": {"status": {"enum": ["OK", "NO_EVIDENCE"]}}},
        "then": {"properties": {
            "index": {"type": "object"},
            "config": {"required": ["mode", "top_k", "candidate_k", "rrf_constant"]},
        }},
    })
    Draft202012Validator.check_schema(schema)
    return schema


def validate(response, schema):
    json.dumps(response, allow_nan=False)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(response)
    rows = response["results"]
    if [r["rank"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Ranks must be consecutive")
    if len({r["chunk_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate chunk IDs")
    config = response["config"]
    if "top_k" in config and "candidate_k" in config:
        if config["top_k"] > config["candidate_k"]:
            raise ValueError("top_k exceeds candidate_k")
        if len(rows) > config["top_k"]:
            raise ValueError("Too many returned results")
    for row in rows:
        if hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]:
            raise ValueError("Text hash mismatch")
        scores = row["scores"]
        if (scores["distance"] is None) != (scores["similarity"] is None):
            raise ValueError("Distance and similarity must be present together")
        if scores["distance"] is not None and not math.isclose(
            scores["similarity"], 1 - scores["distance"], abs_tol=1e-6
        ):
            raise ValueError("Inconsistent dense scores")
        if ("dense" in row["retrieved_by"]) != (scores["distance"] is not None):
            raise ValueError("Dense provenance mismatch")
        if ("bm25" in row["retrieved_by"]) != (scores["bm25"] is not None):
            raise ValueError("BM25 provenance mismatch")
        if (config["mode"] == "hybrid") != (scores["rrf"] is not None):
            raise ValueError("RRF score mismatch")
    return response


class BM25:
    """Okapi BM25 with positive log1p IDF, k1=1.5 and b=0.75."""
    def __init__(self, texts):
        self.lengths = []
        self.postings = defaultdict(list)
        for index, text in enumerate(texts):
            counts = Counter(self.tokens(text))
            self.lengths.append(sum(counts.values()))
            for token, count in counts.items():
                self.postings[token].append((index, count))
        self.average = sum(self.lengths) / max(len(self.lengths), 1) or 1.0

    @staticmethod
    def tokens(text):
        return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())

    def search(self, query, allowed, limit):
        scores = defaultdict(float)
        n = len(self.lengths)
        for term in set(self.tokens(query)):
            postings = self.postings.get(term, [])
            idf = math.log1p((n - len(postings) + 0.5) / (len(postings) + 0.5))
            for index, tf in postings:
                if index in allowed:
                    denominator = tf + 1.5 * (0.25 + 0.75 * self.lengths[index] / self.average)
                    scores[index] += idf * tf * 2.5 / denominator
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def fuse(lists, constant=60):
    scores = defaultdict(float)
    for ids in lists:
        for rank, cid in enumerate(dict.fromkeys(ids), 1):
            scores[cid] += 1.0 / (constant + rank)
    return sorted(scores, key=lambda cid: (-scores[cid], cid)), dict(scores)


def evidence_coverage(question, rows, strategy):
    ids = {r["chunk_id"] for r in rows}
    covered = [bool(ids.intersection(e["chunk_ids"][strategy])) for e in question["gold_evidence"]]
    return {"hit": int(any(covered)), "recall": sum(covered) / len(covered),
            "complete": int(all(covered))}


class Workshop:
    def __init__(self, baseline, project_root, data_root, base):
        self.baseline = baseline
        self.model = baseline.model
        self.collections = dict(baseline.collections)
        self.manifests = deepcopy(baseline.manifests)
        self.manifest_paths = dict(baseline.manifest_paths)
        self.project_root, self.data_root, self.base = map(Path, (project_root, data_root, base))
        self.output = self.base / "m4_workshop_results"
        self.output.mkdir(exist_ok=True)
        old_path = self.project_root / "schemas/retrieval_response.schema.json"
        self.schema = make_schema(json.loads(old_path.read_text(encoding="utf-8")))
        self.schema_path = self.project_root / "schemas/retrieval_response.v2.schema.json"
        if self.schema_path.exists() and json.loads(self.schema_path.read_text(encoding="utf-8")) != self.schema:
            raise RuntimeError("An incompatible v2 schema already exists; review it first")
        save_json(self.schema_path, self.schema)
        self.rows, self.by_id, self.lexical = {}, {}, {}
        self.docs = {}
        self.dates_ready = False
        self.qa_path = self.project_root / "data/evaluation/qa_reviewed.json"
        self.validate_startup()
        print("WORKSHOP READY; schema:", self.schema_path)

    def validate_startup(self):
        if set(self.manifests) != {"A", "B", "C"}:
            raise ValueError("All three strategies A/B/C are required")
        spec = self.manifests["B"]["embedding"]
        dimension = getattr(self.model, "get_embedding_dimension", None)
        if not callable(dimension):
            dimension = self.model.get_sentence_embedding_dimension
        if dimension() != spec["dimension"]:
            raise ValueError("Model dimension mismatch")
        actual_revision = getattr(self.model._first_module().auto_model.config, "_commit_hash", None)
        if actual_revision is None:
            loaded_path = Path(str(getattr(self.model._first_module().auto_model.config, "_name_or_path", "")))
            expected_cache_name = "models--" + spec["model"].replace("/", "--")
            if loaded_path.parent.name == "snapshots" and loaded_path.parent.parent.name == expected_cache_name:
                actual_revision = loaded_path.name
        if actual_revision != spec["revision"]:
            raise ValueError("Model revision cannot be verified; load the pinned local snapshot in section 0C")
        for strategy, manifest in self.manifests.items():
            if manifest["embedding"] != spec:
                raise ValueError("Embedding identity differs between strategies")
            collection = self.collections[strategy]
            if collection.count() != manifest["chunk_count"]:
                raise ValueError("Collection count mismatch")
            metadata = collection.metadata or {}
            for key, expected in {"model_name": spec["model"], "model_revision": spec["revision"],
                                  "embedding_dimension": spec["dimension"]}.items():
                if metadata.get(key) != expected:
                    raise ValueError("Collection metadata mismatch: " + key)
        print("Startup identity and dimension checks passed")

    def load_corpus(self):
        if self.rows:
            return
        for strategy, collection in self.collections.items():
            rows = []
            for offset in range(0, collection.count(), 1000):
                batch = collection.get(offset=offset, limit=1000, include=["documents", "metadatas"])
                for cid, text, meta in zip(batch["ids"], batch["documents"], batch["metadatas"]):
                    rows.append({"chunk_id": cid, "text": text, "meta": meta})
            rows.sort(key=lambda row: row["chunk_id"])
            if len({r["chunk_id"] for r in rows}) != self.manifests[strategy]["chunk_count"]:
                raise ValueError("Corpus IDs or count mismatch")
            self.rows[strategy] = rows
            self.by_id[strategy] = {r["chunk_id"]: r for r in rows}
            print("Corpus loaded:", strategy, len(rows))
        path = self.data_root / "data/documents/documents.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    self.docs[row["document_id"]] = row

    def build_bm25(self):
        self.load_corpus()
        for strategy in "ABC":
            if strategy not in self.lexical:
                self.lexical[strategy] = BM25([r["text"] for r in self.rows[strategy]])
                print("BM25 ready:", strategy)

    @staticmethod
    def where(filters):
        mapping = {"language": "document_language", "section": "document_section_id"}
        conditions = [{mapping[k]: {"$eq": v}} for k, v in filters.items() if k in mapping]
        if "date_from_ts" in filters:
            conditions.append({"document_published_at_ts": {"$gte": filters["date_from_ts"]}})
        if "date_to_ts" in filters:
            conditions.append({"document_published_at_ts": {"$lt": filters["date_to_ts"]}})
        return conditions[0] if len(conditions) == 1 else {"$and": conditions} if conditions else None

    @staticmethod
    def matches(meta, filters):
        for key, field in {"language": "document_language", "section": "document_section_id"}.items():
            if key in filters and meta.get(field) != filters[key]:
                return False
        if "date_from_ts" in filters or "date_to_ts" in filters:
            value = meta.get("document_published_at_ts")
            if type(value) is not int:
                return False
            if "date_from_ts" in filters and value < filters["date_from_ts"]:
                return False
            if "date_to_ts" in filters and value >= filters["date_to_ts"]:
                return False
        return True

    def retrieve(self, query, strategy="B", top_k=5, candidate_k=30, filters=None,
                 mode="dense", rrf_constant=60):
        started = time.perf_counter()
        response = {"schema_version": "retrieval_response.v2", "request_id": uuid.uuid4().hex,
                    "query": query if isinstance(query, str) else "",
                    "retrieval_query": query.strip() if isinstance(query, str) else "",
                    "status": "ERROR", "index": None, "config": {"mode": "dense"},
                    "filters": None, "results": [], "warnings": [], "error": None,
                    "timing_ms": {k: 0.0 for k in ("encode", "search", "rerank", "total")}}
        error_code = "INVALID_REQUEST"
        try:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a nonempty string")
            if not isinstance(strategy, str) or strategy not in self.collections:
                raise ValueError("Strategy must be A/B/C")
            if mode not in ("dense", "bm25", "hybrid"):
                raise ValueError("Unknown retrieval mode")
            if type(top_k) is not int or type(candidate_k) is not int or not 1 <= top_k <= candidate_k <= 1000:
                raise ValueError("Require 1 <= top_k <= candidate_k <= 1000")
            if type(rrf_constant) is not int or rrf_constant < 1:
                raise ValueError("RRF constant must be a positive integer")
            if filters is not None and not isinstance(filters, dict):
                raise ValueError("Filters must be a dict or None")
            filters = filters or {}
            if set(filters) - {"language", "section", "date_from_ts", "date_to_ts"}:
                raise ValueError("Unsupported filter")
            for key, value in filters.items():
                if key.startswith("date_"):
                    if type(value) is not int:
                        raise ValueError("Date boundaries must be integer UTC timestamps")
                    if not self.dates_ready:
                        raise ValueError("Build and activate the derived date index first")
                elif not isinstance(value, str) or not value.strip():
                    raise ValueError("Language and section must be nonempty strings")
            if all(k in filters for k in ("date_from_ts", "date_to_ts")):
                if filters["date_from_ts"] >= filters["date_to_ts"]:
                    raise ValueError("Date start must be before date end")
            token_count = len(self.model.tokenizer(query.strip(), truncation=False)["input_ids"])
            if token_count > self.model.max_seq_length:
                raise ValueError("Query exceeds the embedding model token limit")
            response["config"] = dict(mode=mode, top_k=top_k, candidate_k=candidate_k, rrf_constant=rrf_constant)
            response["filters"] = filters.copy() or None
            collection = self.collections[strategy]
            response["index"] = dict(strategy=strategy, collection=collection.name,
                                     manifest_sha256=digest(self.manifest_paths[strategy]))
            dense, lexical, records = {}, {}, {}
            if mode in ("dense", "hybrid"):
                error_code = "ENCODING_FAILED"
                t = time.perf_counter()
                vector = self.model.encode([query.strip()], normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False)
                dimension = self.manifests[strategy]["embedding"]["dimension"]
                if vector.shape != (1, dimension) or not np.isfinite(vector).all():
                    raise ValueError("Invalid query vector dimensions or values")
                if not np.isclose(np.linalg.norm(vector[0]), 1.0, atol=1e-4):
                    raise ValueError("Query vector must be normalized")
                response["timing_ms"]["encode"] = (time.perf_counter() - t) * 1000
                error_code = "SEARCH_FAILED"
                t = time.perf_counter()
                kwargs = {"where": self.where(filters)} if filters else {}
                raw = collection.query(query_embeddings=vector.tolist(),
                    n_results=min(candidate_k, self.manifests[strategy]["chunk_count"]),
                    include=["documents", "metadatas", "distances"], **kwargs)
                for index, cid in enumerate(raw["ids"][0]):
                    if cid in dense:
                        raise ValueError("Duplicate dense candidate")
                    dense[cid] = float(raw["distances"][0][index])
                    records[cid] = {"chunk_id": cid, "text": raw["documents"][0][index],
                                    "meta": raw["metadatas"][0][index]}
                response["timing_ms"]["search"] += (time.perf_counter() - t) * 1000
            if mode in ("bm25", "hybrid"):
                error_code = "BM25_NOT_READY"
                if strategy not in self.lexical:
                    raise RuntimeError("Call build_bm25() before lexical or hybrid retrieval")
                error_code = "SEARCH_FAILED"
                t = time.perf_counter()
                rows = self.rows[strategy]
                allowed = {i for i, row in enumerate(rows) if self.matches(row["meta"], filters)}
                for i, score in self.lexical[strategy].search(query, allowed, candidate_k):
                    row = rows[i]
                    cid = row["chunk_id"]
                    lexical[cid] = score
                    records.setdefault(cid, row)
                response["timing_ms"]["search"] += (time.perf_counter() - t) * 1000
            error_code = "INVALID_EVIDENCE"
            dense_ids = sorted(dense, key=lambda cid: (dense[cid], cid))
            lexical_ids = sorted(lexical, key=lambda cid: (-lexical[cid], cid))
            if mode == "hybrid":
                ranked, rrf_scores = fuse([dense_ids, lexical_ids], rrf_constant)
            else:
                ranked, rrf_scores = (dense_ids if mode == "dense" else lexical_ids), {}
            citation_keys = ("url", "title", "published_at_utc", "source_name", "source_domain")
            for cid in ranked[:top_k]:
                row = records[cid]
                text, meta = row["text"], row["meta"]
                if not isinstance(text, str) or not text.strip() or not self.matches(meta, filters):
                    raise ValueError("Invalid text or candidate violates filters")
                response["results"].append({
                    "rank": len(response["results"]) + 1, "chunk_id": cid,
                    "document_id": meta["document_id"], "chunk_index": meta["chunk_index"],
                    "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "citation": {k: meta["citation_" + k] for k in citation_keys},
                    "retrieved_by": [name for name, scores in (("dense", dense), ("bm25", lexical)) if cid in scores],
                    "scores": {"distance": dense.get(cid), "similarity": 1 - dense[cid] if cid in dense else None,
                               "bm25": lexical.get(cid), "rrf": rrf_scores.get(cid), "rerank": None},
                })
            response["status"] = "OK" if response["results"] else "NO_EVIDENCE"
            validate(response, self.schema)
        except Exception as exc:
            response.update(status="ERROR", results=[], error={"code": error_code, "message": str(exc)})
        response["timing_ms"]["total"] = (time.perf_counter() - started) * 1000
        return validate(response, self.schema)

    def export_contract_examples(self):
        examples = {
            "OK": self.retrieve("What happened to Australian tourists in Nepal?"),
            "NO_EVIDENCE": self.retrieve("news", filters={"language": "__missing_language__"}),
            "ERROR": self.retrieve(""),
        }
        folder = self.project_root / "documents/m4/contract_examples_v2"
        for expected, response in examples.items():
            if response["status"] != expected:
                raise RuntimeError(f"Expected {expected}, received {response}")
            validate(response, self.schema)
            save_json(folder / (expected + ".json"), response)
        mutations = []
        for field in ("url", "title", "published_at_utc"):
            broken = deepcopy(examples["OK"])
            del broken["results"][0]["citation"][field]
            mutations.append(broken)
        broken = deepcopy(examples["OK"])
        broken["config"]["top_k"] = "five"
        mutations.append(broken)
        broken = deepcopy(examples["NO_EVIDENCE"])
        broken["index"] = None
        mutations.append(broken)
        for broken in mutations:
            try:
                validate(broken, self.schema)
            except Exception:
                continue
            raise AssertionError("Malformed response was accepted")
        print("CONTRACT CHECKS PASSED; actual response examples:", folder)
        return examples

    def find_documents(self, term="", limit=15):
        self.load_corpus()
        hits = [d for d in self.docs.values() if term.lower() in (d["title"] + " " + d["body_text"]).lower()]
        for doc in hits[:limit]:
            print(doc["document_id"], "\n", doc["title"], "\n", doc["url"], "\n")
        print("Matching documents:", len(hits))

    def show_document(self, document_id):
        self.load_corpus()
        doc = self.docs[document_id]
        print(doc["title"], "\n", doc["url"], "\n\n", doc["body_text"])

    def load_qa(self):
        return json.loads(self.qa_path.read_text(encoding="utf-8")) if self.qa_path.exists() else []

    def add_question(self, question_id, split, question, reference_answer, evidence,
                     reviewer, confirmed=False, answerable=True, no_answer_reason=""):
        self.load_corpus()
        if confirmed is not True or not reviewer.strip():
            raise ValueError("Human review must be explicitly confirmed and named")
        if split not in ("dev", "test") or not question_id.strip() or not question.strip():
            raise ValueError("Provide an ID, dev/test split, and nonempty question")
        records = self.load_qa()
        if any(r["question_id"] == question_id or normalize(r["question"]).lower() == normalize(question).lower()
               for r in records):
            raise ValueError("Duplicate question ID or question text")
        if type(answerable) is not bool:
            raise ValueError("answerable must be boolean")
        gold = []
        if answerable:
            if not reference_answer.strip() or not evidence:
                raise ValueError("Answerable questions require an answer and supporting passages")
            for number, item in enumerate(evidence, 1):
                document_id, quote = item["document_id"], normalize(item["quote"])
                if not quote or quote not in normalize(self.docs[document_id]["body_text"]):
                    raise ValueError("Evidence quote must occur in the original document body")
                matches = {}
                for strategy in "ABC":
                    matches[strategy] = [r["chunk_id"] for r in self.rows[strategy]
                        if r["meta"]["document_id"] == document_id and quote in normalize(r["text"])]
                    if not matches[strategy]:
                        raise ValueError(f"Quote has no complete match in {strategy}; split it into smaller evidence units")
                gold.append({"evidence_id": f"e{number}", "document_id": document_id,
                             "quote": quote, "chunk_ids": matches})
            current_docs = {g["document_id"] for g in gold}
            other_docs = {e["document_id"] for r in records if r["split"] != split for e in r["gold_evidence"]}
            if current_docs & other_docs:
                raise ValueError("A source document cannot appear in both dev and test")
        elif evidence or reference_answer.strip() or not no_answer_reason.strip():
            raise ValueError("Unanswerable questions need empty evidence/answer and a documented review reason")
        record = dict(question_id=question_id, split=split, question=question.strip(),
                      answerable=answerable, reference_answer=reference_answer.strip(),
                      gold_evidence=gold, reviewer=reviewer.strip(), reviewed=True,
                      no_answer_reason=no_answer_reason.strip())
        records.append(record)
        if self.qa_path.exists():
            backup = self.qa_path.with_name("qa_reviewed.backup-" + uuid.uuid4().hex + ".json")
            backup.write_bytes(self.qa_path.read_bytes())
        save_json(self.qa_path, records)
        print("QUESTION SAVED:", question_id)
        self.qa_progress()

    def qa_progress(self):
        records = self.load_qa()
        counts = {f"{split}_{kind}": sum(r["split"] == split and r["answerable"] == flag for r in records)
                  for split in ("dev", "test") for kind, flag in (("answerable", True), ("unanswerable", False))}
        print(json.dumps(counts, indent=2))
        return counts

    def remove_question(self, question_id, confirmed=False):
        if confirmed is not True:
            raise ValueError("Confirm removal explicitly")
        records = self.load_qa()
        updated = [r for r in records if r["question_id"] != question_id]
        if len(records) == len(updated):
            raise ValueError("Question ID was not found")
        backup = self.qa_path.with_name("qa_reviewed.backup-" + uuid.uuid4().hex + ".json")
        backup.write_bytes(self.qa_path.read_bytes())
        save_json(self.qa_path, updated)
        print("Removed question; previous QA saved to:", backup)

    @staticmethod
    def date_filters(start, end):
        lower, upper = timestamp(start), timestamp(end)
        if lower >= upper:
            raise ValueError("Date start must precede end")
        return {"date_from_ts": lower, "date_to_ts": upper}

    def unit_checks(self):
        from .workshop_checks import run_checks
        return run_checks(self.schema)

    def pack(self, rows, token_budget=1024):
        selected, context = [], ""
        for row in rows:
            block = f"[{row['chunk_id']}] {row['citation']['title']}\n{row['citation']['url']}\n{row['text']}\n\n"
            candidate = context + block
            count = len(self.model.tokenizer(candidate, truncation=False, add_special_tokens=False,
                                             verbose=False)["input_ids"])
            if count > token_budget:
                break
            selected.append(row)
            context = candidate
        used = len(self.model.tokenizer(context, truncation=False, add_special_tokens=False,
                                       verbose=False)["input_ids"])
        return selected, context, used

    def run_grid(self, split="dev", modes=("dense",), strategies=("A", "B", "C"),
                 ks=(3, 5, 10), token_budget=1024, frozen_config=None):
        if split not in ("dev", "test"):
            raise ValueError("Split must be dev or test")
        if type(token_budget) is not int or token_budget < 1:
            raise ValueError("Token budget must be a positive integer")
        if split == "test":
            if frozen_config is None:
                raise ValueError("Freeze a development configuration before testing")
            config = json.loads(Path(frozen_config).read_text(encoding="utf-8"))
            if config["qa_sha256"] != digest(self.qa_path):
                raise ValueError("QA changed after configuration selection")
            if config["schema_sha256"] != digest(self.schema_path):
                raise ValueError("Schema changed after configuration selection")
            if config["index_hashes"] != {s: digest(p) for s, p in self.manifest_paths.items()}:
                raise ValueError("Index changed after configuration selection")
            modes, strategies, ks = (config["mode"],), (config["strategy"],), (config["top_k"],)
            token_budget = config["token_budget"]
        counts = self.qa_progress()
        needed = 8 if split == "dev" else 20
        if counts[f"{split}_answerable"] < needed:
            raise ValueError(f"Need at least {needed} reviewed answerable {split} questions")
        if counts[f"{split}_unanswerable"] < (2 if split == "dev" else 3):
            raise ValueError("Need at least 2 dev / 3 test reviewed unanswerable questions")
        questions = [q for q in self.load_qa() if q["split"] == split]
        if any(mode in ("bm25", "hybrid") for mode in modes):
            self.build_bm25()
        folder = self.output / (split + "-" + uuid.uuid4().hex)
        folder.mkdir()
        spec = self.manifests["B"]["embedding"]
        from importlib.metadata import version
        metadata = dict(qa_sha256=digest(self.qa_path), schema_sha256=digest(self.schema_path),
                        index_hashes={s: digest(p) for s, p in self.manifest_paths.items()},
                        token_budget=token_budget, budget_tokenizer=spec,
                        budget_kind="embedding-tokenizer proxy; not the final M5 tokenizer",
                        candidate_k=30, rrf_constant=60, bm25="positive log1p IDF; k1=1.5; b=0.75",
                        source_sha256=digest(__file__), split=split,
                        packages={name: version(name) for name in ("chromadb", "sentence-transformers", "numpy", "jsonschema")})
        save_json(folder / "run_metadata.json", metadata)
        summary = []
        with (folder / "responses.jsonl").open("w", encoding="utf-8") as stream:
            for mode in modes:
                for strategy in strategies:
                    warmup = self.retrieve(questions[0]["question"], strategy=strategy, mode=mode)
                    if warmup["status"] == "ERROR":
                        raise RuntimeError(warmup["error"])
                    for k in ks:
                        measurements, latencies, errors = [], [], 0
                        for question in questions:
                            response = self.retrieve(question["question"], strategy=strategy, top_k=k, mode=mode)
                            selected, context, used = self.pack(response["results"], token_budget)
                            record = dict(question_id=question["question_id"], answerable=question["answerable"],
                                          retrieval=response, context=context, context_tokens=used,
                                          selected_chunk_ids=[r["chunk_id"] for r in selected])
                            if question["answerable"]:
                                record["raw_metrics"] = evidence_coverage(question, response["results"], strategy)
                                record["budget_metrics"] = evidence_coverage(question, selected, strategy)
                                measurements.append(record)
                            errors += response["status"] == "ERROR"
                            latencies.append(response["timing_ms"]["total"])
                            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        row = dict(mode=mode, strategy=strategy, top_k=k, answerable_count=len(measurements),
                                   total_count=len(questions), error_count=errors,
                                   latency_p50_ms=float(np.percentile(latencies, 50)),
                                   latency_p95_ms=float(np.percentile(latencies, 95)))
                        for group, prefix in (("raw_metrics", "raw"), ("budget_metrics", "budget")):
                            for metric in ("hit", "recall", "complete"):
                                row[f"{prefix}_{metric}"] = sum(r[group][metric] for r in measurements) / len(measurements)
                        summary.append(row)
                        print(mode, strategy, k, "Hit:", round(row["raw_hit"], 3),
                              "Budget recall:", round(row["budget_recall"], 3), "Errors:", errors, flush=True)
        save_json(folder / "summary.json", summary)
        with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print("EXPERIMENT SAVED:", folder)
        return folder

    def freeze(self, dev_folder, mode, strategy, top_k):
        folder = Path(dev_folder)
        metadata = json.loads((folder / "run_metadata.json").read_text(encoding="utf-8"))
        rows = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        matches = [r for r in rows if r["mode"] == mode and r["strategy"] == strategy and r["top_k"] == top_k]
        if metadata["split"] != "dev" or len(matches) != 1 or matches[0]["error_count"]:
            raise ValueError("Select an error-free configuration from a development run")
        if metadata["qa_sha256"] != digest(self.qa_path):
            raise ValueError("QA changed since development evaluation; rerun development")
        config = {**metadata, "mode": mode, "strategy": strategy, "top_k": top_k,
                  "development_folder": str(folder)}
        path = self.output / ("frozen_config-" + uuid.uuid4().hex + ".json")
        save_json(path, config)
        print("CONFIGURATION FROZEN:", path)
        return path

    def create_date_index(self):
        if self.dates_ready:
            print("Derived date index already active in this Workshop instance")
            return
        self.load_corpus()
        times = {}
        for strategy, rows in self.rows.items():
            times[strategy] = {r["chunk_id"]: timestamp(r["meta"]["citation_published_at_utc"]) for r in rows}
        import chromadb
        folder = self.base / ("m4_date_index_" + uuid.uuid4().hex)
        client = chromadb.PersistentClient(path=str(folder / "chroma"))
        new_collections, new_manifests, new_paths = {}, {}, {}
        for strategy in "ABC":
            source = self.collections[strategy]
            destination = client.create_collection(
                name=source.name + "_dates", embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}}, metadata=dict(source.metadata or {}))
            for offset in range(0, source.count(), 500):
                batch = source.get(offset=offset, limit=500, include=["documents", "metadatas", "embeddings"])
                metadata = []
                for cid, meta in zip(batch["ids"], batch["metadatas"]):
                    metadata.append({**meta, "document_published_at_ts": times[strategy][cid]})
                destination.add(ids=batch["ids"], embeddings=np.asarray(batch["embeddings"]).tolist(),
                                documents=batch["documents"], metadatas=metadata)
            if destination.count() != source.count():
                raise ValueError("Derived index count mismatch")
            augmentation_path = folder / (strategy + "_timestamps.json")
            save_json(augmentation_path, times[strategy])
            manifest = deepcopy(self.manifests[strategy])
            manifest["vector_store"].update(collection=destination.name, persist_path=str(folder / "chroma"))
            manifest["derived_metadata"] = dict(field="document_published_at_ts", unit="UTC Unix seconds",
                interval="inclusive start, exclusive end", source_manifest_sha256=digest(self.manifest_paths[strategy]),
                timestamps_sha256=digest(augmentation_path), date_policy="reject missing or timezone-naive dates")
            path = folder / (strategy + ".manifest.json")
            save_json(path, manifest)
            new_collections[strategy], new_manifests[strategy], new_paths[strategy] = destination, manifest, path
            print("Derived date collection ready:", strategy, destination.count(), flush=True)
        self.collections, self.manifests, self.manifest_paths = new_collections, new_manifests, new_paths
        self.date_client = client
        for strategy, rows in self.rows.items():
            for row in rows:
                row["meta"]["document_published_at_ts"] = times[strategy][row["chunk_id"]]
        self.dates_ready = True
        save_json(folder / "READY.json", {"collections": {s: c.name for s, c in self.collections.items()}})
        print("DATE INDEX ACTIVE:", folder)

    def open_date_index(self, folder):
        folder = Path(folder)
        if not (folder / "READY.json").is_file():
            raise ValueError("Derived date index has no completion marker")
        import chromadb
        client = chromadb.PersistentClient(path=str(folder / "chroma"))
        collections, manifests, paths = {}, {}, {}
        self.load_corpus()
        augmented = {}
        for strategy in "ABC":
            path = folder / (strategy + ".manifest.json")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            mapping_path = folder / (strategy + "_timestamps.json")
            if digest(mapping_path) != manifest["derived_metadata"]["timestamps_sha256"]:
                raise ValueError("Timestamp mapping hash mismatch")
            if manifest["embedding"] != self.manifests[strategy]["embedding"]:
                raise ValueError("Derived index embedding identity mismatch")
            augmented[strategy] = json.loads(mapping_path.read_text(encoding="utf-8"))
            if set(augmented[strategy]) != set(self.by_id[strategy]):
                raise ValueError("Derived chunk IDs do not match the loaded corpus")
            collection = client.get_collection(manifest["vector_store"]["collection"], embedding_function=None)
            if collection.count() != manifest["chunk_count"]:
                raise ValueError("Derived index count mismatch")
            collections[strategy], manifests[strategy], paths[strategy] = collection, manifest, path
        self.collections, self.manifests, self.manifest_paths = collections, manifests, paths
        self.date_client, self.dates_ready = client, True
        for strategy, rows in self.rows.items():
            for row in rows:
                row["meta"]["document_published_at_ts"] = augmented[strategy][row["chunk_id"]]
        self.validate_startup()
        print("DATE INDEX REOPENED:", folder)

    def integration_checks(self):
        self.validate_startup()
        self.export_contract_examples()
        self.build_bm25()
        checks = 0
        for mode in ("dense", "bm25", "hybrid"):
            response = self.retrieve("Australian tourists Nepal", mode=mode,
                                     filters={"language": "en", "section": "world"})
            if response["status"] != "OK":
                raise AssertionError(response)
            for row in response["results"]:
                saved = self.collections["B"].get(ids=[row["chunk_id"]], include=["documents", "metadatas"])
                assert saved["documents"][0] == row["text"]
                for key, value in row["citation"].items():
                    assert saved["metadatas"][0]["citation_" + key] == value
                assert self.matches(saved["metadatas"][0], {"language": "en", "section": "world"})
            checks += 1
        assert self.retrieve("", mode="hybrid")["status"] == "ERROR"
        assert self.retrieve("news", top_k=0)["status"] == "ERROR"
        assert self.retrieve("news", filters={"unknown": "x"})["status"] == "ERROR"
        if self.dates_ready:
            ts = self.rows["B"][0]["meta"]["document_published_at_ts"]
            for mode in ("dense", "bm25", "hybrid"):
                response = self.retrieve("news", mode=mode, filters={"date_from_ts": ts, "date_to_ts": ts + 1})
                assert response["status"] != "ERROR", response
                for row in response["results"]:
                    assert timestamp(row["citation"]["published_at_utc"]) == ts
            assert self.retrieve("news", filters={"date_from_ts": ts, "date_to_ts": ts})["status"] == "ERROR"
        record = {"status": "PASS", "checked_modes": checks, "date_checks": self.dates_ready,
                  "schema_sha256": digest(self.schema_path), "created_at": datetime.now(timezone.utc).isoformat()}
        save_json(self.output / ("integration-checks-" + uuid.uuid4().hex + ".json"), record)
        print("INTEGRATION CHECKS PASSED")
        return record
