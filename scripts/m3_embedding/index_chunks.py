from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import chromadb
import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from scripts.common.jsonl import sha256_file
from scripts.common.paths import DATA_ROOT, PROJECT_ROOT

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64
DEFAULT_STRATEGIES = ("A", "B", "C")
CHUNK_SCHEMA_VERSION = "news_chunks.v1"


class EmbeddingCache:
    """Small SQLite cache keyed by model revision and normalized chunk text."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL
            )"""
        )
        self.connection.commit()

    def get_many(self, keys: Iterable[str]) -> dict[str, np.ndarray]:
        keys = list(dict.fromkeys(keys))
        found: dict[str, np.ndarray] = {}
        # Stay below SQLite's bound-variable limit on older Python/SQLite builds.
        for start in range(0, len(keys), 500):
            batch = keys[start:start + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT cache_key, dimension, vector FROM embeddings WHERE cache_key IN ({placeholders})",
                batch,
            )
            for key, dimension, blob in rows:
                found[key] = np.frombuffer(blob, dtype=np.float32, count=dimension).copy()
        return found

    def put_many(self, vectors: dict[str, np.ndarray]) -> None:
        if not vectors:
            return
        rows = [
            (key, int(vector.size), np.asarray(vector, dtype=np.float32).tobytes())
            for key, vector in vectors.items()
        ]
        with self.connection:
            self.connection.executemany(
                "INSERT OR REPLACE INTO embeddings(cache_key, dimension, vector) VALUES (?, ?, ?)",
                rows,
            )

    def close(self) -> None:
        self.connection.close()


class SentenceEmbeddingModel:
    """Local Sentence Transformers model shared by indexing and M4 query encoding."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        kwargs = {"device": device} if device else {}
        self.model = SentenceTransformer(model_name, **kwargs)
        dimension_fn = getattr(self.model, "get_embedding_dimension", None)
        if dimension_fn is None:
            dimension_fn = self.model.get_sentence_embedding_dimension
        self.dimension = int(dimension_fn())
        model_config = getattr(getattr(self.model, "_first_module", lambda: None)(), "auto_model", None)
        config = getattr(model_config, "config", None)
        self.revision = getattr(config, "_commit_hash", None)
        if not self.revision and Path(model_name).is_dir():
            config_path = Path(model_name) / "config.json"
            if config_path.is_file():
                self.revision = "local-" + hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]
        if not self.revision:
            try:
                snapshot_path = Path(snapshot_download(model_name, local_files_only=True))
                candidate = snapshot_path.name
                if re.fullmatch(r"[0-9a-f]{40}", candidate):
                    self.revision = candidate
            except Exception:
                pass
        self.revision = self.revision or "unresolved"
        self.fingerprint = hashlib.sha256(
            f"{model_name}@{self.revision}".encode("utf-8")
        ).hexdigest()

    def encode_documents(self, texts: list[str], batch_size: int) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )[0]
        return np.asarray(vector, dtype=np.float32).tolist()


@dataclass(frozen=True)
class IndexResult:
    collection_name: str
    input_path: str
    input_sha256: str
    chunk_count: int
    embedding_dimension: int
    cached_vectors_used: int
    vectors_created: int
    manifest_path: str


def _cache_key(model_fingerprint: str, text: str) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return hashlib.sha256(f"{model_fingerprint}:{text_hash}".encode("ascii")).hexdigest()


def _metadata(record: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Flatten chunk/citation fields into Chroma's scalar metadata contract."""
    result: dict[str, str | int | float | bool] = {
        "schema_version": str(record.get("schema_version", CHUNK_SCHEMA_VERSION)),
        "document_id": str(record["document_id"]),
        "chunk_index": int(record["chunk_index"]),
        "strategy": str(record["strategy"]),
    }
    citation = record.get("citation") or {}
    for key in ("url", "title", "published_at_utc", "source_name", "source_domain"):
        value = citation.get(key)
        if value is not None:
            result[f"citation_{key}"] = str(value)
    for key, value in (record.get("metadata") or {}).items():
        if isinstance(value, (str, int, float, bool)):
            result[f"document_{key}"] = value
    return result


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if record.get("schema_version") != CHUNK_SCHEMA_VERSION:
                raise ValueError(f"Unsupported chunk schema at {path}:{line_number}")
            if not record.get("chunk_id") or not record.get("document_id") or not record.get("text", "").strip():
                raise ValueError(f"Missing chunk_id, document_id, or text at {path}:{line_number}")
            yield record


def _collection_name(strategy: str, input_sha256: str, model: SentenceEmbeddingModel) -> str:
    digest = hashlib.sha256(f"{model.model_name}@{model.revision}".encode()).hexdigest()[:10]
    return f"chunks_{strategy.lower()}_{input_sha256[:10]}_{digest}"


def _manifest_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def build_index(
    chunk_path: Path,
    *,
    model: SentenceEmbeddingModel,
    persist_dir: Path,
    cache_path: Path,
    manifest_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IndexResult:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not chunk_path.is_file():
        raise FileNotFoundError(chunk_path)

    input_sha = sha256_file(chunk_path)
    strategy = chunk_path.stem.removeprefix("chunks_")
    if strategy not in {"A", "B", "C"}:
        raise ValueError(f"Expected chunks_A/B/C.jsonl, got {chunk_path.name}")
    collection_name = _collection_name(strategy, input_sha, model)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "hnsw:space": "cosine",
            "model_name": model.model_name,
            "model_revision": model.revision,
            "embedding_dimension": model.dimension,
            "chunk_schema_version": CHUNK_SCHEMA_VERSION,
            "chunk_strategy": strategy,
            "input_sha256": input_sha,
        },
    )

    cache = EmbeddingCache(cache_path)
    chunk_count = cached_used = vectors_created = 0
    seen_ids: set[str] = set()
    try:
        records: list[dict[str, Any]] = []

        def write_batch(batch: list[dict[str, Any]]) -> None:
            nonlocal chunk_count, cached_used, vectors_created
            keys = [_cache_key(model.fingerprint, row["text"]) for row in batch]
            vectors = cache.get_many(keys)
            cached_used += sum(key in vectors for key in keys)
            missing_text: dict[str, str] = {}
            for key, row in zip(keys, batch):
                if key not in vectors:
                    missing_text.setdefault(key, row["text"])
            if missing_text:
                encoded = model.encode_documents(list(missing_text.values()), batch_size)
                if encoded.ndim != 2 or encoded.shape[1] != model.dimension:
                    raise ValueError("Embedding model returned an unexpected vector shape")
                created = {
                    key: np.asarray(vector, dtype=np.float32)
                    for key, vector in zip(missing_text, encoded)
                }
                cache.put_many(created)
                vectors.update(created)
                vectors_created += len(created)

            collection.upsert(
                ids=[row["chunk_id"] for row in batch],
                embeddings=[vectors[key].tolist() for key in keys],
                documents=[row["text"] for row in batch],
                metadatas=[_metadata(row) for row in batch],
            )
            chunk_count += len(batch)

        for record in _iter_records(chunk_path):
            chunk_id = record["chunk_id"]
            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id in {chunk_path}: {chunk_id}")
            seen_ids.add(chunk_id)
            records.append(record)
            if len(records) >= batch_size:
                write_batch(records)
                records = []
        if records:
            write_batch(records)
    finally:
        cache.close()

    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{collection_name}.manifest.json"
    manifest = {
        "schema_version": "embedding_index.manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input": {"path": _manifest_path(chunk_path), "sha256": input_sha,
                  "chunk_schema_version": CHUNK_SCHEMA_VERSION, "strategy": strategy},
        "embedding": {"backend": "sentence-transformers", "model": model.model_name,
                      "revision": model.revision, "dimension": model.dimension,
                      "normalization": "L2"},
        "vector_store": {"backend": "ChromaDB", "persist_path": _manifest_path(persist_dir),
                         "collection": collection_name, "distance": "cosine"},
        "embedding_cache": _manifest_path(cache_path),
        "batch_size": batch_size,
        "chunk_count": chunk_count,
        "cached_vectors_used": cached_used,
        "vectors_created": vectors_created,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return IndexResult(
        collection_name=collection_name,
        input_path=str(chunk_path),
        input_sha256=input_sha,
        chunk_count=chunk_count,
        embedding_dimension=model.dimension,
        cached_vectors_used=cached_used,
        vectors_created=vectors_created,
        manifest_path=str(manifest_path),
    )


def load_collection(collection_name: str, persist_dir: Path | None = None) -> Any:
    """Load a persisted Chroma collection for M4 without rebuilding its vectors."""
    path = persist_dir or (DATA_ROOT / "vector_store" / "chroma")
    client = chromadb.PersistentClient(path=str(path))
    return client.get_collection(name=collection_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed M2 chunks and persist a Chroma vector index (M3).")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES),
                        help="Comma-separated M2 chunk variants: A, B, C (default: all)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence Transformers model ID or local path")
    parser.add_argument("--device", help="Optional model device, e.g. cpu, mps, cuda")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--chunks-dir", type=Path, default=DATA_ROOT / "chunks")
    parser.add_argument("--persist-dir", type=Path, default=DATA_ROOT / "vector_store" / "chroma")
    parser.add_argument("--cache", type=Path, default=DATA_ROOT / "vector_store" / "embedding_cache.sqlite3")
    parser.add_argument("--manifest-dir", type=Path, default=DATA_ROOT / "vector_store" / "manifests")
    return parser


def main() -> int:
    args = _parser().parse_args()
    strategies = [part.strip().upper() for part in args.strategies.split(",") if part.strip()]
    if not strategies or any(item not in {"A", "B", "C"} for item in strategies):
        raise SystemExit("--strategies must contain A, B, and/or C")
    model = SentenceEmbeddingModel(args.model, args.device)
    for strategy in strategies:
        result = build_index(
            args.chunks_dir / f"chunks_{strategy}.jsonl",
            model=model,
            persist_dir=args.persist_dir,
            cache_path=args.cache,
            manifest_dir=args.manifest_dir,
            batch_size=args.batch_size,
        )
        print(json.dumps(result.__dict__, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
