from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scripts.common.jsonl import read_jsonl, sha256_file, write_json_atomic, write_jsonl_atomic
from scripts.common.paths import DATA_ROOT, DOCUMENTS

CHUNK_SCHEMA_VERSION = "news_chunks.v1"
DEFAULT_CHARS = 1200
DEFAULT_OVERLAP = 150


def _record(document: dict[str, Any], index: int, text: str, strategy: str) -> dict[str, Any]:
    text = text.strip()
    citation = {key: document.get(key, "") for key in ("url", "title", "published_at_utc", "source_name", "source_domain")}
    chunk_id = hashlib.sha256(f"{document['document_id']}:{strategy}:{index}:{text}".encode("utf-8")).hexdigest()
    return {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "document_id": document["document_id"],
        "chunk_index": index,
        "strategy": strategy,
        "text": text,
        "word_count": len(text.split()),
        "citation": citation,
        "metadata": {key: document.get(key) for key in ("section_id", "section_name", "language") if document.get(key) is not None},
    }


def fixed_chunks(document: dict[str, Any], chunk_chars: int = DEFAULT_CHARS, overlap_chars: int = DEFAULT_OVERLAP) -> list[dict[str, Any]]:
    if chunk_chars <= 0 or overlap_chars < 0 or overlap_chars >= chunk_chars:
        raise ValueError("Require chunk_chars > overlap_chars >= 0")
    text = re.sub(r"\s+", " ", document["body_text"]).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
            if boundary > start + chunk_chars // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(_record(document, len(chunks), piece, "fixed_overlap"))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def paragraph_chunks(document: dict[str, Any], chunk_chars: int = DEFAULT_CHARS) -> list[dict[str, Any]]:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document["body_text"]) if p.strip()]
    # M1 bodies may flatten paragraphs. Sentence boundaries provide a fallback while retaining paragraph-first grouping.
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_chars:
            units.append(paragraph)
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip())
    groups: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > chunk_chars:
            for offset in range(0, len(unit), chunk_chars):
                part = unit[offset:offset + chunk_chars].strip()
                if current:
                    groups.append(current); current = ""
                if part:
                    groups.append(part)
            continue
        candidate = f"{current}\n\n{unit}" if current else unit
        if current and len(candidate) > chunk_chars:
            groups.append(current); current = unit
        else:
            current = candidate
    if current:
        groups.append(current)
    return [_record(document, i, text, "paragraph") for i, text in enumerate(groups)]



def sentence_chunks(document: dict[str, Any], chunk_chars: int = DEFAULT_CHARS) -> list[dict[str, Any]]:
    """Pack whole sentences up to a character limit and prefix each chunk with its article title."""
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    title = (document.get("title") or "").strip()
    prefix = f"{title}\n\n" if title else ""
    budget = max(1, chunk_chars - len(prefix))
    body = re.sub(r"\s+", " ", document["body_text"]).strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()]
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= budget:
            units.append(sentence)
        else:
            units.extend(sentence[i:i + budget].strip() for i in range(0, len(sentence), budget))
    groups: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip() if current else unit
        if current and len(candidate) > budget:
            groups.append(current)
            current = unit
        else:
            current = candidate
    if current:
        groups.append(current)
    return [_record(document, i, prefix + text, "sentence_title") for i, text in enumerate(groups)]


def _stats(path: Path, rows: list[dict[str, Any]], strategy: str, params: dict[str, int]) -> dict[str, Any]:
    lengths = [row["word_count"] for row in rows]
    return {
        "schema_version": "chunks.manifest.v1", "strategy": strategy,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_path": "data/documents/documents.jsonl", "input_sha256": sha256_file(DOCUMENTS),
        "output_path": str(path.relative_to(DATA_ROOT.parent)).replace("\\", "/"),
        "output_sha256": sha256_file(path), "document_count": len({r["document_id"] for r in rows}),
        "chunk_count": len(rows), "parameters": params,
        "word_count": {"total": sum(lengths), "min": min(lengths, default=0), "max": max(lengths, default=0), "mean": round(sum(lengths)/len(lengths), 2) if lengths else 0},
    }


def build(strategy: str, output: Path, chunk_chars: int, overlap_chars: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunkers = {"fixed_overlap": fixed_chunks, "paragraph": paragraph_chunks, "sentence_title": sentence_chunks}
    chunker = chunkers[strategy]
    rows: list[dict[str, Any]] = []
    for doc in read_jsonl(DOCUMENTS):
        rows.extend(chunker(doc, chunk_chars, overlap_chars) if strategy == "fixed_overlap" else chunker(doc, chunk_chars))
    write_jsonl_atomic(output, rows)
    params = {"chunk_chars": chunk_chars, **({"overlap_chars": overlap_chars} if strategy == "fixed_overlap" else {})}
    return rows, _stats(output, rows, strategy, params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible news text chunks.")
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHARS)
    parser.add_argument("--overlap-chars", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    out_dir = DATA_ROOT / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for strategy, filename in (("fixed_overlap", "chunks_A.jsonl"), ("paragraph", "chunks_B.jsonl"), ("sentence_title", "chunks_C.jsonl")):
        path = out_dir / filename
        _, manifest = build(strategy, path, args.chunk_chars, args.overlap_chars)
        write_json_atomic(out_dir / filename.replace(".jsonl", ".manifest.json"), manifest)
        print(f"{filename}: {manifest['chunk_count']} chunks; {manifest['output_sha256']}")

if __name__ == "__main__":
    main()
