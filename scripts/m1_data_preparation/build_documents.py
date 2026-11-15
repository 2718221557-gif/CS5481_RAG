from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from scripts.common.contracts import DOCUMENT_SCHEMA_VERSION, validate_document
from scripts.common.jsonl import sha256_file, write_json_atomic, write_jsonl_atomic
from scripts.common.paths import (
    DOCUMENTS,
    DOCUMENTS_MANIFEST,
    PROJECT_ROOT,
    RAW_ARTICLES,
    STATE_DB,
)
from scripts.common.text import count_words
from scripts.m1_data_preparation.state_store import (
    article_rows,
    open_state,
    parse_metadata,
    utc_now_iso,
)


LOGGER = logging.getLogger("m1.documents")
RAW_SCHEMA_VERSION = "raw_news_articles.v1"


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def document_record(row: Any) -> dict[str, Any]:
    metadata = parse_metadata(row)
    record = {
        "schema_version": DOCUMENT_SCHEMA_VERSION,
        "document_id": row["article_id"],
        "source_name": row["source_name"],
        "source_domain": row["source_domain"],
        "source_tier": row["source_tier"],
        "url": row["canonical_url"],
        "api_url": row["api_url"],
        "title": row["title"],
        "standfirst": row["standfirst"],
        "byline": row["byline"],
        "section_id": row["section_id"],
        "section_name": row["section_name"],
        "content_type": row["content_type"],
        "published_at_utc": row["published_at_utc"],
        "first_seen_at_utc": row["first_seen_at_utc"],
        "fetched_at_utc": row["fetched_at_utc"],
        "language": row["language"],
        "word_count": count_words(row["body_text"]),
        "body_text": row["body_text"],
        "body_html": row["body_html"],
        "content_sha256": row["content_sha256"],
        "metadata": metadata,
    }
    validate_document(record)
    return record


def raw_record(row: Any) -> dict[str, Any]:
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "record_id": row["article_id"],
        "source_name": row["source_name"],
        "source_domain": row["source_domain"],
        "api_url": row["api_url"],
        "fetched_at_utc": row["fetched_at_utc"],
        "payload": {
            "title": row["title"],
            "standfirst": row["standfirst"],
            "byline": row["byline"],
            "section_id": row["section_id"],
            "section_name": row["section_name"],
            "content_type": row["content_type"],
            "published_at_utc": row["published_at_utc"],
            "body_html": row["body_html"],
            "metadata": parse_metadata(row),
            "source_api_json": (
                json.loads(row["raw_json"]) if row["raw_json"] else None
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze M1 state into raw and normalized JSONL contracts."
    )
    parser.add_argument("--state-db", type=Path, default=STATE_DB)
    parser.add_argument("--raw-output", type=Path, default=RAW_ARTICLES)
    parser.add_argument("--documents-output", type=Path, default=DOCUMENTS)
    parser.add_argument("--manifest", type=Path, default=DOCUMENTS_MANIFEST)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def run(args: argparse.Namespace) -> int:
    connection = open_state(args.state_db)
    try:
        rows = article_rows(
            connection,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        crawl_runs = [
            dict(row)
            for row in connection.execute(
                """
                SELECT run_id, started_at_utc, finished_at_utc, status,
                       start_date, end_date, sections, query,
                       api_queries, article_count, inserted_count,
                       updated_count, skipped_count, error
                FROM crawl_runs
                ORDER BY started_at_utc, run_id
                """
            ).fetchall()
        ]
    finally:
        connection.close()
    if not rows:
        raise RuntimeError("No articles matched; crawl data before building documents")

    documents = [document_record(row) for row in rows]
    raw_records = [raw_record(row) for row in rows]
    document_count = write_jsonl_atomic(
        args.documents_output,
        documents,
        sort_key="document_id",
    )
    raw_count = write_jsonl_atomic(
        args.raw_output,
        raw_records,
        sort_key="record_id",
    )

    dates = [record["published_at_utc"][:10] for record in documents]
    sections = Counter(record["section_id"] or "unknown" for record in documents)
    raw_manifest = args.manifest.with_name("raw_articles.manifest.json")
    manifest = {
        "schema_version": "documents.manifest.v1",
        "created_at_utc": utc_now_iso(),
        "documents": {
            "path": relative_to_root(args.documents_output),
            "sha256": sha256_file(args.documents_output),
            "record_count": document_count,
        },
        "raw": {
            "path": relative_to_root(args.raw_output),
            "sha256": sha256_file(args.raw_output),
            "record_count": raw_count,
        },
        "date_min": min(dates),
        "date_max": max(dates),
        "total_words": sum(int(record["word_count"]) for record in documents),
        "sections": dict(sections.most_common()),
        "crawl_runs": crawl_runs,
        "source": {
            "name": "The Guardian",
            "domain": "theguardian.com",
            "trust_tier": "established-major-outlet",
        },
    }
    write_json_atomic(args.manifest, manifest)
    write_json_atomic(
        raw_manifest,
        {
            "schema_version": "raw_articles.manifest.v1",
            "created_at_utc": manifest["created_at_utc"],
            "path": relative_to_root(args.raw_output),
            "sha256": manifest["raw"]["sha256"],
            "record_count": raw_count,
        },
    )
    LOGGER.info(
        "Built %s documents and %s raw records; manifest=%s",
        document_count,
        raw_count,
        args.manifest.resolve(),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return run(args)
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
