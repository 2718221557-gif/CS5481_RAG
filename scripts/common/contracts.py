from __future__ import annotations

from typing import Any


DOCUMENT_SCHEMA_VERSION = "news_documents.v1"

DOCUMENT_REQUIRED_FIELDS = {
    "schema_version",
    "document_id",
    "source_name",
    "source_domain",
    "url",
    "title",
    "body_text",
    "published_at_utc",
    "language",
    "word_count",
    "content_sha256",
}

def validate_document(record: dict[str, Any]) -> None:
    missing = DOCUMENT_REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError(f"Document record is missing fields: {sorted(missing)}")
    if record["schema_version"] != DOCUMENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported document schema_version: {record['schema_version']!r}"
        )
    if not record["body_text"]:
        raise ValueError(f"Document {record['document_id']} has empty body_text")
