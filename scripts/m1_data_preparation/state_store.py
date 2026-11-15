from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    article_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_domain TEXT NOT NULL,
    source_tier TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    api_url TEXT,
    title TEXT NOT NULL,
    standfirst TEXT NOT NULL DEFAULT '',
    byline TEXT NOT NULL DEFAULT '',
    section_id TEXT NOT NULL DEFAULT '',
    section_name TEXT NOT NULL DEFAULT '',
    content_type TEXT NOT NULL DEFAULT '',
    published_at_utc TEXT NOT NULL,
    fetched_at_utc TEXT NOT NULL,
    first_seen_at_utc TEXT NOT NULL,
    language TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    body_text TEXT NOT NULL,
    body_html TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    raw_json TEXT,
    crawl_run_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_articles_published
    ON articles(published_at_utc);
CREATE INDEX IF NOT EXISTS idx_articles_section
    ON articles(section_id);
CREATE INDEX IF NOT EXISTS idx_articles_content_hash
    ON articles(content_sha256);
CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_canonical_url
    ON articles(canonical_url);

CREATE TABLE IF NOT EXISTS crawl_runs (
    run_id TEXT PRIMARY KEY,
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    status TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date TEXT NOT NULL,
    sections TEXT NOT NULL,
    query TEXT NOT NULL,
    api_queries INTEGER NOT NULL DEFAULT 0,
    article_count INTEGER NOT NULL DEFAULT 0,
    inserted_count INTEGER NOT NULL DEFAULT 0,
    updated_count INTEGER NOT NULL DEFAULT 0,
    skipped_count INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
"""


@dataclass(frozen=True)
class ArticleRecord:
    article_id: str
    source_name: str
    source_domain: str
    source_tier: str
    canonical_url: str
    api_url: str
    title: str
    standfirst: str
    byline: str
    section_id: str
    section_name: str
    content_type: str
    published_at_utc: str
    fetched_at_utc: str
    language: str
    word_count: int
    body_text: str
    body_html: str
    content_sha256: str
    metadata_json: str
    raw_json: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def open_state(path: Path) -> sqlite3.Connection:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=60000")
    connection.executescript(STATE_SCHEMA)
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(articles)")
    }
    if "raw_json" not in columns:
        connection.execute("ALTER TABLE articles ADD COLUMN raw_json TEXT")
    connection.commit()
    return connection


def upsert_article(
    connection: sqlite3.Connection,
    article: ArticleRecord,
    crawl_run_id: str,
) -> str:
    existing = connection.execute(
        "SELECT 1 FROM articles WHERE article_id = ?", (article.article_id,)
    ).fetchone()
    duplicate_url = connection.execute(
        "SELECT article_id FROM articles "
        "WHERE canonical_url = ? AND article_id <> ? LIMIT 1",
        (article.canonical_url, article.article_id),
    ).fetchone()
    if duplicate_url:
        return "duplicate"
    duplicate_content = connection.execute(
        "SELECT article_id FROM articles "
        "WHERE content_sha256 = ? AND article_id <> ? LIMIT 1",
        (article.content_sha256, article.article_id),
    ).fetchone()
    if duplicate_content:
        return "duplicate"

    connection.execute(
        """
        INSERT INTO articles (
            article_id, source_name, source_domain, source_tier,
            canonical_url, api_url, title, standfirst, byline,
            section_id, section_name, content_type, published_at_utc,
            fetched_at_utc, first_seen_at_utc, language, word_count,
            body_text, body_html, content_sha256, metadata_json,
            raw_json, crawl_run_id
        ) VALUES (
            :article_id, :source_name, :source_domain, :source_tier,
            :canonical_url, :api_url, :title, :standfirst, :byline,
            :section_id, :section_name, :content_type, :published_at_utc,
            :fetched_at_utc, :fetched_at_utc, :language, :word_count,
            :body_text, :body_html, :content_sha256, :metadata_json,
            :raw_json, :crawl_run_id
        )
        ON CONFLICT(article_id) DO UPDATE SET
            canonical_url = excluded.canonical_url,
            api_url = excluded.api_url,
            title = excluded.title,
            standfirst = excluded.standfirst,
            byline = excluded.byline,
            section_id = excluded.section_id,
            section_name = excluded.section_name,
            content_type = excluded.content_type,
            published_at_utc = excluded.published_at_utc,
            fetched_at_utc = excluded.fetched_at_utc,
            language = excluded.language,
            word_count = excluded.word_count,
            body_text = excluded.body_text,
            body_html = excluded.body_html,
            content_sha256 = excluded.content_sha256,
            metadata_json = excluded.metadata_json,
            raw_json = excluded.raw_json,
            crawl_run_id = excluded.crawl_run_id
        """,
        {**article.__dict__, "crawl_run_id": crawl_run_id},
    )
    return "updated" if existing else "inserted"


def article_rows(
    connection: sqlite3.Connection,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    params: list[Any] = []
    if start_date:
        clauses.append("substr(published_at_utc, 1, 10) >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("substr(published_at_utc, 1, 10) <= ?")
        params.append(end_date)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return connection.execute(
        "SELECT * FROM articles" + where + " ORDER BY published_at_utc, article_id",
        params,
    ).fetchall()


def parse_metadata(row: sqlite3.Row) -> dict[str, Any]:
    try:
        metadata = json.loads(row["metadata_json"] or "{}")
    except json.JSONDecodeError:
        metadata = {}
    return metadata if isinstance(metadata, dict) else {}
