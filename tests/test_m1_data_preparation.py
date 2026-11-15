from __future__ import annotations

from datetime import date

from scripts.common.contracts import validate_document
from scripts.common.jsonl import read_jsonl
from scripts.common.text import content_hash, html_to_text
from scripts.m1_data_preparation.build_documents import main as build_documents_main
from scripts.m1_data_preparation.crawl_guardian import date_slices
from scripts.m1_data_preparation.state_store import (
    ArticleRecord,
    open_state,
    upsert_article,
)


def test_html_to_text_preserves_full_article_text() -> None:
    text = html_to_text("<p>First <b>paragraph</b>.</p><p>Second paragraph.</p>")
    assert "First paragraph." in text
    assert "Second paragraph." in text
    assert "\n\n" in text


def test_date_slices_are_inclusive() -> None:
    assert list(date_slices(date(2026, 8, 1), date(2026, 8, 5), 2)) == [
        (date(2026, 8, 1), date(2026, 8, 2)),
        (date(2026, 8, 3), date(2026, 8, 4)),
        (date(2026, 8, 5), date(2026, 8, 5)),
    ]


def test_state_deduplicates_by_article_id(tmp_path) -> None:
    connection = open_state(tmp_path / "state.sqlite3")
    record = make_record()
    assert upsert_article(connection, record, "run-1") == "inserted"
    assert upsert_article(connection, record, "run-2") == "updated"
    count = connection.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    assert count == 1
    connection.close()


def test_m1_builds_raw_and_documents_outputs(tmp_path) -> None:
    state_path = tmp_path / "state.sqlite3"
    connection = open_state(state_path)
    assert upsert_article(connection, make_record(), "run") == "inserted"
    connection.commit()
    connection.close()

    documents_path = tmp_path / "documents.jsonl"
    raw_path = tmp_path / "articles.jsonl"
    manifest_path = tmp_path / "documents.manifest.json"
    code = build_documents_main(
        [
            "--state-db",
            str(state_path),
            "--documents-output",
            str(documents_path),
            "--raw-output",
            str(raw_path),
            "--manifest",
            str(manifest_path),
        ]
    )
    assert code == 0
    documents = list(read_jsonl(documents_path))
    raw = list(read_jsonl(raw_path))
    assert len(documents) == 1
    assert len(raw) == 1
    validate_document(documents[0])
    assert documents[0]["document_id"] == "world/2026/aug/01/example"
    assert manifest_path.exists()


def make_record() -> ArticleRecord:
    body = "This is a complete English article. " * 20
    return ArticleRecord(
        article_id="world/2026/aug/01/example",
        source_name="The Guardian",
        source_domain="theguardian.com",
        source_tier="established-major-outlet",
        canonical_url="https://www.theguardian.com/world/2026/aug/01/example",
        api_url="https://content.guardianapis.com/world/2026/aug/01/example",
        title="Example title",
        standfirst="Summary",
        byline="Reporter",
        section_id="world",
        section_name="World news",
        content_type="article",
        published_at_utc="2026-08-01T10:00:00Z",
        fetched_at_utc="2026-08-02T00:00:00Z",
        language="en",
        word_count=len(body.split()),
        body_text=body,
        body_html="<p>Complete English article.</p>",
        content_sha256=content_hash("Example title", body),
        metadata_json="{}",
        raw_json='{"id":"world/2026/aug/01/example"}',
    )
