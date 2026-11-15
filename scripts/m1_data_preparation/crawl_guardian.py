from __future__ import annotations

import argparse
import json
import logging
import os
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import requests

from scripts.common.paths import STATE_DB
from scripts.common.text import (
    content_hash,
    count_words,
    html_to_inline_text,
    html_to_text,
    looks_english,
)
from scripts.m1_data_preparation.state_store import (
    ArticleRecord,
    open_state,
    upsert_article,
    utc_now_iso,
)


LOGGER = logging.getLogger("m1.crawl")
ENDPOINT = "https://content.guardianapis.com/search"


class RateLimiter:
    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("--requests-per-minute must be positive")
        self.interval = 60.0 / requests_per_minute
        self.last_request = 0.0

    def wait(self) -> None:
        delay = self.interval - (time.monotonic() - self.last_request)
        if delay > 0:
            time.sleep(delay)
        self.last_request = time.monotonic()


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD, got {value!r}") from exc


def date_slices(start: date, end: date, slice_days: int) -> Iterable[tuple[date, date]]:
    if start > end:
        raise ValueError("--start-date must be on or before --end-date")
    if slice_days < 1:
        raise ValueError("--slice-days must be at least 1")
    current = start
    while current <= end:
        yield current, min(current + timedelta(days=slice_days - 1), end)
        current += timedelta(days=slice_days)


def parse_article(
    result: dict[str, Any],
    *,
    min_words: int,
    store_html: bool,
) -> ArticleRecord | None:
    fields = result.get("fields") or {}
    body_html = str(fields.get("body") or "")
    body_text = html_to_text(body_html)
    word_count = count_words(body_text)
    if word_count < min_words or not looks_english(body_text):
        return None

    title = html_to_inline_text(str(fields.get("headline") or result.get("webTitle") or ""))
    article_id = str(result.get("id") or "").strip()
    canonical_url = str(result.get("webUrl") or "").strip()
    published_at = str(result.get("webPublicationDate") or "").strip()
    if not title or not article_id or not canonical_url or not published_at:
        return None

    tags = [
        {
            "id": tag.get("id"),
            "type": tag.get("type"),
            "title": tag.get("webTitle"),
        }
        for tag in result.get("tags") or []
        if isinstance(tag, dict)
    ]
    metadata = {
        "pillar_id": result.get("pillarId"),
        "pillar_name": result.get("pillarName"),
        "short_url": fields.get("shortUrl"),
        "api_word_count": fields.get("wordcount"),
        "tags": tags,
    }
    fetched_at = utc_now_iso()
    return ArticleRecord(
        article_id=article_id,
        source_name="The Guardian",
        source_domain="theguardian.com",
        source_tier="established-major-outlet",
        canonical_url=canonical_url,
        api_url=str(result.get("apiUrl") or ""),
        title=title,
        standfirst=html_to_text(str(fields.get("standfirst") or "")),
        byline=html_to_inline_text(str(fields.get("byline") or "")),
        section_id=str(result.get("sectionId") or ""),
        section_name=str(result.get("sectionName") or ""),
        content_type=str(result.get("type") or ""),
        published_at_utc=published_at,
        fetched_at_utc=fetched_at,
        language=str(result.get("lang") or "en"),
        word_count=word_count,
        body_text=body_text,
        body_html=body_html if store_html else "",
        content_sha256=content_hash(title, body_text),
        metadata_json=json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        raw_json=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
    )


def request_page(
    session: requests.Session,
    params: dict[str, Any],
    limiter: RateLimiter,
    *,
    max_retries: int,
) -> dict[str, Any]:
    safe_params = {key: value for key, value in params.items() if key != "api-key"}
    for attempt in range(max_retries + 1):
        limiter.wait()
        try:
            response = session.get(ENDPOINT, params=params, timeout=(15, 90))
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Guardian request failed after retries: {type(exc).__name__}"
                ) from None
            delay = min(60, 2 ** attempt)
            LOGGER.warning(
                "Network error (%s); retrying in %ss",
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
            continue
        if response.status_code == 429 or response.status_code >= 500:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Guardian returned HTTP {response.status_code}; query={safe_params!r}"
                )
            try:
                delay = float(response.headers.get("Retry-After", ""))
            except ValueError:
                delay = min(60, 2 ** attempt)
            LOGGER.warning("HTTP %s; retrying in %.1fs", response.status_code, delay)
            time.sleep(delay)
            continue
        if not response.ok:
            raise RuntimeError(
                f"Guardian returned HTTP {response.status_code}: "
                f"{response.text[:300]}; query={safe_params!r}"
            )
        payload = response.json()
        api_response = payload.get("response") or {}
        if api_response.get("status") != "ok":
            raise RuntimeError(
                f"Guardian API error: {api_response.get('message', 'unknown error')}"
            )
        return api_response
    raise RuntimeError("Unreachable request state")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl Guardian full text into M1's resumable state database."
    )
    parser.add_argument("--start-date", type=parse_date, required=True)
    parser.add_argument("--end-date", type=parse_date, required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--sections", default="")
    parser.add_argument("--query", default="")
    parser.add_argument("--content-types", default="article")
    parser.add_argument("--min-words", type=int, default=120)
    parser.add_argument("--slice-days", type=int, default=1)
    parser.add_argument("--max-pages-per-slice", type=int, default=200)
    parser.add_argument("--max-articles", type=int, default=0)
    parser.add_argument("--requests-per-minute", type=float, default=50.0)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--state-db", type=Path, default=STATE_DB)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def run(args: argparse.Namespace) -> int:
    api_key = args.api_key or os.environ.get("GUARDIAN_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Set GUARDIAN_API_KEY or pass --api-key. "
            "Register at https://open-platform.theguardian.com/access/"
        )
    if args.min_words < 0 or args.max_articles < 0:
        raise ValueError("Word/count limits cannot be negative")
    if args.max_pages_per_slice < 1:
        raise ValueError("--max-pages-per-slice must be at least 1")

    allowed_types = {
        value.strip().lower()
        for value in args.content_types.split(",")
        if value.strip()
    }
    sections = "|".join(
        value.strip() for value in args.sections.split(",") if value.strip()
    )
    connection = open_state(args.state_db)
    run_id = uuid.uuid4().hex
    connection.execute(
        """
        INSERT INTO crawl_runs (
            run_id, started_at_utc, status, start_date, end_date, sections, query
        ) VALUES (?, ?, 'running', ?, ?, ?, ?)
        """,
        (
            run_id,
            utc_now_iso(),
            args.start_date.isoformat(),
            args.end_date.isoformat(),
            sections,
            args.query,
        ),
    )
    connection.commit()

    counters = {
        "api_queries": 0,
        "article_count": 0,
        "inserted_count": 0,
        "updated_count": 0,
        "skipped_count": 0,
    }
    limiter = RateLimiter(args.requests_per_minute)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "CS5481-Team-News-KB/2.0 (academic use)",
            "Accept": "application/json",
        }
    )

    def finish(status: str, error: str | None = None) -> None:
        connection.execute(
            """
            UPDATE crawl_runs SET
                finished_at_utc = ?, status = ?, api_queries = ?,
                article_count = ?, inserted_count = ?, updated_count = ?,
                skipped_count = ?, error = ?
            WHERE run_id = ?
            """,
            (
                None if status == "running" else utc_now_iso(),
                status,
                counters["api_queries"],
                counters["article_count"],
                counters["inserted_count"],
                counters["updated_count"],
                counters["skipped_count"],
                error,
                run_id,
            ),
        )
        connection.commit()

    try:
        stop = False
        for slice_start, slice_end in date_slices(
            args.start_date, args.end_date, args.slice_days
        ):
            if stop:
                break
            LOGGER.info("Crawling %s through %s", slice_start, slice_end)
            page = 1
            total_pages: int | None = None
            while True:
                params: dict[str, Any] = {
                    "api-key": api_key,
                    "from-date": slice_start.isoformat(),
                    "to-date": slice_end.isoformat(),
                    "use-date": "first-publication",
                    "lang": "en",
                    "order-by": "oldest",
                    "page": page,
                    "page-size": 50,
                    "show-fields": "headline,byline,body,wordcount,standfirst,shortUrl",
                    "show-tags": "keyword",
                }
                if sections:
                    params["section"] = sections
                if args.query:
                    params["q"] = args.query

                api_response = request_page(
                    session,
                    params,
                    limiter,
                    max_retries=args.max_retries,
                )
                counters["api_queries"] += 1
                results = api_response.get("results") or []
                if total_pages is None:
                    total_pages = int(api_response.get("pages") or 0)
                    LOGGER.info(
                        "Slice total=%s pages=%s",
                        api_response.get("total", "unknown"),
                        total_pages,
                    )
                if page == 1 and total_pages > args.max_pages_per_slice:
                    LOGGER.warning(
                        "Slice needs %s pages but cap is %s; reduce --slice-days",
                        total_pages,
                        args.max_pages_per_slice,
                    )

                for result in results:
                    if args.max_articles and counters["article_count"] >= args.max_articles:
                        stop = True
                        break
                    if (
                        allowed_types
                        and str(result.get("type") or "").lower() not in allowed_types
                    ):
                        counters["skipped_count"] += 1
                        continue
                    article = parse_article(
                        result,
                        min_words=args.min_words,
                        store_html=not args.no_html,
                    )
                    if article is None:
                        counters["skipped_count"] += 1
                        continue
                    outcome = upsert_article(connection, article, run_id)
                    if outcome == "duplicate":
                        counters["skipped_count"] += 1
                    else:
                        counters["article_count"] += 1
                        counters[f"{outcome}_count"] += 1
                connection.commit()
                if stop or not results or page >= total_pages:
                    break
                if page >= args.max_pages_per_slice:
                    break
                page += 1

        finish("completed")
        LOGGER.info(
            "Stored=%s inserted=%s updated=%s skipped=%s queries=%s",
            counters["article_count"],
            counters["inserted_count"],
            counters["updated_count"],
            counters["skipped_count"],
            counters["api_queries"],
        )
        LOGGER.info("Next: build M1 JSONL artifacts from %s", args.state_db)
        return 0
    except KeyboardInterrupt:
        finish("interrupted")
        LOGGER.warning("Interrupted; state has been committed")
        return 130
    except Exception as exc:
        finish("failed", str(exc))
        raise
    finally:
        session.close()
        connection.close()


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
