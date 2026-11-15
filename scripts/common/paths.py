from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"

RAW_ARTICLES = DATA_ROOT / "raw" / "guardian" / "articles.jsonl"
DOCUMENTS = DATA_ROOT / "documents" / "documents.jsonl"
DOCUMENTS_MANIFEST = DATA_ROOT / "manifests" / "documents.manifest.json"
STATE_DB = DATA_ROOT / "state" / "guardian.sqlite3"


def ensure_data_layout() -> None:
    """Create only the shared directories; stage scripts create their own files."""
    directories = [
        DATA_ROOT / "raw" / "guardian",
        DATA_ROOT / "documents",
        DATA_ROOT / "manifests",
        DATA_ROOT / "state",
        DATA_ROOT / "archive",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
