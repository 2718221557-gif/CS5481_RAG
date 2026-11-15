from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.common.contracts import validate_document
from scripts.common.jsonl import read_jsonl


def validate_unique(rows: list[dict[str, Any]], key: str, label: str) -> None:
    values = [record[key] for record in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate {key} values")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate handoff JSONL files before passing them downstream."
    )
    parser.add_argument("--documents", type=Path)
    args = parser.parse_args(argv)

    summary: dict[str, Any] = {}
    if args.documents:
        rows = list(read_jsonl(args.documents))
        for record in rows:
            validate_document(record)
        validate_unique(rows, "document_id", "documents")
        summary["documents"] = {
            "path": str(args.documents.resolve()),
            "record_count": len(rows),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
