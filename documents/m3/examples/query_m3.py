"""Runnable M3 handoff example; not the full M4 retrieval/evaluation module."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a saved M3 collection with its pinned embedding model.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--strategy", choices=("A", "B", "C"), default="B")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--language", help="Optional document_language equality filter, e.g. en")
    parser.add_argument("--section", help="Optional document_section_id equality filter, e.g. world")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--manifest", type=Path, help="Explicit manifest path if multiple index versions exist")
    parser.add_argument("--persist-dir", type=Path, help="Override Chroma path, e.g. a temporary validation copy")
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("--query must not be blank")
    if args.top_k < 1:
        parser.error("--top-k must be positive")

    root = args.project_root.resolve()
    if args.manifest:
        manifest_path = args.manifest.resolve()
    else:
        candidates = sorted((root / "data/vector_store/manifests").glob(
            f"chunks_{args.strategy.lower()}_*.manifest.json"
        ))
        if len(candidates) != 1:
            parser.error(f"Expected one {args.strategy} manifest; found {len(candidates)}. Use --manifest.")
        manifest_path = candidates[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["input"]["strategy"] != args.strategy:
        parser.error("Manifest strategy does not match --strategy")
    embedding = manifest["embedding"]
    if not re.fullmatch(r"[0-9a-f]{40}", embedding["revision"]):
        parser.error("This example requires a pinned Hugging Face commit revision")
    if embedding["normalization"] != "L2" or manifest["vector_store"]["distance"] != "cosine":
        parser.error("This example expects the delivered L2/cosine index")

    persist_dir = args.persist_dir or (root / manifest["vector_store"]["persist_path"])
    if not (persist_dir / "chroma.sqlite3").is_file():
        parser.error(f"Missing saved database at {persist_dir}; extract the Release data archive first")
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(
        name=manifest["vector_store"]["collection"], embedding_function=None
    )
    if collection.count() != manifest["chunk_count"]:
        raise RuntimeError("Collection count does not match the supplied manifest")
    model = SentenceTransformer(
        embedding["model"], revision=embedding["revision"], device="cpu"
    )
    query_vectors = model.encode(
        [args.query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    if query_vectors.shape != (1, embedding["dimension"]):
        raise RuntimeError("Query embedding dimension does not match the saved index")

    filters = []
    if args.language:
        filters.append({"document_language": {"$eq": args.language}})
    if args.section:
        filters.append({"document_section_id": {"$eq": args.section}})
    query_args = {}
    if filters:
        query_args["where"] = filters[0] if len(filters) == 1 else {"$and": filters}
    result = collection.query(
        query_embeddings=query_vectors.tolist(),
        n_results=min(args.top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
        **query_args,
    )
    # Chroma query results have an outer list for queries, including for one query.
    print(json.dumps({
        "query": args.query,
        "collection": collection.name,
        "ids": result["ids"],
        "documents": result["documents"],
        "metadatas": result["metadatas"],
        "distances": result["distances"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
