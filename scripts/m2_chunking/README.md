# M2 Chunking

Run from the repository root with `python -m scripts.m2_chunking.chunk_documents`.

The command reads `data/documents/documents.jsonl` and writes three alternatives under `data/chunks/`:

- `chunks_A.jsonl`: fixed character windows, default 1,200 characters with 150-character overlap; prefers sentence boundaries.
- `chunks_B.jsonl`: paragraph-first grouping up to 1,200 characters; oversized paragraphs are split at sentence boundaries, with character slicing as a final fallback.
- `chunks_C.jsonl`: sentence-first grouping up to 1,200 characters, with the article title prefixed to each chunk; exceptionally long sentences use character slicing.

Each output has a corresponding `.manifest.json` with input/output SHA-256, parameters, document and chunk counts, and word-count statistics. Defaults can be changed with `--chunk-chars` and `--overlap-chars` (the overlap applies to A only).

Each JSONL row uses `news_chunks.v1`: deterministic `chunk_id`, `document_id`, zero-based `chunk_index`, `strategy`, `text`, `word_count`, `citation` (URL, title, publication time, source), and available section/language metadata. M3 should embed `text` and retain the rest as vector metadata for citations.

## Comparison (current dataset)

| Strategy | Chunks | Mean words | Advantages | Trade-offs |
| --- | ---: | ---: | --- | --- |
| Fixed windows with overlap (A) | 17,015 | 169.11 | Simple and reproducible; overlap helps preserve context across boundaries. | Repeats text, increasing embedding and storage cost; may split paragraphs and sentences. |
| Paragraph-first (B) | 15,914 | 159.13 | Preserves paragraph context and avoids intentional overlap; lowest chunk count here. | Results depend on paragraph boundaries being present in source text; very long paragraphs need fallback splitting. |
| Sentence-first with title (C) | 16,432 | 166.94 | Keeps sentences intact in most cases and repeats the title to add topic context to each chunk. | Repeats title tokens; sentence segmentation can be imperfect, and long sentences need fallback splitting. |

These are corpus statistics, not retrieval-quality results. Strategy A's total chunk word count is larger than the source because overlap duplicates words; strategy C also repeats article titles. To claim that a strategy retrieves better, M3/M4 should build comparable indexes and evaluate the same QA/query set with the same embedding model and top-k, reporting metrics such as Recall@k or Hit@k, latency, and citation relevance.
