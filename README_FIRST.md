# M4 retrieval delivery

This delivery contains the implemented Dense, BM25 and RRF hybrid retrieval module, reviewed evaluation records, response schemas, source-verifiable examples and actual experiment outputs. It does not implement M5 answer generation.

## Verified results

24 reviewed answerable test questions from 8 science articles, plus 3 development questions from one Nepal/Tibet flood event. No reviewed unanswerable questions are present. Original QA and experiment files are retained unchanged.

| Method | Chunking | Hit@5 | Evidence Recall@5 | Budget Recall |
|---|---|---:|---:|---:|
| Dense | A | 87.5% | 87.5% | 87.5% |
| Dense | B | 95.8% | 95.8% | 91.7% |
| Dense | C | 95.8% | 95.8% | 87.5% |
| BM25 | A | 91.7% | 91.7% | 87.5% |
| BM25 | B | 95.8% | 95.8% | 91.7% |
| BM25 | C | 100.0% | 100.0% | 87.5% |
| Hybrid | A | 100.0% | 100.0% | 91.7% |
| Hybrid | B | 100.0% | 100.0% | 91.7% |
| Hybrid | C | 100.0% | 100.0% | 95.8% |

Settings: top_k=5, candidate_k=30, RRF constant=60, 1024-token context budget. Tokens use the embedding tokenizer as an M5 proxy. `pack()` stops at the first block that would exceed the budget. Evidence recall is macro-averaged coverage of annotated evidence units, not recall over every relevant corpus chunk. Complete@5 requires all annotated evidence units. Questions from the same article are correlated. Multi-fact questions do not necessarily require multiple chunks.

Dense was evaluated first; the BM25/hybrid comparison was added after observing Dense results. Hybrid/C is an evaluation-informed demonstration choice, not independently tested after selection. There is no statistically established superiority or generalization claim. 100% retrieval hit rate is not answer correctness.

## Contents

- `scripts/m4_retrieval/`: the tested retrieval implementation.
- `schemas/`: v1 and v2 response contracts. M5 should use Workshop v2.
- `data/evaluation/qa_reviewed.json`: all 27 reviewed records.
- `results/`: raw responses, summaries, QA snapshot, environment hashes and test record.
- `documents/m4/`: contract examples, final real-database integration record and Hybrid/C M5 example.
- `provenance/`: M3 index manifests and corrected question drafts.
- `M4_HANDOFF.ipynb`: clean initialization, integration verification, M5 example and optional experiment reproduction. No question-authoring cells.
- `requirements-m4.txt`: observed package versions on the tested Python 3.11 Windows environment; a clean cross-platform installation has not been tested.
- `verify_delivery.py`: offline artifact and unit verification; it does not need Chroma/model downloads.

## External dependencies

The repository upload deliberately excludes runtime databases, model caches and the original 536 MB archive. Obtain `CS5481_RAG_stage3_data.tar.gz` from the [M3 Release](https://github.com/2718221557-gif/CS5481_RAG/releases/tag/v0.3.0-m3). Expected SHA256: `89776eabda1e3dd4b1db8f096b970c0fb156357b40017dc8f78c49178f3ff586`. Extract with a trusted archive tool so `M4_DATA_ROOT/data/documents/documents.jsonl`, `data/chunks/`, `data/vector_store/manifests/` and `data/vector_store/chroma/` exist. Do not copy a live database while another process is writing it.

The original machine uses `D:/CS5481_work/m3_data`, runtime database `D:/CS5481_work/runtime_chroma_v030`, and cache `D:/CS5481_work/model_cache`. The runtime directory name is historical: the actual tested Chroma version is 1.5.9, not 0.3.0.

Model: `sentence-transformers/all-MiniLM-L6-v2`; revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; dimension 384; L2-normalized embeddings; cosine distance. A compatible snapshot cache is required for offline startup. Set `M4_ALLOW_MODEL_DOWNLOAD=1` before starting Jupyter to explicitly allow downloading this pinned snapshot when absent.

## Start on the original machine

In Anaconda Prompt, activate `cs5481-m4`, change directory to THIS delivery folder, and run `jupyter lab`. Open `M4_HANDOFF.ipynb`, choose Python (cs5481-m4), and run Initialization 1 through 4 in order. Then run Integration verification and the M5 retrieval example. Do not run all cells: evaluation cells at the end intentionally rerun experiments.

## Start on another machine

1. Use Python 3.11. Install `requirements-m4.txt` in an isolated environment. Resolve unavailable platform wheels explicitly; do not silently change versions when reproducing recorded results.
2. Supply M3 data and the pinned model as described above.
3. Set `M4_WORK_BASE` to a writable working directory, `M4_DATA_ROOT` to the extracted M3 root, and optionally `M4_RUN_DB` to a working Chroma copy. If the runtime path does not exist, Initialization 2 copies the M3 database before opening it. Existing databases without the expected ready marker are rejected intentionally.
4. Register the kernel with `python -m ipykernel install --user --name cs5481-m4 --display-name "Python (cs5481-m4)"`.
5. Start Jupyter from THIS delivery folder and execute initialization in order. All code and comments in the handoff notebook are English.

Original run metadata includes original absolute paths for provenance. Those fields are historical; use the supplied folder-relative result files to inspect the delivery.

## M5 interface

Use `w.retrieve(query, strategy="C", mode="hybrid", top_k=5, candidate_k=30, rrf_constant=60)` after notebook initialization. `w` is a Workshop instance, not the v1 Retriever directly. Use `w.pack(response["results"], token_budget=1024)` for the demonstrated context selection. Recompute the generation budget with the actual M5 tokenizer.

Read `status`, `results[].text`, `results[].chunk_id`, `results[].document_id`, and `results[].citation`. `OK` means candidates were retrieved, not that the question is answerable. M5 must check support for every claim, cite only evidence actually supplied to it, and refuse unsupported answers. `ERROR` is a system failure, not an unanswerable question. `NO_EVIDENCE` means no candidates under the active request; the contract example uses an impossible language filter and does not measure semantic refusal.

`documents/m4/m5_hybrid_C_example.json` contains a real retrieval response plus context and selected chunk IDs. Scores have mode-dependent meanings; RRF combines ranks, not incomparable raw scores. No date index was activated for this delivery.

## Verification and remaining scope

The saved record shows 10 unit tests passed, response contract checks passed, integration checks passed for all 3 modes, and Hybrid/C citation round-trip passed. Date checks are false (not implemented in the active index). The package verifier additionally checks saved responses and metrics without rerunning retrieval.

Not completed: curated unanswerable-question evaluation, M5 answer correctness/refusal evaluation, independent post-selection testing, cross-encoder reranking, and a clean installation test on another machine. These are not represented as completed work. The legacy `run_grid` requires larger dev/unanswerable sets; use the supplied fixed-evaluation notebook cells for this delivery's 3-dev/24-test protocol.
