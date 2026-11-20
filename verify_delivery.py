"""Verify the delivered artifacts without loading models or vector databases."""
from pathlib import Path
import sys
import json
import math
from statistics import mean

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from m4_retrieval.workshop import validate, digest, evidence_coverage
from m4_retrieval.workshop_checks import run_checks

schema = json.loads((ROOT / "schemas/retrieval_response.v2.schema.json").read_text(encoding="utf-8"))
unit = run_checks(schema)
qa_path = ROOT / "data/evaluation/qa_reviewed.json"
qa = json.loads(qa_path.read_text(encoding="utf-8"))
assert len(qa) == 27 and all(r["reviewed"] for r in qa)
assert len({r["question_id"] for r in qa}) == len(qa)
assert sum(r["split"] == "dev" for r in qa) == 3
assert sum(r["split"] == "test" for r in qa) == 24
questions = {q["question_id"]: q for q in qa}
dev_docs = {e["document_id"] for q in qa if q["split"] == "dev" for e in q["gold_evidence"]}
test_docs = {e["document_id"] for q in qa if q["split"] == "test" for e in q["gold_evidence"]}
assert not dev_docs.intersection(test_docs)
response_count = 0
observed_groups = {}
for metadata_path in sorted((ROOT / "results").rglob("run_metadata.json")):
    folder = metadata_path.parent
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert meta["status"] == "COMPLETE"
    assert meta["qa_sha256"] == digest(qa_path)
    assert meta["schema_sha256"] == digest(ROOT / "schemas/retrieval_response.v2.schema.json")
    assert meta["workshop_source_sha256"] == digest(ROOT / "scripts/m4_retrieval/workshop.py")
    for strategy, expected in meta["index_hashes"].items():
        files = list((ROOT / "provenance/index_manifests").glob(f"chunks_{strategy.lower()}_*.manifest.json"))
        assert len(files) == 1 and digest(files[0]) == expected
    groups = {}
    with (folder / "responses.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            response = record["retrieval"]
            validate(response, schema)
            assert response["status"] != "ERROR"
            strategy = record["strategy"]
            assert response["index"]["strategy"] == strategy
            mode = response["config"]["mode"]
            assert response["config"]["top_k"] == 5
            question = questions[record["question_id"]]
            assert question["question"] == response["query"]
            raw = evidence_coverage(question, response["results"], strategy)
            selected_ids = set(record["selected_chunk_ids"])
            assert selected_ids <= {r["chunk_id"] for r in response["results"]}
            selected = [r for r in response["results"] if r["chunk_id"] in selected_ids]
            budget = evidence_coverage(question, selected, strategy)
            assert raw == record["raw_metrics"] and budget == record["budget_metrics"]
            assert record["context_tokens"] <= meta["token_budget"]
            groups.setdefault((mode, strategy), []).append(record)
            response_count += 1
    for key, records in groups.items():
        assert len(records) == 24 and len({r["question_id"] for r in records}) == 24
        observed_groups[key] = records
    rows = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    for row in rows:
        key = (row["mode"], row["strategy"])
        records = groups.get(key)
        if records is None:
            continue
        assert row["error_count"] == 0
        for prefix, field in (("raw", "raw_metrics"), ("budget", "budget_metrics")):
            for metric in ("hit", "recall", "complete"):
                assert math.isclose(row[f"{prefix}_{metric}"], mean(r[field][metric] for r in records))
assert response_count == 216 and len(observed_groups) == 9
for status in ("OK", "NO_EVIDENCE", "ERROR"):
    example = json.loads((ROOT / f"documents/m4/contract_examples_v2/{status}.json").read_text(encoding="utf-8"))
    validate(example, schema)
    assert example["status"] == status
integration = json.loads((ROOT / "documents/m4/final_integration_checks.json").read_text(encoding="utf-8"))
assert integration["integration"]["status"] == "PASS"
assert integration["integration"]["checked_modes"] == 3
assert integration["hybrid_C_citation_roundtrip"] == "PASS"
demo = json.loads((ROOT / "documents/m4/m5_hybrid_C_example.json").read_text(encoding="utf-8"))
validate(demo["retrieval"], schema)
assert demo["retrieval"]["index"]["strategy"] == "C"
assert demo["retrieval"]["config"]["mode"] == "hybrid"
nb = json.loads((ROOT / "M4_HANDOFF.ipynb").read_text(encoding="utf-8"))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        source = "".join(c["source"])
        assert not any("\u4e00" <= char <= "\u9fff" for char in source)
        compile(source, "M4_HANDOFF.ipynb", "exec")
print("DELIVERY VERIFIED: 27 QA records; 216 responses; 9 configurations; 10 unit tests; schemas and notebook syntax.")
print("Saved integration evidence checked; live database/model loading was not rerun by this verifier.")
