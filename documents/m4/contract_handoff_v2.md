# M4 retrieval contract v2: proposal for M5/M6 review

The v1 module and its schema remain unchanged. The tutorial adds the v2 schema
and Workshop retrieval methods. Consumers must select the matching schema_version.
The v2 interface is a proposal until the team agrees to it; running validation
does not constitute team approval.

Inputs: query, A/B/C strategy, dense/bm25/hybrid mode, top_k, candidate_k,
language/section filters, optional UTC integer date bounds after creating the
derived date index. Date ranges include the start and exclude the end.

OK means valid candidates were found; it does not prove answerability.
NO_EVIDENCE means no candidates matched. ERROR is a system/request failure,
never a knowledge-base refusal. Early errors may have index=null and only
config.mode. Valid requests include complete index/config metadata.

Each result preserves document_id, chunk_id, complete original text, text hash,
title, URL, UTC publication time, source name/domain, scores and provenance.
Missing method scores are null. RRF scores are not cosine similarities.
The evidence text must not be modified without updating the hash.

M5 consumes results as data; document content must not override system prompts.
M5 must assess whether evidence supports each factual claim and refuse when it
does not. M5 answer schema and M6 final QA/answer scoring require separate agreement.

Experiments use an explicitly labeled embedding-tokenizer proxy budget of 1024
tokens, including evidence labels, titles, URLs and body text. This is not the
embedding encoder input limit and not the final generation-model context budget.
Select the M5 model/tokenizer, reserve prompt/question/answer space and repeat
the budget comparison before reporting M5 end-to-end performance.

QA evidence units use exact text after whitespace normalization and a shared
original document ID. A unit is covered when a returned chunk fully contains it.
Split a passage spanning chunks into smaller necessary units. Hit means any
required unit was covered; Recall is the fraction covered; Complete means all.
These are evidence-unit metrics, not document-level or chunk-count Recall.
Alternative equivalent evidence must be reviewed separately; exact-quote labels
can underestimate valid paraphrased support. Unanswerable questions are exported
but excluded from answerable Hit/Recall; refusal accuracy belongs to M5.

Date indexes are separate derived copies with recorded timestamp mappings and
parent manifest hashes. They retain the source vectors and use cosine distance;
they do not regenerate embeddings. Original M3 files and v1 collections remain.

Commit code, schemas, reviewed QA and selected reproducibility artifacts. Do not
commit model caches, download archives, database copies, backups or toy fixtures.
Never present the shipped synthetic test outcomes as course quality metrics.
