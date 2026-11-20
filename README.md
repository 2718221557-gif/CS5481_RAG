# CS5481 News RAG

当前已完成 M1 新闻数据采集和 M2 Chunking，并提供 M3 embedding 与持久化向量库构建模块。各阶段通过 `documents/interfaces/接口文档.md` 中的数据契约衔接。

## 目录

```text
documents/              项目要求、分工和接口文档
data/raw/               原始文章
data/documents/         标准文档 JSONL
data/manifests/         数据清单和校验信息
data/state/             M1 状态数据库
scripts/common/         共用工具
scripts/m1_data_preparation/
scripts/m2_chunking/
scripts/m3_embedding/
tests/
```

## 运行

在 Anaconda Prompt 中：

```bat
cd /d "~\project"         打开项目根目录
python -m pip install -r requirements.txt
set GUARDIAN_API_KEY=YOUR_GUARDIAN_KEY
set START_DATE=YYYY-MM-DD
set END_DATE=YYYY-MM-DD
python -m scripts.m1_data_preparation.crawl_guardian --start-date "%START_DATE%" --end-date "%END_DATE%" --sections world,politics,business,technology,science --requests-per-minute 50
python -m scripts.m1_data_preparation.build_documents
```

重复执行即可更新已有文章并补充新日期范围，不会重复写入。

M1 使用说明见
[`scripts/m1_data_preparation/README.md`](scripts/m1_data_preparation/README.md)。

## M2 文本切分

M2 将 M1 的标准文档 `data/documents/documents.jsonl` 切分成适合向量化和检索的文本块。在仓库根目录运行以下命令，即可在 `data/chunks/` 下生成三种方案：

```bash
python -m scripts.m2_chunking.chunk_documents
```

- `chunks_A.jsonl`：固定长度切分并带重叠（默认每块 1,200 字符，重叠 150 字符）。
- `chunks_B.jsonl`：优先按段落组合，超长段落再按句子或字符切分。
- `chunks_C.jsonl`：优先按句子组合，并在每块前附上文章标题。

每套结果都有对应的 manifest，记录参数、数量和 SHA-256。M3 应使用每条记录的 `text` 生成 embedding，并保留引用和 metadata 字段。详细说明、统计和方法比较见 [M2 英文说明](scripts/m2_chunking/README.md) 与 [M2 中文说明](scripts/m2_chunking/README_zh.md)。

M3 Embedding 与向量库使用说明见 [`scripts/m3_embedding/README_zh.md`](scripts/m3_embedding/README_zh.md)。

## M3 完整交付

M3 的实现、接口、索引 manifests 和个人报告已纳入版本控制。完整的 M1/M2 数据与 M3 向量库、缓存通过 [M3 Release](https://github.com/2718221557-gif/CS5481_RAG/releases/tag/v0.3.0-m3) 提供。下载、校验、解压和读取向量的方法见 [M3 交付说明](documents/m3/DELIVERY.md)。

负责 M4 检索的同学请先看 [M3 → M4 对接文档](documents/m3/M4_HANDOFF.md)：含实际数据格式、函数接口、集合配置、查询示例和验证结果。


## M4 Retrieval and Evaluation Delivery

M4 now includes Dense, BM25 and RRF hybrid retrieval with source-verifiable citations, A/B/C chunking comparisons, reviewed QA, and real response examples. Start with [M4 delivery and limitations](README_FIRST.md), then open [M4 handoff notebook](M4_HANDOFF.ipynb) from the repository root. Use the isolated M4 environment in [requirements-m4.txt](requirements-m4.txt); do not assume the M3 dependency versions match the tested M4 runtime.

- [Reviewed QA: 3 dev + 24 answerable test questions](data/evaluation/qa_reviewed.json)
- [Nine-configuration comparison](results/dense_fixed_20261120_075738_836725/lexical_comparison_1a171826/results_table.md)
- [Final integration verification](documents/m4/final_integration_checks.json)
- [Real Hybrid/C input example for M5](documents/m4/m5_hybrid_C_example.json)
- [M4 to M5 contract](documents/m4/contract_handoff_v2.md)

On the 24-question science subset, Hybrid/C achieves Hit@5 of 100% and budgeted evidence recall of 95.8% with a 1024 embedding-tokenizer budget. These are retrieval metrics, not answer correctness. The configuration was selected after observing this evaluation; no independent post-selection result is claimed. Unanswerable-question evaluation and M5 generation/refusal evaluation remain outstanding. Large M3 data and model caches are external dependencies, not included in this commit.

To verify saved artifacts and unit tests without a model download, run `python verify_delivery.py` in the M4 environment. The handoff notebook contains separate cells for real database integration checks and optional experiment reproduction.
