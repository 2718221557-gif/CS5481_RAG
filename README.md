# CS5481 News RAG

当前完成 M1 新闻数据采集。各阶段通过 `documents/interfaces/接口文档.md` 中的
数据契约衔接。

## 目录

```text
documents/              项目要求、分工和接口文档
data/raw/               原始文章
data/documents/         标准文档 JSONL
data/manifests/         数据清单和校验信息
data/state/             M1 状态数据库
scripts/common/         共用工具
scripts/m1_data_preparation/
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
