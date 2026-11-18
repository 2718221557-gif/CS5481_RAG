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

M2 Chunking 使用说明见 [`scripts/m2_chunking/README_zh.md`](scripts/m2_chunking/README_zh.md)。
M3 Embedding 与向量库使用说明见 [`scripts/m3_embedding/README_zh.md`](scripts/m3_embedding/README_zh.md)。

## M3 完整交付

M3 的实现、接口、索引 manifests 和个人报告已纳入版本控制。完整的 M1/M2 数据与 M3 向量库、缓存通过 [M3 Release](https://github.com/2718221557-gif/CS5481_RAG/releases/tag/v0.3.0-m3) 提供。下载、校验、解压和读取向量的方法见 [M3 交付说明](documents/m3/DELIVERY.md)。
