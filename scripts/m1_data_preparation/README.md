# M1 数据采集

M1 负责 The Guardian 新闻全文采集、清洗、去重和标准文档生成。

## Anaconda Prompt 运行

先进入本机的项目根目录，即包含 `requirements.txt` 的目录。

```bat
python -m pip install -r requirements.txt
set GUARDIAN_API_KEY=YOUR_GUARDIAN_KEY
```

设置日期并抓取：

```bat
set START_DATE=YYYY-MM-DD
set END_DATE=YYYY-MM-DD
python -m scripts.m1_data_preparation.crawl_guardian --start-date "%START_DATE%" --end-date "%END_DATE%" --sections world,politics,business,technology,science --requests-per-minute 50
```

生成最终文件：

```bat
python -m scripts.m1_data_preparation.build_documents
```

## 重跑

重复执行上面的抓取和 `build_documents` 即可。SQLite 状态库会保留已有文章，并按
文章 ID、URL 和正文哈希去重；新日期只追加新文章，`build_documents` 会重新生成
完整的 JSONL。

## 输出

| 文件 | 作用 | 使用方式 |
| --- | --- | --- |
| `data/raw/guardian/articles.jsonl` | 保存来源正文 HTML、标题和元数据的原始版本 | 用于审计正文提取和重新清洗，一般不作为后续检索的直接输入 |
| `data/documents/documents.jsonl` | 清洗、去重后的标准文档，每行一篇完整英文新闻 | 后续阶段的主要输入；使用 `document_id` 关联文章，用 `body_text` 读取全文 |
| `data/manifests/documents.manifest.json` | 记录文档数量、日期范围、栏目分布、抓取参数和 SHA-256 | 用于确认当前数据版本和复现实验 |
| `data/manifests/raw_articles.manifest.json` | 记录原始文章文件的数量和 SHA-256 | 用于校验 `articles.jsonl` 是否完整 |
| `data/state/guardian.sqlite3` | M1 的断点续跑和去重状态库 | 仅供 M1 脚本使用，后续阶段不要直接读取 |

读取标准文档的示例：

```python
import json

with open("data/documents/documents.jsonl", encoding="utf-8") as stream:
    for line in stream:
        article = json.loads(line)
        document_id = article["document_id"]
        text = article["body_text"]
        url = article["url"]
```
