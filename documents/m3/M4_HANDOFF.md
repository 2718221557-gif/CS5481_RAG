# M3 → M4 交接说明：读取向量、编码问题与接入检索

本说明面向负责 M4 检索的组员，描述这次交付的**实际文件、已实现接口和已有结果**。M3 已完成文档向量化、缓存、Chroma 持久化索引和查询编码辅助接口；完整 M4 检索服务、策略比较、延迟测试仍由 M4 实现。

- [交付总说明与报告](DELIVERY.md)
- [完整数据下载：v0.3.0-m3 Release](https://github.com/2718221557-gif/CS5481_RAG/releases/tag/v0.3.0-m3)
- [M3 实现](../../scripts/m3_embedding/index_chunks.py) · [M3 使用说明](../../scripts/m3_embedding/README_zh.md)
- [可运行查询示例](examples/query_m3.py)：用于验证接入，不代表完整 M4 模块。

## 1. M4 先做这几步

1. 获取包含本说明的项目代码，在新克隆目录操作。`git clone` 或 GitHub “Download ZIP” **不包含完整数据库**。
2. 从上述 Release 下载 `CS5481_RAG_stage3_data.tar.gz` 和 `SHA256SUMS.txt` 到项目根目录，校验并解压。
3. 在独立 Python 环境中安装依赖。交付数据库已在 `chromadb==1.5.9` 下读取验证。
4. 先检查集合记录数和一条保存的向量，再用相同模型 revision 编码查询。

在项目根目录运行：

```bash
# macOS；Linux 对应命令是 sha256sum -c SHA256SUMS.txt
shasum -a 256 -c SHA256SUMS.txt
tar -xzf CS5481_RAG_stage3_data.tar.gz
python -m pip install -r requirements.txt
# 如需使用本次验证过的 Chroma 版本：
python -m pip install 'chromadb==1.5.9'

python documents/m3/examples/query_m3.py \
  --strategy B --top-k 5 \
  --query "What happened to Australian tourists in Nepal?"
```

Windows 可使用系统 `tar` 解压；查询命令也可合并为一行执行。首次编码问题需要联网下载模型，或事先准备对应模型缓存；读取保存的向量不需要下载模型。

数据包大小为 **535,923,525 bytes**，SHA-256：

```text
89776eabda1e3dd4b1db8f096b970c0fb156357b40017dc8f78c49178f3ff586
```

包内路径以 `data/` 开头，应解压到项目根目录。它包含全部原始/清洗新闻、chunks、Chroma 索引、embedding 缓存和 manifests；不含模型权重、Python 环境或 M1 爬虫状态数据库。M4 查询现有索引只需要 `data/vector_store/chroma/`、对应索引 manifest 和正确模型；缓存数据库及 M1 原始数据不参与在线检索。

## 2. 代码与数据位置

```text
scripts/
  common/paths.py                  # 根据源码位置确定 PROJECT_ROOT 和 DATA_ROOT
  m2_chunking/chunk_documents.py   # 上游：生成 A/B/C 三套 chunks
  m3_embedding/index_chunks.py     # 编码、缓存、写入 Chroma、生成 manifest
data/
  chunks/
    chunks_A.jsonl
    chunks_B.jsonl
    chunks_C.jsonl
    chunks_A.manifest.json         # M2 清单；B/C 各有一份
  vector_store/
    chroma/
      chroma.sqlite3
      <segment UUID>/...          # HNSW 等持久化文件，必须与 SQLite 一起保留
    embedding_cache.sqlite3      # 重建时复用向量，不是查询入口
    manifests/
      chunks_a_09ffa2bf18_c6c1836bed.manifest.json
      chunks_b_4ef08cb904_c6c1836bed.manifest.json
      chunks_c_e9f8efe647_c6c1836bed.manifest.json
documents/m3/
  DELIVERY.md
  M4_HANDOFF.md
  examples/query_m3.py
  CS5481_RAG_stage3_data.manifest.json
  SHA256SUMS.txt
```

不要仅复制 `chroma.sqlite3`；请保留整个 `chroma/` 目录。Chroma 客户端打开持久化目录时可能执行内部写入，因此目录需要可写。不要在共享网络目录上同时让多个进程修改同一份本地数据库；每位组员可解压自己的副本。

## 3. 已交付的三个索引

| 文件/manifest 策略标签 | chunk 与 Chroma metadata 中的 `strategy` | 分块方式 | collection | 记录数 |
| --- | --- | --- | --- | ---: |
| A | `fixed_overlap` | 默认 1,200 字符窗口、150 字符重叠；优先在靠近末尾的句界结束 | `chunks_a_09ffa2bf18_c6c1836bed` | 17,015 |
| B | `paragraph` | 段落优先；过长段落按句子处理，必要时按字符切分 | `chunks_b_4ef08cb904_c6c1836bed` | 15,914 |
| C | `sentence_title` | 句子优先组合，每块添加文章标题；默认总预算 1,200 字符 | `chunks_c_e9f8efe647_c6c1836bed` | 16,432 |

**注意：`where={"strategy": "B"}` 不会匹配 B 集合的记录。** B 是文件/清单标签，记录字段实际是 `paragraph`。每种策略单独成库，选择 collection 就已选定策略。

三套索引共有以下设置：

| 项目 | 实际值 |
| --- | --- |
| 文档语料规模 | 3,152 篇文档；三套分块结果总计 49,361 条记录 |
| 模型 | `sentence-transformers/all-MiniLM-L6-v2` |
| 模型 revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| 向量维度 | 384 |
| 编码输出 | float32，L2 归一化 |
| Chroma 距离空间 | cosine |
| 建库 batch size | 64 |
| 实际建库设备 | CPU |
| 缓存唯一键数量 | 49,336；重复文本可共享向量，但 chunk 记录仍分别保存 |

A/B/C 必须在相同问题集、相同模型版本、相同 top-k 等条件下比较。现有数量和分块统计不能证明某种策略检索质量更好。

## 4. M2 输入格式：`news_chunks.v1`

`chunks_*.jsonl` 是 UTF-8 JSON Lines：**每行一个完整 JSON 对象**，不能直接用 `json.load()` 读取整个文件。换行符等在 JSON 字符串内转义。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `schema_version` | string | 固定为 `news_chunks.v1` |
| `chunk_id` | string | SHA-256 标识；由文档 ID、策略名、块序号和完整块文本生成 |
| `document_id` | string | 原始文章的稳定标识 |
| `chunk_index` | integer | 文章内从 0 开始的块序号 |
| `strategy` | string | `fixed_overlap` / `paragraph` / `sentence_title` |
| `text` | string | 实际送入 embedding 模型的文本 |
| `word_count` | integer | `len(text.split())`；不是模型 token 数 |
| `citation` | object | 原文 URL、标题、UTC 发布时间、来源名、来源域名 |
| `metadata` | object | 可用的 `section_id`、`section_name`、`language` |

下面摘自 B 文件第一条记录。**只有 `text` 为展示截短；`word_count=150` 对应原始完整文本，省略标记不是原数据。**

```json
{
  "schema_version": "news_chunks.v1",
  "chunk_id": "bc539165d46fb6403d8766de54aefbf57a1bf215ee8e7866c4306629f0a4e4ea",
  "document_id": "australia-news/2026/aug/27/australian-tourists-reported-missing-nepal-tibet-floods",
  "chunk_index": 0,
  "strategy": "paragraph",
  "text": "The Australian government is urgently seeking information on the welfare of at least 35 Australians missing in Nepal, after a catastrophic flash flood swept through the Nepal-Tibet border region. [为展示省略后续原文]",
  "word_count": 150,
  "citation": {
    "url": "https://www.theguardian.com/australia-news/2026/aug/27/australian-tourists-reported-missing-nepal-tibet-floods",
    "title": "Australian officials ‘working around the clock’ to check welfare of tourists missing in Nepal",
    "published_at_utc": "2026-08-27T18:18:50Z",
    "source_name": "The Guardian",
    "source_domain": "theguardian.com"
  },
  "metadata": {
    "section_id": "world",
    "section_name": "World news",
    "language": "en"
  }
}
```

读取一条完整记录：

```python
import json
from pathlib import Path

with Path("data/chunks/chunks_B.jsonl").open(encoding="utf-8") as stream:
    first_chunk = json.loads(next(stream))
print(first_chunk["chunk_id"], first_chunk["text"])
```

## 5. Chroma 保存的字段和索引清单

M3 将嵌套引用信息展开为 Chroma 的标量 metadata。检索结果已经包含原文引用，M4/M5 不需要回查 M1 状态数据库。

| M2 字段/生成结果 | Chroma 位置 | 类型 |
| --- | --- | --- |
| `chunk_id` | `ids[i]` | string；**不重复保存为 metadata.chunk_id** |
| `text` | `documents[i]` | string |
| embedding | `embeddings[i]` | 384 个浮点数 |
| `schema_version` | `metadatas[i]["schema_version"]` | string |
| `document_id` | `metadatas[i]["document_id"]` | string |
| `chunk_index` | `metadatas[i]["chunk_index"]` | integer |
| `strategy` | `metadatas[i]["strategy"]` | string |
| `citation.url` | `metadatas[i]["citation_url"]` | string |
| `citation.title` | `metadatas[i]["citation_title"]` | string |
| `citation.published_at_utc` | `metadatas[i]["citation_published_at_utc"]` | ISO 8601 UTC string |
| `citation.source_name` | `metadatas[i]["citation_source_name"]` | string |
| `citation.source_domain` | `metadatas[i]["citation_source_domain"]` | string |
| `metadata.section_id` | `metadatas[i]["document_section_id"]` | string（有值时保存） |
| `metadata.section_name` | `metadatas[i]["document_section_name"]` | string（有值时保存） |
| `metadata.language` | `metadatas[i]["document_language"]` | string（有值时保存） |
| `word_count` | 未保存 | 需要时从返回文本计算，或关联原始 chunk JSONL |

通用 `_metadata()` 对其他 `metadata` 中的 string/int/float/bool 也采用 `document_` 前缀；`None`、列表和嵌套对象不会保留。不要假设所有可能字段永远存在，组装引用时可用 `.get()` 并检查必需字段。

以 B 的索引 manifest 为例，重要字段为：

```json
{
  "schema_version": "embedding_index.manifest.v1",
  "input": {
    "path": "data/chunks/chunks_B.jsonl",
    "sha256": "4ef08cb904ad43c5983b39228288be28a58d8868054e11589cae4b76b43e2243",
    "chunk_schema_version": "news_chunks.v1",
    "strategy": "B"
  },
  "embedding": {
    "backend": "sentence-transformers",
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "dimension": 384,
    "normalization": "L2"
  },
  "vector_store": {
    "backend": "ChromaDB",
    "persist_path": "data/vector_store/chroma",
    "collection": "chunks_b_4ef08cb904_c6c1836bed",
    "distance": "cosine"
  },
  "embedding_cache": "data/vector_store/embedding_cache.sqlite3",
  "batch_size": 64,
  "chunk_count": 15914,
  "cached_vectors_used": 21,
  "vectors_created": 15892
}
```

这是清单字段节选，省略了 `created_at_utc`。路径相对于项目根目录。B 的 `cached_vectors_used + vectors_created` 比记录数少 1，是因为同一批次内还有 1 条重复文本复用了刚生成的向量；不是漏写记录。

M4 应从 manifest 获取 collection、模型 revision、维度和路径。不要依据目录里文件数量推断向量条数，也不要只检查“都是 384 维”就混用不同模型。

## 6. 最小可运行接入代码

### 6.1 查看已有向量：`get()`

在项目根目录运行。这里只读取已保存的数据，不加载 Sentence Transformers 模型：

```python
import chromadb

client = chromadb.PersistentClient(path="data/vector_store/chroma")
collection = client.get_collection(
    name="chunks_b_4ef08cb904_c6c1836bed",
    embedding_function=None,
)
print(collection.count())  # 本次 B 应为 15914
saved = collection.get(limit=1, include=["documents", "metadatas", "embeddings"])
print(saved["ids"][0])
print(saved["documents"][0])
print(saved["metadatas"][0])
print(len(saved["embeddings"][0]))  # 384
```

`get()` 返回**一层记录列表**：`ids[i]`、`documents[i]`、`metadatas[i]`、`embeddings[i]` 相互对应。它不会按语义相关性排序，也不返回检索距离。Python 客户端的 embeddings 可能是 NumPy 数组；导出 JSON 时可调用 `.tolist()`。

第 4 节 B 样例 `bc539165...a4e4ea` 的已保存向量前 8 维实际为：

```text
[0.0093174381, -0.0030454046, 0.0256584641, 0.0660836622,
 0.0572646745, 0.0229742564, -0.0361184180, -0.0608412139]
```

这里只展示完整 384 维中的前 8 维；每个数值不是单独的关键词标签或概率。

### 6.2 编码问题并执行 `query()`

这个例子明确固定 manifest 中的模型 revision，并自行传入 query embedding。**不要改成 `query_texts=[question]`，也不要让 Chroma 自动选择默认 embedding 模型。**

```python
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

root = Path.cwd()  # 先切换到项目根目录
manifest = json.loads((root / "data/vector_store/manifests/"
    "chunks_b_4ef08cb904_c6c1836bed.manifest.json").read_text(encoding="utf-8"))

client = chromadb.PersistentClient(
    path=str(root / manifest["vector_store"]["persist_path"])
)
collection = client.get_collection(
    name=manifest["vector_store"]["collection"], embedding_function=None
)
assert collection.count() == manifest["chunk_count"]

model = SentenceTransformer(
    manifest["embedding"]["model"],
    revision=manifest["embedding"]["revision"],
    device="cpu",
)
question = "What happened to Australian tourists in Nepal?"
vectors = model.encode(
    [question], normalize_embeddings=True,
    convert_to_numpy=True, show_progress_bar=False,
)
assert vectors.shape == (1, manifest["embedding"]["dimension"])

result = collection.query(
    query_embeddings=vectors.tolist(),
    n_results=5,
    include=["documents", "metadatas", "distances"],
)
for rank, chunk_id in enumerate(result["ids"][0], start=1):
    i = rank - 1
    distance = float(result["distances"][0][i])
    metadata = result["metadatas"][0][i]
    print(rank, chunk_id, "distance=", distance,
          "cosine_similarity=", 1.0 - distance)
    print(metadata["citation_url"])
    print(result["documents"][0][i])
```

模型在服务启动时加载一次并复用，不应每个问题重新加载。以上为便于直接运行的接入示例；正式 M4 还应处理参数、过滤、空结果、错误与计时。

### 6.3 `get()` 与 `query()` 返回结构不能混用

| 项目 | `get()` | `query()` |
| --- | --- | --- |
| 用途 | 按 ID、条件或数量读取记录 | 对一个或多个查询向量检索近邻 |
| IDs | `ids[record_i]` | `ids[query_i][rank_i]` |
| 文本 | `documents[record_i]` | `documents[query_i][rank_i]` |
| metadata | `metadatas[record_i]` | `metadatas[query_i][rank_i]` |
| 保存的向量 | 显式 include 后 `embeddings[record_i]` | 显式 include 后 `embeddings[query_i][rank_i]` |
| 距离 | 无 | include 后 `distances[query_i][rank_i]` |

即使只提交一个问题，`query()` 仍有查询这一外层维度，因此结果要取 `[0][i]`。多个查询可以一次传入二维 `query_embeddings`，外层顺序与输入问题顺序一致。`ids` 自动返回，不要把 `"ids"` 放到 `include` 中；未请求的可选字段可能为 `None`。

过滤后可能少于 top-k，或得到 `ids=[[]]`；应遍历实际返回的 ID，不能假设一定有 k 条。cosine **distance 越小越相近**，`similarity = 1 - distance`；它不是概率。浮点误差可能让自匹配距离略小于 0。当前没有通过 QA 集标定的“相关/不相关”阈值，不应凭示例数字制定拒答规则。

### 6.4 一次实际自然语言查询结果

本次在原数据库的临时副本上运行了下面的示例命令，使用缓存的指定模型 revision、CPU 和 Chroma 1.5.9：

```bash
python documents/m3/examples/query_m3.py --strategy B --top-k 2 \
  --language en --section world \
  --query "What happened to Australian tourists in Nepal?"
```

返回值中 `ids` 与 `distances` 的**实际节选**如下，保留了单问题仍有外层列表的结构：

```json
{
  "ids": [[
    "311913283b1e52c3ec35a129bcbfef1abfbde853b3abe83538c991a8da2d76c9",
    "86a81e22bbe562055b1c33929ef78b82377767521a32364e24208b1e02d1a22f"
  ]],
  "distances": [[0.30353236198425293, 0.30654650926589966]]
}
```

两条结果均来自项目中标题为 “Australian officials ‘working around the clock’ to check welfare of tourists missing in Nepal” 的文章，`chunk_index` 分别为 1 和 3，`citation_url` 与第 4 节样例一致。第一条文本以 `“This is the trekking season in Nepal, it’s a very popular time for people to visit in the adventure tourism industry,” he said.` 开始。

这是一次查询通路检查，不是完整质量实验；同一篇文章返回多个块也说明 M4 需要自行决定是否在送给 M5 前进行文档级去重。硬件、依赖版本和近似搜索状态可能带来细微数值或排序差异。

## 7. 过滤示例与日期限制

以下字段和值来自实际数据库，可以作为 `collection.query(..., where=where)` 的条件：

```python
# 一种语言
where = {"document_language": {"$eq": "en"}}

# 多个栏目之一（栏目来自元数据，不要只从 URL 前缀推断）
where = {"document_section_id": {"$in": ["world", "politics"]}}

# 同时满足语言和栏目；多条件用 $and
where = {"$and": [
    {"document_language": {"$eq": "en"}},
    {"document_section_id": {"$eq": "world"}},
]}

# B 的真实策略名；选择 B collection 后通常不必重复过滤
where = {"strategy": {"$eq": "paragraph"}}

# 来源与特定文章也可过滤
where = {"citation_source_domain": {"$eq": "theguardian.com"}}
where = {"document_id": {"$eq": "australia-news/2026/aug/27/"
    "australian-tourists-reported-missing-nepal-tibet-floods"}}
```

示例脚本支持组合过滤：

```bash
python documents/m3/examples/query_m3.py --strategy B --top-k 5 \
  --language en --section world \
  --query "What happened to Australian tourists in Nepal?"
```

`citation_published_at_utc` 目前是 `2026-08-27T18:18:50Z` 这种**字符串**，没有数值 timestamp 字段。不要直接给这个字段使用数值范围操作 `$gte`/`$lte`。M4 如果需要日期区间，应与组内约定增加数值时间字段并更新索引版本，或者对候选结果解析日期后过滤。后过滤会减少候选数，需要明确定义补取策略，不能保证原 top-k 里过滤后就代表整个日期区间内最相关的 k 条。

## 8. 已有 M3 Python 接口及边界

导入位置：`scripts.m3_embedding.index_chunks`。当前源码的公开辅助接口如下：

```python
class SentenceEmbeddingModel:
    def __init__(self, model_name: str = DEFAULT_MODEL,
                 device: str | None = None) -> None: ...
    def encode_documents(self, texts: list[str], batch_size: int) -> np.ndarray: ...
    def encode_query(self, text: str) -> list[float]: ...

def load_collection(collection_name: str,
                    persist_dir: Path | None = None) -> Any: ...

def build_index(
    chunk_path: Path, *, model: SentenceEmbeddingModel,
    persist_dir: Path, cache_path: Path, manifest_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IndexResult: ...

class EmbeddingCache:
    def __init__(self, path: Path) -> None: ...
    def get_many(self, keys: Iterable[str]) -> dict[str, np.ndarray]: ...
    def put_many(self, vectors: dict[str, np.ndarray]) -> None: ...
    def close(self) -> None: ...
```

- `SentenceEmbeddingModel` 公开属性包括 `model_name`、`model`、`dimension`、`revision`、`fingerprint`。`encode_documents()` 返回 `(文本数, 维度)` 的归一化 float32 数组；`encode_query()` 返回一条归一化向量的 Python list。
- **当前构造函数没有 `revision` 参数，CLI 也没有 `--revision`。** 它加载模型后记录解析到的 revision；这不等于强制加载某个历史 revision。为确保 M4 查询与此次已保存索引一致，本说明的接入代码直接使用 `SentenceTransformer(..., revision=manifest["embedding"]["revision"])`。
- `load_collection()` 默认加载 `DATA_ROOT / "vector_store" / "chroma"`，不会重建文档向量。当前内部 `get_collection()` 未显式传 `embedding_function=None`，所以本说明使用显式关闭默认 embedder 的写法；若调用现有 helper，也应始终传 `query_embeddings`，不要调用 `query_texts`。
- `build_index()` 负责按批次编码、缓存、upsert 和写 manifest；M4 查询无需调用。输入 basename 必须对应 `chunks_A/B/C.jsonl`；重复 chunk ID 或不支持的 schema 会报错。
- `IndexResult` 字段是 `collection_name`、`input_path`、`input_sha256`、`chunk_count`、`embedding_dimension`、`cached_vectors_used`、`vectors_created`、`manifest_path`。
- 缓存键根据模型名称/revision 的 SHA-256 指纹与**原样 chunk text** 的 SHA-256 生成；代码没有对文本再做标准化。修改空格或换行也会改变缓存键。SQLite 表 `embeddings` 保存 `cache_key`、`dimension`、float32 `vector` BLOB。

重建命令：

```bash
python -m scripts.m3_embedding.index_chunks \
  --strategies A,B,C --device cpu --batch-size 64
```

还支持 `--model`、`--chunks-dir`、`--persist-dir`、`--cache`、`--manifest-dir`。同一输入与模型通过稳定 chunk ID upsert；输入 SHA 或模型指纹变化会产生新 collection。M4 应明确选定对应 manifest，不要自动选“目录里的第一个文件”。示例脚本发现同一策略有多个 manifest 时要求用 `--manifest` 明确指定。

## 9. 建议的 M4 → M5 输出契约（待组内确认）

**下面是建议，不是现有 M4 已实现接口。** 当前项目尚没有正式 `retrieve()` 或统一结果类。建议 M4 输出每个问题对应的结果对象，保留以下内容：

| 建议字段 | 类型/来源 | 用途 |
| --- | --- | --- |
| `query` | string | 原始问题 |
| `strategy` | `A` / `B` / `C` | 对比实验标识 |
| `collection` | string，来自 manifest | 定位具体数据/模型版本 |
| `requested_top_k` | integer | 记录检索参数 |
| `filters` | object 或 null | 记录生效的条件 |
| `results` | list | 无证据时为空列表 |
| `results[].rank` | 从 1 开始的 integer | 结果次序 |
| `results[].chunk_id` | Chroma ID | 精确引用块 |
| `results[].document_id` | metadata.document_id | 文档归属、文档级去重 |
| `results[].chunk_index` | metadata.chunk_index | 文档内位置 |
| `results[].text` | Chroma document | M5 可用的证据原文 |
| `results[].distance` | float | 原始 cosine distance |
| `results[].similarity` | float = 1 − distance | 可选便利字段；不是置信概率 |
| `results[].citation` | URL、标题、发布时间、来源名/域名 | 恢复 M2 的嵌套 citation 对象 |
| `timing` | object | 建议区分查询编码与向量检索耗时，单位写明 |

M5 可给最终选用的证据分配 `[S1]`、`[S2]` 标签，并保留标签到 `chunk_id + document_id + URL` 的映射。引用顺序可能在去重或重排后变化，不能把临时排名当永久 chunk ID。是否支持回答由 M5 根据证据决定；M4 不应补写或生成新闻事实来填充空结果。

建议先实现一个 collection 的检索与引用字段透传，再加 A/B/C 切换、top-k、过滤与耗时记录，最后交给 M6 用固定 QA 集比较。去重、重排、拒答阈值若引入，应单独记录规则并评估其影响。

## 10. 已验证内容与仍需完成的工作

本次交付已完成以下检查：

- 压缩包 30 个文件逐一解压读取并比对源文件 SHA-256；M2 三份输入哈希与 M3 manifest 一致。
- Chroma 与 embedding cache 的 SQLite `quick_check` 均为 `ok`。
- 将整个 Chroma 目录复制到临时位置后通过 Chroma 1.5.9 成功重新打开；A/B/C 实际记录数分别为 17,015 / 15,914 / 16,432。
- 每个集合抽查一条，均有文本、引用 URL 和 384 维向量；L2 范数约为 1。
- 将抽查的已存向量作为 query embedding，自匹配近邻的 cosine distance 分别约为 `5.96e-8`、`0.0`、`-1.19e-7`，符合浮点误差范围内的近零距离。
- [查询示例脚本](examples/query_m3.py) 已在复制的 B 数据库上运行成功，固定模型 revision，完成自然语言编码和语言/栏目组合过滤；实际结果见第 6.4 节。单独的语言 `$eq`、栏目 `$in` 和策略 `$eq` 也已通过读取检查。
- M2/M3 CLI `--help` 可运行；已有 4 个 M1 测试函数的断言全部通过。测试环境未安装 pytest，因此使用独立临时目录直接执行测试函数，并非一次完整 pytest 命令运行。

这些检查验证了数据完整性、持久化重开和查询通路，**不代表已测得检索命中率、回答正确率或平均延迟**。自匹配也不能替代自然语言问题检索评估。

已知限制及分工：

| 事项 | 当前情况 / 下一步 |
| --- | --- |
| 模型输入截断 | 此模型默认最多 256 wordpieces；M2 按字符数分块，部分文本可能超过模型长度而被截断。尚未统计截断比例。完整文本仍保存在 Chroma document，但 embedding 可能未覆盖末尾；应在检索分析中检查。 |
| 模型版本锁定 | M3 CLI 没有 `--revision`；现有索引查询按本说明显式固定 revision。重新生成数据后应重新核对 manifest。 |
| 环境复现 | requirements 使用版本范围，不能据此声称所有机器逐位复现；本次 Chroma 读取验证版本为 1.5.9。 |
| 检索参数与质量 | M4 需实现并评估 top-k、过滤、可能的去重/重排；现有代码没有正式检索服务和延迟报告。 |
| 引用和拒答 | M3 已保留来源字段；M5 需实现逐项引用、证据约束与无依据拒答。 |
| 课程评估 | M6 需至少 20 组 QA，并测检索命中率与回答正确性；M4 需提供可重复检索结果和参数，支持至少两种 chunking 的公平比较。 |
