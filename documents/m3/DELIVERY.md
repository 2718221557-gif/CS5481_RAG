# M3 交付与使用

本次提交来自 `CS5481_RAG-stage3`，包括 M3 embedding、SQLite 向量缓存、Chroma 持久化索引、M4 查询编码/集合加载接口、三份索引 manifest，以及 M3 个人报告。远程仓库原来的 `scripts/m2_chunking` 是空占位文件，本次同时补入本地已有的 M2 实现，作为 M3 的上游依赖。

M3 报告：[English Word](M3_Report_EN.docx) · [中文说明](M3_Report_CN.md)。M2 是项目已有模块，不计作 M3 新实现。

## 下载完整数据

代码和小型 manifest 保存在 Git。大型数据通过 [v0.3.0-m3 Release](https://github.com/2718221557-gif/CS5481_RAG/releases/tag/v0.3.0-m3) 分发：

- `CS5481_RAG_stage3_data.tar.gz`：535,923,525 bytes，含原始新闻、清洗后的文档、A/B/C chunks、完整 Chroma 数据库/索引、向量缓存及 manifests。
- `SHA256SUMS.txt`：压缩包校验值。
- `CS5481_RAG_stage3_data.manifest.json`：逐文件大小和 SHA-256，以及各策略记录数。

数据包 SHA-256：

```text
89776eabda1e3dd4b1db8f096b970c0fb156357b40017dc8f78c49178f3ff586
```

将压缩包和 `SHA256SUMS.txt` 下载到项目根目录，校验后解压：

```bash
# macOS
shasum -a 256 -c SHA256SUMS.txt
# Linux 可使用：sha256sum -c SHA256SUMS.txt
tar -xzf CS5481_RAG_stage3_data.tar.gz
python -m pip install -r requirements.txt
```

压缩包内路径以 `data/` 开头。解压会写入相同路径；已有其他数据版本时请先备份或使用新克隆目录。GitHub 的 “Download ZIP” 和 `git clone` 只下载代码，不包含 Release 数据包。数据包不含 M1 爬虫运行状态、操作系统临时文件、Python 环境或模型权重；重新采集和查询编码需要各自的配置及模型。

M4 接入的详细字段、API 和运行示例见 [M3 → M4 对接文档](M4_HANDOFF.md)。

## 已保存的向量

| 策略 | Chroma collection | 记录数 | 维度 |
| --- | --- | ---: | ---: |
| A | `chunks_a_09ffa2bf18_c6c1836bed` | 17,015 | 384 |
| B | `chunks_b_4ef08cb904_c6c1836bed` | 15,914 | 384 |
| C | `chunks_c_e9f8efe647_c6c1836bed` | 16,432 | 384 |

总计 49,361 条文本块记录。缓存有 49,336 个不同的文本/模型键；重复文本共享缓存向量。模型为 `sentence-transformers/all-MiniLM-L6-v2`，revision 为 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，向量经 L2 归一化，检索距离为 cosine。

直接查看已保存向量不需要重新下载模型。在项目根目录运行：

```python
import chromadb

client = chromadb.PersistentClient(path="data/vector_store/chroma")
collection = client.get_collection(
    "chunks_b_4ef08cb904_c6c1836bed", embedding_function=None
)
print("记录数：", collection.count())
sample = collection.get(limit=1, include=["embeddings", "documents", "metadatas"])
print("原文：", sample["documents"][0])
print("来源：", sample["metadatas"][0])
print("向量维度：", len(sample["embeddings"][0]))
print("向量：", sample["embeddings"][0].tolist())
```

现有数据库由 ChromaDB 1.5.9 生成。若使用其他 Chroma 版本出现兼容问题，可在独立环境中安装 `chromadb==1.5.9` 读取。M4 查询必须使用与索引相同的模型版本及归一化方式。

## 从 chunks 重建

```bash
python -m scripts.m3_embedding.index_chunks --strategies A,B,C --device cpu --batch-size 64
```

此命令会使用已有向量缓存。首次加载模型需要网络或本机模型缓存。当前 CLI 记录实际模型 revision，但未提供 `--revision` 锁定参数；远程模型更新后重新构建的集合名可能不同。

本次数据检查覆盖 SQLite 完整性、集合记录数、chunk 文件校验值、384 维向量及引用 metadata。M4/M5/M6 的完整检索、生成、拒答和至少 20 条 QA 评估仍由对应模块完成；M3 验证不能替代最终检索质量评估。
