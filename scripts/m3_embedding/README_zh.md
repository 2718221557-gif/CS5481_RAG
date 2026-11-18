# M3：Embedding 与向量库

本模块将 M2 的 `news_chunks.v1` JSONL 编码为向量，并存入本机持久化 ChromaDB。它同时生成适用于 M4 的查询编码器和集合加载函数；语义检索逻辑由 M4 实现。

## 依赖与模型

在项目根目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

默认使用开源 Sentence Transformers 模型 `sentence-transformers/all-MiniLM-L6-v2`，本地生成并 L2 归一化 embedding。首次运行会从模型仓库下载模型文件；也可以通过 `--model` 指向其他模型 ID 或本机模型目录。默认自动选择设备，可用 `--device cpu`、`--device mps` 或 `--device cuda` 指定。MPS 需要 macOS 14 或更新版本及支持 MPS 的 PyTorch；本项目本次因当前系统不支持 MPS，使用 CPU 完成建库。

## 建立索引

```bash
python -m scripts.m3_embedding.index_chunks --strategies A,B,C --batch-size 64
```

可先只构建某个 chunk 方案，例如 `--strategies B`。默认读取 `data/chunks/chunks_A.jsonl`、`chunks_B.jsonl` 和 `chunks_C.jsonl`。

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--strategies` | `A,B,C` | 选择一个或多个 M2 策略 |
| `--model` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformers 模型 ID 或本地模型目录 |
| `--device` | 自动选择 | 可指定 `cpu`、`mps` 或 `cuda` |
| `--batch-size` | `64` | 批量编码和写入向量库的 chunk 数 |
| `--chunks-dir` | `data/chunks` | M2 chunk 输入目录 |
| `--persist-dir` | `data/vector_store/chroma` | Chroma 持久化目录 |
| `--cache` | `data/vector_store/embedding_cache.sqlite3` | embedding SQLite 缓存 |
| `--manifest-dir` | `data/vector_store/manifests` | 索引 manifest 目录 |

## 输出与可复现

- Chroma 集合保存在 `data/vector_store/chroma/`，距离空间为 cosine。
- 每条向量保留 chunk 文本及 `document_id`、`chunk_index`、策略、原文 URL、标题、发布时间、来源和可用的栏目/语言 metadata。
- 集合名称由策略、chunk 文件 SHA-256 前缀及模型指纹构成。更换 chunk 文件或模型 revision 会创建新集合，便于并行比较并保留旧索引。
- SQLite 缓存以模型指纹和 chunk 文本 SHA-256 为键。重复构建相同输入时可重用已计算向量；稳定 `chunk_id` 用于 Chroma upsert。
- 每个集合生成一个 `embedding_index.manifest.v1` 文件，包含输入校验和、模型名/修订号/维度、向量库路径/集合名、距离度量、批次大小及缓存统计。

M4 可从 manifest 中读取集合名称，使用 `load_collection(name)` 加载索引；查询问题可由 `SentenceEmbeddingModel.encode_query(question)` 编码。若模型升级或 chunk 内容变化，请使用对应的新 manifest 和集合，不要混用不同维度或模型版本的查询向量。

## 已构建结果

本项目三套索引均已使用同一个模型版本完成构建，方便 M4 在相同 embedding 空间下比较检索效果：

| 策略 | 集合 | 向量数 | 维度 |
| --- | --- | ---: | ---: |
| A | `chunks_a_09ffa2bf18_c6c1836bed` | 17,015 | 384 |
| B | `chunks_b_4ef08cb904_c6c1836bed` | 15,914 | 384 |
| C | `chunks_c_e9f8efe647_c6c1836bed` | 16,432 | 384 |

模型 revision：`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`。每套 manifest 的 chunk 数与 Chroma 实际记录数一致，且抽查记录保留了原文 URL。

索引与缓存文件位于 `data/vector_store/`，大型二进制文件通过 [M3 Release](https://github.com/2718221557-gif/CS5481_RAG/releases/tag/v0.3.0-m3) 分发，三份索引 manifest 保存在 Git 中。组员可按 [交付说明](../../documents/m3/DELIVERY.md) 下载并解压现有向量库，也可安装依赖后运行上面的命令重建。CPU 环境下完整构建耗时较长；M3 不会改写 M2 源文件。

## M4 对接

负责检索的同学请阅读 [M3 → M4 对接文档](../../documents/m3/M4_HANDOFF.md)，其中包括实际字段映射、模型版本约束、Chroma 查询返回格式、metadata 过滤和运行示例。
