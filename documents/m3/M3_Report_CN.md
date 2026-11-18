# M3 Embedding 与向量数据库报告中文说明

这份报告对应你负责的 M3，可将英文正文中的四个小节分别并入小组报告的 Introduction、Methodology、Experiments 和 Discussions。英文 Word 的附录列出配置、接口、建库命令和证据，供小组复现或放入总报告附录。

## 中文正文

### 引言

本项目实现了面向新闻知识库的 embedding 与向量数据库模块。M3 将 M2 产出的文本块转换成可供语义检索的向量记录，同时保留后续引用所需的来源信息。这一部分对应 Topic 5 的文档向量化和知识库存储要求。目前已构建 A、B、C 三个持久化集合，共包含 49,361 条记录。

### 方法

我们采用预训练的 sentence-transformers/all-MiniLM-L6-v2 模型，不进行额外训练。该模型为句子和短段落生成 384 维向量。本地推理避免了付费 embedding API 的调用，并为三种 chunking 策略提供相同的编码模型。程序以 64 条文本为一批，生成 float32 向量并进行 L2 归一化，Chroma 使用 cosine 距离。

输入采用 M2 的 news_chunks.v1 格式。加载阶段检查 schema、非空标识与文本，以及重复 chunk ID。SQLite 缓存根据模型名称、识别到的版本和原始文本的哈希确定是否已有向量；批次内相同且未缓存的文本只编码一次。建库采用稳定 chunk ID 执行 upsert，使同一集合的重复运行可以更新已有记录。

向量、原始 chunk 文本及 metadata 一起存储于 Chroma 本地持久化集合。Metadata 包含 document_id、chunk_index、策略、原文 URL、标题、发布时间、来源名称与域名，以及可用的栏目和语言信息。这些信息支持 M4 检索后的文档定位，也为 M5 构建引用提供数据基础。集合名称包含输入文件和模型指纹；manifest 记录输入 SHA-256、模型版本、维度及构建参数。load_collection 与 encode_query 提供面向 M4 的集合加载和问题编码接口。

### 构建结果与验证

三套现有 chunk 数据均在 CPU 上使用同一模型快照完成索引构建：

| 策略 | 向量记录数 | 向量维度 |
| --- | ---: | ---: |
| A 固定窗口与重叠 | 17,015 | 384 |
| B 段落优先 | 15,914 | 384 |
| C 句子优先并附标题 | 16,432 | 384 |
| 合计 | 49,361 | 384 |

持久化集合的实际记录数与各自 manifest 一致；抽查记录保留了来源 URL。在 B 集合完成的一次查询接口检查中，384 维问题向量能够用于查询，并返回文本与引用 metadata。这些证据说明建库流程及基本接口能够运行。

### 讨论

当前结果说明索引构建完成，尚不能据此判断哪种 chunking 策略的检索质量最好。检索命中率、答案正确性和至少 20 组 QA 的评估仍需与 M4、M6 协作完成。模型默认对超过 256 个 word pieces 的输入进行截断，而 M2 按字符长度切分，因此后续需要统计 token 长度与截断比例。

Manifest 保存了本次模型 commit 与数据哈希，提供了版本追踪依据。但是当前加载器没有强制固定 revision，requirements.txt 也只限定了依赖版本范围，因此不能声称已实现逐位可复现。正式复现实验时应保存本次模型快照并锁定软件环境。

## 你可以使用的个人贡献说明

I was responsible for M3, including the pretrained embedding model wrapper, batched encoding, embedding cache, persistent vector-store construction, preservation of citation metadata, index manifests, and interfaces for query encoding and collection loading. I also completed full index construction for the three chunking strategies and checked the stored record counts and basic query interface.

姓名、贡献比例和签名需要由你们小组确认后填写。作业第 1 页要求小组报告正文包含 introduction、methodology、experiments 和 discussions；正文最多 6 页，参考文献与附录不限页数。个人贡献及比例和所有组员签名应放在单独的贡献页，并按作业说明与附加幻灯片一起组织最终提交。本文件只是你的 M3 小节。

## 展示时可用的三句话

1. 我负责把三种 chunking 方案生成的文本块，用相同的 MiniLM 模型转换为 384 维向量，并存入 Chroma。
2. 我保留了每条向量对应的原文信息，并实现了批处理、缓存、集合加载和 manifest，方便后续检索与引用模块使用。
3. 目前三套索引共 49,361 条记录，建库和基本查询接口已经跑通；检索质量仍要用小组统一的问题集进行评估。

## 需要保持准确的表述

- 不把 B 策略较少的向量数解释为检索效果更好，也不填写未测得的准确率、延迟或加速比。
- 缓存依据原始文本哈希，没有额外执行文本归一化；L2 归一化是针对向量。
- 49,361 是数据库记录数；49,336 是本次新编码的不同文本数。两者不同是缓存与同批去重造成的，不是漏建记录。
- M3 保存引用 metadata，不等于已经验证生成答案的引用正确性；这属于后续模块的工作。
- 本次模型 revision 为 1110a243fdf4706b3f48f1d95db1a4f5529b4d41。其他成员查询现有索引时应使用匹配的模型快照，不能仅检查维度相同。

## 参考资料

- 模型说明：https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Chroma 持久化客户端：https://docs.trychroma.com/docs/run-chroma/clients
- SentenceTransformer API：https://sbert.net/docs/package_reference/sentence_transformer/model.html
