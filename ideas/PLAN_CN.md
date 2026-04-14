# COMP 4221 NLP课程项目 — 最终实施计划

> **目标**：在充满噪音的新闻语料库上构建一个可复现、有详尽文档说明的NLP管道。
> **评分优先级**：清晰的方法论 > 可复现性 > 深入的分析 > 原始性能。

---

## 1. 原则

- **只**使用提供的 `sample.txt` 和 `text_news.txt`。不使用任何外部数据语料库。
- 主管道中**不使用外部预训练模型**。允许使用本地开源库（如 scikit-learn, rank_bm25 等）。
- 所有需要人工判断的交付物（查询、相关性标签、主题标注）均由学生作者完成，不使用任何大语言模型（LLM）。
- 所有最终结果必须能追溯到：提供的语料库 + 作者的代码 + 作者的手动判断。

---

## 2. 任务选择

| 任务 | 选择 | 理由 |
|------|--------|-----------|
| 任务 1 | 轻量、非破坏性预处理 | 为下游任务保留信息 |
| 任务 2 | Unigram TF-IDF (基线) vs BM25 (改进) | 直接符合大作业示例；易于解释 |
| 任务 3 | **Track A** — 带手动标签的主题分类 | 更常规，混淆矩阵和 macro-F1 是天然契合的指标 |

---

## 3. 仓库结构

```
project/
├── data/
│   ├── raw/
│   │   ├── sample.txt
│   │   └── text_news.txt
│   ├── processed/
│   │   ├── text_news_cleaned.jsonl      # 内部格式
│   │   └── text_news_cleaned.txt        # 提交格式 (@@id 样式)
│   ├── annotations/
│   │   └── trackA_labels.jsonl
│   └── queries.json
├── src/
│   ├── task1_preprocess.py
│   ├── task2_retrieval.py
│   ├── compute_p_at_5.py
│   ├── task3_classification.py
│   └── utils/
│       ├── tokenizer.py                 # 为 TF-IDF 和 BM25 提供共享的 tokenize()
│       ├── annotator.py
│       └── labels.yaml
├── scripts/
│   ├── run_task1.sh
│   ├── run_retrieval.sh
│   └── run_task3.sh
├── outputs/
│   ├── task1_stats.md                   # task1_stats.pdf 的来源
│   ├── retrieval_results/
│   │   ├── top5_evaluation.csv
│   │   ├── summary.csv
│   │   └── per_query/
│   ├── figures/
│   │   ├── doc_length_hist.png
│   │   ├── zipf_plot.png
│   │   └── confusion_matrix.png
│   └── split.json
├── docs/
│   └── annotation_guideline.md
├── requirements.txt
└── README.md
```

---

## 4. 阶段 1 — 任务 1：预处理与语料库审计

### 4.1 解析器 (Parser)

- 以 `@@` 后跟数字 ID 作为分隔符进行拆分。
- 提取 `doc_id` (数字部分) 和 `raw_text`。

### 4.2 清洗步骤

| 步骤 | 操作 | 原因 |
|------|--------|--------|
| 间隔噪音 | 映射 `@ @ @ @ @ @ @ @ @ @` → `<GAP>` | 保持结构而不污染词汇表 |
| 大小写 | 所有文本小写化 | 标准归一化 |
| 空白符 | 压缩多个空格 / 归一化 `\r\n` | 清晰的格式 |
| 停用词 | 在此阶段**不移除** | 任务 2 检索在计算 BM25 `avgdl` 时需要它们；任务 3 向量化器会在内部处理 |
| 标点符号 | 在此阶段**不移除** | 任务 2 摘要中的句子拆分依赖于标点符号 |
| 罕见词元 | 在此阶段**不移除** | 过度修剪会损害检索召回率 |

### 4.3 输出格式

**内部** — `text_news_cleaned.jsonl`:
```json
{"doc_id": "3000041", "raw_text": "...", "clean_text": "..."}
```

**提交** — `text_news_cleaned.txt`:
保持与原始 `text_news.txt` **完全相同的行结构**：
```
@@3000041  doc 1 的清洗后文本 ...
@@3000341  doc 2 的清洗后文本 ...
```
每个文档保留在**单独一行**。`@@id` 和清洗后的文本在同一行，用两个空格分隔。

### 4.4 需报告的统计数据

- 文档数量
- 总词元数
- 平均文档长度 (词元)
- 中位数文档长度 (词元)
- 预处理**前**的词汇表大小
- 轻度清洗**后**的词汇表大小
- 3个清洗前/后的示例对比

### 4.5 可视化

- **文档长度分布直方图** (强制要求)
- **齐夫定律图 (Zipf's law plot)** — 对数-对数词频分布 (推荐)

### 4.6 监督 vs 无监督判断

需明确写出：
- 语料库不包含标签列，也没有主题/分类元数据。
- 文档 ID (`@@3000041`) 只是数字标识符，不是有意义的标签。
- 因此，该语料库是一个**无监督（无标签）的文本集合**。
- 这就是为什么任务 3 需要手动标注来创建标签。

### 4.7 交付物

- `src/task1_preprocess.py` (对应要求的 `preprocess.py`)
- `data/processed/text_news_cleaned.jsonl`
- `data/processed/text_news_cleaned.txt`
- `outputs/task1_stats.md` (导出为 `task1_stats.pdf`)
- `outputs/figures/doc_length_hist.png`
- `outputs/figures/zipf_plot.png`

---

## 5. 阶段 2A — 任务 2：查询脚手架

### 5.1 格式

创建包含 10 个槽位的 `data/queries.json`。最终的查询**必须由学生自己编写**。

### 5.2 查询编写指南

- 编写为**自然语言形式的信息需求**，而不是关键词列表。
- 涵盖语料库中存在的多样化主题。
- 可接受的风格示例：
  - "有哪些文档讨论了经济增长或通货膨胀趋势？" (Which documents discuss economic growth or inflation trends?)
  - "查找关于体育比赛或运动员表现的报告。" (Find reports about sports matches or athlete performance.)
  - "哪些文章涉及刑事案件、法院判决或执法？" (Which articles cover criminal cases, court decisions, or law enforcement?)

---

## 6. 阶段 2B — 任务 3：标注设置

### 6.1 标签方案

首先对随机挑选的 30–50 个文档进行**初步浏览 (pilot browse)**，以确定实际的主题分布。然后选择：

**选项 A (5个类别)**：
- 政治 (Politics)
- 商业/经济 (Business/Economy)
- 体育 (Sports)
- 文化/娱乐 (Culture/Entertainment)
- 其他 (Other)

**选项 B (6个类别)** — 如果初步浏览确认有大量涉及犯罪/法律的报道：
- 政治 (Politics)
- 商业/经济 (Business/Economy)
- 体育 (Sports)
- 犯罪/法律 (Crime/Law)
- 文化/娱乐 (Culture/Entertainment)
- 其他 (Other)

决策规则：如果初步浏览显示样本中有 ≥10% 的文档明显关于犯罪、执法、法院或监狱，则使用 6 个类别。否则保持 5 个。

### 6.2 标注数量

| 参数 | 值 |
|-----------|-------|
| 最小值 (作业要求) | 120 个文档 |
| 初始批次 | 160 个文档 (随机采样) |
| 重新采样触发条件 | 任何类别的文档 < 15 个 |
| 重新采样数量 | 每轮 20 个文档 |
| 最大重新采样轮数 | 2 轮 |
| 目标总数 | **200 个文档** |

所有重新采样均使用**随机选择**，从不针对特定类别进行选择。

### 6.3 标注工具 (`annotator.py`)

- 读取 `text_news_cleaned.jsonl`，显示 **raw_text**
- 每个标签配有键盘快捷键
- 支持撤销 (Undo) 
- 增量保存至 `data/annotations/trackA_labels.jsonl`

### 6.4 标注指南 (`docs/annotation_guideline.md`)

必须包含：
- 每个标签的定义
- 每个标签的包含/排除规则
- 如何处理混合主题的文档
- 何时分配为 `Other` 类别
- 标签集更改的修订日志

---

## 7. 阶段 3 — 任务 2：检索管道

### 7.1 共享词元生成器 (Tokenizer)

在 `src/utils/tokenizer.py` 中编写一个统一的 `tokenize(text)` 函数。
**TF-IDF 和 BM25 必须使用相同的函数。** 这保证了比较的公平性。

该 Tokenizer 应当：
- 小写化
- 处理标点符号（剥离或拆分）
- 应用一致的停用词策略
- **过滤掉 `<gap>` 词元** — 它们几乎出现在每个文档中，没有任何区分能力。包含它们会污染 IDF 分数并夸大 BM25 的 `avgdl`。

TF-IDF 集成：
```python
TfidfVectorizer(tokenizer=tokenize, preprocessor=lambda x: x, token_pattern=None)
```

BM25 集成：
```python
corpus_tokenized = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(corpus_tokenized)
```

### 7.2 基线 — Unigram TF-IDF

- 结合共享 tokenizer 的 `TfidfVectorizer`
- 仅使用 Unigram
- 使用余弦相似度进行排序

### 7.3 改进 — BM25

- 结合共享 tokenizer 的 `rank_bm25.BM25Okapi`
- 除非开发集调优证明需要修改，否则使用默认参数 (k1=1.5, b=0.75)

### 7.4 摘要 (Snippet) 提取策略

简化的感知查询摘要（不要过度设计）：

1. 找到包含最多查询词元的第一个句子
2. 如果该句子太短，则扩展到相邻的句子
3. 如果句子拆分不可靠，则回退到 30-50 个词元的窗口
4. 长度限制为最多 **250 个字符**

### 7.5 输出

**主表** — `outputs/retrieval_results/top5_evaluation.csv`:

| 字段 | 描述 |
|-------|-------------|
| Query | 查询文本 |
| Model | `tfidf` 或 `bm25` |
| Rank | 排名 1–5 |
| Doc_ID | 文档标识符 |
| Snippet | 感知查询的摘录片段 |
| Is_Relevant | `Y` / `N` / 空白 (由学生最终填写) |

**每个查询的详情文件** — `outputs/retrieval_results/per_query/query_N.json`

**总结** — `outputs/retrieval_results/summary.csv`:
- 基线胜出次数 / 改进胜出次数 / 平局次数

### 7.6 评估

- 学生手动对**至少 5 个查询**进行相关性标记（必须标记前 5 个结果中的所有 5 个）。
- `compute_p_at_5.py` 仅对完全标记的查询计算两个系统的 Precision@5 指标。

### 7.7 错误分析

从结果中挑选并记录：
- **1 个成功案例**：改进后的系统明显优于基线的查询，并解释原因。
- **1 个失败案例**：两个系统都没有返回相关结果，或者改进系统表现更差的查询，并进行分析。

---

## 8. 阶段 4 — 任务 3：分类

### 8.1 数据划分

- **仅在所有标注完成后**执行
- 采用分层抽样 `stratify=y`
- 固定随机种子 `random_state=42`
- **比例: 70 / 15 / 15** (训练集 / 开发集 / 测试集)
  - 基于 200 个文档：约 140 训练，约 30 开发，约 30 测试
  - 较大的训练集有利于小数据学习
- 将划分的索引保存到 `outputs/split.json`

### 8.2 模型

**基线模型**:
- `CountVectorizer`
- `MultinomialNB`

**改进模型**:
- `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)`
- `LinearSVC(C=1.0)`

允许在开发集上进行极简的参数调优（例如，尝试 C ∈ {0.1, 1.0, 10.0}）。不要把它变成大规模超参数搜索。

### 8.3 指标

- 准确率 (Accuracy)
- 宏平均 F1 (Macro-F1)
- 混淆矩阵 (保存为 `outputs/figures/confusion_matrix.png`)

### 8.4 错误分析 — "3种常见的错误类型"

**不要**写通用的假阳性/假阴性描述。相反应该：

1. 从混淆矩阵中识别出**前 3 个混淆对**
2. 对于每一对，展示 1-2 个具有代表性的被误分类的文档：
   - `doc_id / 真实标签 / 预测标签 / 文本摘录`
3. 解释**为什么**这种混淆是可以理解的（例如：词汇重叠、短文档、混合主题）

关于 `Other` 的特别说明：如果 `Other` 类别是一个主要的混淆源（这很有可能），请将其固有的模糊性作为 3 种错误类型之一进行讨论。`Other` 类别没有清晰的语义边界，因此预计其精确率较低 — 应该承认这一事实而不是掩盖它。

### 8.5 可选补充

- 作为鲁棒性检查的 5 折分层交叉验证结果
- 将其放在附录中，**不作为**主要评估手段
- 主要评估必须使用训练集/开发集/测试集划分（如同作业要求的一样）

---

## 9. 依赖项

```
numpy
pandas
scikit-learn
rank_bm25
matplotlib
seaborn
pyyaml
tqdm
rich
```

规范：
- 不要导入未在 `requirements.txt` 中列出的库
- 不要在 `requirements.txt` 中包含未使用的库

---

## 10. README 结构

```markdown
## Environment
pip install -r requirements.txt
Python >= 3.10

## Task 1: Preprocessing
bash scripts/run_task1.sh

## Task 2: Retrieval
1. Author finalizes data/queries.json (10 natural-language queries)
2. bash scripts/run_retrieval.sh
3. Author fills Is_Relevant in top5_evaluation.csv (≥5 queries)
4. python src/compute_p_at_5.py

## Task 3: Classification
1. Author runs annotator to label documents
   python src/utils/annotator.py
2. bash scripts/run_task3.sh

## Notes
- Author's annotation files are included for reproducibility
- src/task1_preprocess.py corresponds to the required preprocess.py
- outputs/task1_stats.md is the source for task1_stats.pdf
```

---

## 11. 最终提交清单

| 交付项 | 来源文件 | 提交名称 |
|------|--------|-----------|
| 预处理脚本 | `src/task1_preprocess.py` | `preprocess.py` |
| 语料库统计 | `outputs/task1_stats.md` | `task1_stats.pdf` |
| 清洗后的语料 | `data/processed/text_news_cleaned.txt` | `text_news_cleaned.txt` |
| 检索代码 | `src/task2_retrieval.py` | 保持原名 |
| P@5 计算脚本 | `src/compute_p_at_5.py` | 保持原名 |
| 查询文件 | `data/queries.json` | 保持原名 |
| 相关性判断 | `top5_evaluation.csv` | 保持原名 |
| 分类代码 | `src/task3_classification.py` | 保持原名 |
| 标注结果 | `data/annotations/trackA_labels.jsonl` | 保持原名 |
| 标注指南 | `docs/annotation_guideline.md` | 原名或转 PDF |
| 项目说明 | `README.md` | 保持原名 |
| 报告 | (单独撰写) | `report.pdf` |

---

## 12. 报告模板

### 任务 1
- 解析器如何工作
- 为什么选择轻量清洗
- 为什么进行 `<GAP>` 映射
- 带证据的监督 vs 无监督判断分析
- 统计数据和图表显示了什么

### 任务 2
- 为什么选择 TF-IDF 作为基线
- 为什么选择 BM25 作为改进
- 为什么两者共享一个 tokenizer
- Precision@5 结果表
- 1 个附带解释的成功案例
- 1 个附带解释的失败案例

### 任务 3
- 为什么选择 Track A
- 标签定义和标注指南摘要
- 如何划分训练集/开发集/测试集 (比例, 随机种子, 分层策略)
- 基线 vs 改进: 准确率, macro-F1, 混淆矩阵
- 3 种常见错误类型以及具体案例

### 讨论
- 在一个小型、手动标注的课程语料库上，我们强调可解释性、可复现性和定性错误分析，而不是盲目追求复杂模型。
- 承认较小测试集的局限性和标签分布的制约。

---

## 13. 执行时间表

| 时间 | 交付物 |
|-----|-------------|
| **第 1 天** | 项目骨架，完成任务 1（解析器，统计，绘图，清洗语料），标注工具，指南草案 |
| **第 2 天** | 学生敲定 10 个查询，检索管道，top-5 CSV，学生进行相关性评估 (≥5个查询)，计算 P@5，挑选成功/失败案例 |
| **第 3 天** | 学生完成 160-200 个标注，训练分类器，生成混淆矩阵，分析 3 种错误类型，完成可选的 CV 验证补充 |
| **第 4 天** | 撰写报告，验证 PDF 格式，定稿 README，组装最终提交包 |
