# COMP 4221 NLP 课程项目

英文版说明请见 [README.md](README.md)。

本仓库完整实现了 noisy news corpus 作业要求的项目流程：

- Task 1：语料审计与轻量预处理
- Task 2：基于 unigram TF-IDF 和 BM25 的文档检索
- Task 3：六分类主题分类，采用多模型 LLM 辅助标注并由外部模型交叉审计

整个项目采用“脚本优先、可复现优先”的组织方式。一次完整运行后，作业要求的主要产物都会保存在仓库中。

## 环境准备

最稳妥的方式是使用本地虚拟环境，因为部分 macOS 自带 Python 环境不允许直接向系统解释器安装包。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

项目面向 Python 3.10+，当前仓库是在 Python 3.14 环境下验证过的。
`requirements.txt` 锁定了本次可复现实验使用的顶层依赖版本。

如果你还想冻结传递依赖和 Python 版本快照，可以执行：

```bash
bash scripts/build_lockfile.sh
```

它会额外生成：

- `requirements-lock.txt`
- `.python-version`

## 仓库结构

```text
project/
├── .python-version
├── configs/
│   ├── task3_bootstrap.yaml
│   └── task3_review_updates.yaml
├── data/
│   ├── raw/                   # 原始语料
│   ├── processed/             # 清洗后语料
│   ├── annotations/           # Task 2 qrels 与 Task 3 标签
│   └── queries.json           # 10 条检索查询
├── docs/
│   └── annotation_guideline.md
├── outputs/
│   ├── figures/
│   ├── retrieval_results/
│   ├── report.md
│   ├── report.pdf
│   ├── split.json
│   ├── task1_stats.md
│   ├── task1_stats.pdf
│   ├── task2_analysis.md
│   ├── task3_bootstrap_summary.json
│   ├── task3_external_reviews/
│   ├── task3_llm_audit_summary.json
│   ├── task3_llm_audit_summary.md
│   ├── task3_label_review_summary.json
│   ├── task3_metrics.json
│   ├── task3_predictions.csv
│   └── task3_report.md
├── requirements-lock.txt
├── scripts/
│   ├── build_lockfile.sh
│   ├── package_submission.sh
│   ├── prepare_submission.sh
│   ├── run_task3_external_audit.sh
│   ├── run_task1.sh
│   ├── run_retrieval.sh
│   ├── run_task3.sh
│   └── run_full_pipeline.sh
├── src/
│   ├── apply_task3_review_updates.py
│   ├── build_task3_llm_audit.py
│   ├── task1_preprocess.py
│   ├── task2_retrieval.py
│   ├── compute_p_at_5.py
│   ├── task3_external_review.py
│   ├── validate_retrieval_qrels.py
│   ├── verify_submission.py
│   ├── build_task2_analysis.py
│   ├── build_final_report.py
│   ├── task3_classification.py
│   └── utils/
└── preprocess.py              # 兼容作业要求的入口文件
```

## 复现实验结果

以下命令均在仓库根目录执行。

### 一键全流程运行

```bash
bash scripts/run_full_pipeline.sh
```

该脚本会依次执行 Task 1、Task 2、Task 3，并重建最终总报告。
注意：外部大模型审计是单独流程，因为它依赖第三方 API。

### Task 1：预处理与语料审计

```bash
bash scripts/run_task1.sh
```

主要产物：

- `data/processed/text_news_cleaned.jsonl`
- `data/processed/text_news_cleaned.txt`
- `outputs/task1_stats.md`
- `outputs/task1_stats.pdf`
- `outputs/figures/doc_length_hist.png`
- `outputs/figures/zipf_plot.png`

说明：

- 文档以 `@@<数字ID>` 作为切分标记。
- 清洗策略是轻量、非破坏性的：将重复缺口噪声统一映射为 `<GAP>`，统一小写，归一化空白。
- `task1_stats.pdf` 直接由 Python 生成，不依赖外部 PDF engine。

### Task 2：检索

```bash
bash scripts/run_retrieval.sh
```

主要产物：

- `data/queries.json`
- `data/annotations/retrieval_qrels.csv`
- `outputs/retrieval_results/top5_evaluation.csv`
- `outputs/retrieval_results/per_query/*.json`
- `outputs/retrieval_results/qrels_validation.json`
- `outputs/retrieval_results/qrels_validation.md`
- `outputs/retrieval_results/per_query_scores.csv`
- `outputs/retrieval_results/summary.csv`
- `outputs/task2_analysis.md`

说明：

- TF-IDF 与 BM25 共用 `src/utils/tokenizer.py` 中的同一套分词逻辑。
- 该分词器统一小写、稳定切词，并只去除 `<GAP>` 占位符。
- canonical relevance judgments 保存在 `data/annotations/retrieval_qrels.csv` 中，使用模型无关的 query-document 形式。
- `scripts/run_retrieval.sh` 会顺序执行检索、qrels 校验、P@5 计算和 Task 2 分析生成。
- snippet 是 query-aware 的，长度上限为 250 字符。

### Task 3：主题分类

```bash
bash scripts/run_task3.sh
```

主要产物：

- `data/annotations/trackA_labels.jsonl`
- `outputs/task3_label_review_summary.json`
- `outputs/split.json`
- `outputs/figures/confusion_matrix.png`
- `outputs/task3_metrics.json`
- `outputs/task3_predictions.csv`
- `outputs/task3_report.md`

可选交互式标注工具：

```bash
python3 -m src.utils.annotator --sample-size 20
```

说明：

- `scripts/run_task3.sh` 会直接在当前发布标签集上训练与评估，不会重新覆盖标签。
- 早期 bootstrap 方案仍保留在 `configs/task3_bootstrap.yaml` 中，但它不再是最终标签的权威来源。
- 若要单独重建 bootstrap seed labels，可运行：

```bash
bash scripts/bootstrap_task3_labels.sh
```

- 标签集合固定为：
  - `Politics`
  - `Business/Economy`
  - `Sports`
  - `Crime/Law`
  - `Culture/Entertainment`
  - `Other`
- 当前发布标签集经历了三阶段处理：
  - 第一阶段：BM25 关键词匹配生成初始种子标签
  - 第二阶段：GPT-5.4（Codex）逐篇阅读全文后确认或修正标签
  - 第三阶段：Gemini 2.5 Flash 和 Grok 4.20 独立审计全部 200 条标签，一致异议样本被重新裁决
- 这是一套多模型 LLM 辅助标注流水线，而非传统人工标注。其优势是可复现、可审计、有交叉质量检查；局限在于所有标注者都是 LLM，可能存在共享的系统偏差。
- 数据切分固定为 stratified `70/15/15`，随机种子 `42`。
- baseline 模型为 `CountVectorizer + MultinomialNB`。
- improved 模型为 `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True) + LinearSVC`。
- **标签噪声鲁棒性测试**：随机翻转 5–30% 的训练标签后重训，确认 macro-F1 单调下降（0.933 → 0.853），验证分类器学到了真实的主题信号。
- **逐类 F1 分析**：展示每个类别的 F1 分数，识别 bag-of-words 分类器最难区分的类别。
- **系统性错误模式分析**：从误分类文档中提取主导 TF-IDF 关键词，合并 test + dev 错误样本，提供每对混淆类别的领域级解释。

### 可选：Task 3 外部大模型审计

这个流程会调用外部厂商模型对当前发布标签进行独立二次审查，用于发现可疑标签，但不会直接把机器判断当成 gold labels。

设置 API Key 后执行：

```bash
export VERTEX_AI_API_KEY="..."
export XAI_API_KEY="..."
bash scripts/run_task3_external_audit.sh
```

主要产物：

- `outputs/task3_external_reviews/vertex.jsonl`
- `outputs/task3_external_reviews/xai.jsonl`
- `outputs/task3_llm_audit_summary.json`
- `outputs/task3_llm_audit_summary.md`

说明：

- 该流程不会覆盖正式标签文件。
- `src/task3_external_review.py` 支持 `--limit`、`--doc-ids`、`--resume`。
- API Key 只应通过环境变量提供，不应提交进仓库。
- 当前发布标签已经吸收了这轮外部审计中的一致异议裁决，因此最新 audit summary 中剩余的一致异议应为 `0`。

## 当前结果快照

### Task 1

- 文档数：`871`
- 清洗后总 token 数：`1,147,881`
- 平均文档长度：`1317.89`
- 中位文档长度：`1250`
- 清洗前词表大小：`57,401`
- 清洗后词表大小：`47,998`

### Task 2

- TF-IDF 平均 `Precision@5`：`0.76`
- BM25 平均 `Precision@5`：`0.78`
- baseline 胜出次数：`3`
- improved 胜出次数：`3`
- 平局：`4`

### Task 3

- baseline test accuracy：`0.8667`
- baseline test macro-F1：`0.8561`
- improved test accuracy：`0.9333`
- improved test macro-F1：`0.9330`

### Task 3 外部审计

- 跨 provider 一致率：`0.905`
- 对当前发布标签集的剩余一致异议：`0`

## 最终总报告

```bash
$PYTHON -m src.build_final_report
```

生成：

- `outputs/report.md`
- `outputs/report.pdf`

该报告将 Task 1、Task 2、Task 3 的最新结果整合为一个提交友好的总报告。

## 提交准备

如果你想重建所有结果、清理缓存、校验交付文件并生成干净的提交目录：

```bash
bash scripts/prepare_submission.sh
```

也可以单独做校验：

```bash
$PYTHON -m src.verify_submission --root .
$PYTHON -m src.verify_submission --root submission
```

若要在 `submission/` 准备好之后再打一个可上传压缩包：

```bash
bash scripts/package_submission.sh
```

## 作业交付物映射

| 作业项 | 仓库中的文件 |
| --- | --- |
| `preprocess.py` | `preprocess.py` |
| `task1_stats.pdf` | `outputs/task1_stats.pdf` |
| `text_news_cleaned.txt` | `data/processed/text_news_cleaned.txt` |
| Retrieval code | `src/task2_retrieval.py` |
| Precision@5 script | `src/compute_p_at_5.py` |
| Qrels validation script | `src/validate_retrieval_qrels.py` |
| Queries | `data/queries.json` |
| Relevance judgments | `data/annotations/retrieval_qrels.csv` |
| Final report | `outputs/report.pdf` |
| Classification code | `src/task3_classification.py` |
| Annotations | `data/annotations/trackA_labels.jsonl` |
| Annotation guideline | `docs/annotation_guideline.md` |
| 可选外部审计 | `src/task3_external_review.py`, `src/build_task3_llm_audit.py` |

## 实现说明

- `sample.txt` 已复制到 `data/raw/` 中，方便与原始作业环境保持一致。
- 检索 judgments 采用 query-document 粒度的 canonical qrels，因此重复跑排序时不会依赖旧结果顺序。
- qrels 会自动和当前 top-5 结果进行校验，校验结果保存在 `outputs/retrieval_results/` 下。
- Task 3 标签经过"BM25 种子 + GPT-5.4 复核 + 外部多模型交叉审计"三阶段流水线，是一种 LLM 辅助标注方案。其优势是可复现、可审计；局限在于不等价于人工 gold annotation，且 LLM 标签可能对词汇特征有系统偏好，进而使下游 bag-of-words 分类器的准确率偏高。
- 早期 Task 3 bootstrap 参数仍保存在 `configs/task3_bootstrap.yaml` 中，仅作为历史和对照用途。
- 可选外部审计流程会把 provider-specific 结果单独存储，不会覆盖正式标签。
- `requirements.txt` 锁顶层依赖，`requirements-lock.txt` 与 `.python-version` 提供更强的环境冻结能力。
- 所有 shell 脚本通过 `scripts/detect_python.sh` 自动检测 Python ≥ 3.10 解释器（依次检查 venv、Homebrew、系统 Python）。如需指定，可设环境变量 `PYTHON=/path/to/python3`。
- `scripts/prepare_submission.sh` 会在重建产物后生成干净的 `submission/` 目录。
- 整个项目尽量保持脚本化和可复现，不依赖 notebook。
- Task 2 分析包含 **逐查询案例研究**：展示 BM25 和 TF-IDF 各自优势最大的查询的 top-3 检索结果，并附带基于实际文档内容的模型优劣分析。
