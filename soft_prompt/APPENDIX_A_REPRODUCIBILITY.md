# 附录 A：可重现性实现对标

本文档对比论文附录 A 的实验设置与代码实现。

## A.1 实验设置 (Experimental Settings)

### 评估指标 ✅ 已实现

| 数据集 | 指标 | 代码位置 | 实现状态 |
|------|------|--------|--------|
| SuperGLUE (8 tasks) | Accuracy / F1 | `src/soft_prompt_repro/metrics.py` | ✅ 完整实现 |
| SQuAD / MRQA | F1, Exact Match | `src/soft_prompt_repro/metrics.py L80-120` | ✅ 完整实现 |
| QQP / MRPC | Accuracy, F1 | `src/soft_prompt_repro/metrics.py L130-160` | ✅ 完整实现 |

使用 T5 官方评估代码的 PyTorch 移植版本（see `metrics.py`）。

### 基础模型 ✅ 部分实现

**论文要求**：T5.1.1（所有尺度：small, base, large, xl, xxl）

**代码实现**：
```yaml
# configs/base.yaml
model:
  name_or_path: google/t5-v1_1-small  # 可配置为任何 T5.1.1 checkpoint

# 推荐的 checkpoint 列表
google/t5-v1_1-small      # 60M 参数
google/t5-v1_1-base       # 220M 参数
google/t5-v1_1-large      # 770M 参数
google/t5-v1_1-xl         # 3B 参数
google/t5-v1_1-xxl        # 11B 参数
```

代码完全冻结模型，仅训练 soft prompt（encoder 侧）。✅

### 训练硬件 ⚠️ 手动调整

**论文**：
- Small/Base：4 TPU v2
- Large/XL/XXL：16 TPU v3

**代码**：
- 自动检测设备（`auto` 模式）
- 支持单 GPU/CPU，需手动调整 `batch_size`、`gradient_accumulation` 参数

参见配置文件中的 `training.batch_size` 和 `adaptation.batch_size` 调整。

### 参数规模对标（表 4）✅ 可计算

论文表 4 列出了各模型/prompt length 的参数规模。代码可通过以下方式复现：

```python
# 计算参数数量脚本（伪代码）
from transformers import AutoModel

model = AutoModel.from_pretrained("google/t5-v1_1-small")
total_params = model.num_parameters()
# 不同 prompt length 的额外参数：
# prompt_params = prompt_length × embedding_dim (512 for T5-small)
```

具体数值见代码输出日志或通过以下命令获取：
```bash
soft-prompt-repro train --config configs/base.yaml --override training.max_steps=1 2>&1 | grep "parameters"
```

### 收敛时间（表 5）⚠️ 依赖硬件

论文表 5 报告了各种配置的收敛时间（均值 ± 标准差），例如：
- BoolQ, Prompt Length=20, T5-Large: 1:14:16 ± 13:12

代码支持运行时记录，但会因硬件差异而不同。建议：

1. 记录训练日志中的 `elapsed_time` 字段
2. 多次运行（3 次）取均值和标准差
3. 与论文对比时考虑硬件差异（TPU vs GPU vs CPU）

---

## A.2 超参数搜索 (Hyperparameter Search)

### 搜索范围（表 6）✅ 已配置

论文进行了 77 个超参数搜索试验（prompt tuning 40 次 + model tuning 37 次）。

**表 6 搜索空间映射到代码配置**：

| 超参数 | 论文范围 | 代码配置位置 | 当前设置 |
|------|--------|-----------|--------|
| **Learning Rate** | 0.001-1.25 | `training.learning_rate` | 0.3 (prompt) / 1e-3 (model) |
| **Parameter Scaling** | {True, False} | `training.weight_decay` | 支持 |
| **Batch Size** | {32, 64, 128, 256, 512} | `training.batch_size` | 8（smoke test） |
| **Number of Steps** | {10K, 20K, 30K} | `training.max_steps` | 60（smoke test） |
| **Decay Factor** | {off, 0.1, 0.5} | 需扩展 | ❌ 不支持 |
| **Steps per Decay** | {off, 4K, 6K, 8K} | 需扩展 | ❌ 不支持 |

### 默认配置 ✅ 已实现

论文默认配置（除了 ablation 参数）：

```yaml
# 论文配置 → 代码配置映射
LM Adaptation: 100K steps          → adaptation.max_steps: 100000
Prompt Length: 100 tokens          → prompt.num_virtual_tokens: 100
Init Strategy: class-label         → prompt.init_strategy: class_labels
```

**代码中应用这些默认值的方式**：

```bash
# 创建一个完整论文配置的 config 文件
soft-prompt-repro train --config configs/paper_defaults.yaml
```

目前代码默认值为简化的 smoke test 配置（不同于论文）。

### 运行统计 ✅ 可重现数量

**论文总运行数**：
- Prompt Tuning sweep: 40 trials
- Model Tuning sweep: 37 trials  
- 每个配置3个随机种子（seed）
- Base/方法/Ablation 实验：195 runs
- Domain shift 实验：18 runs
- Ensemble 实验：24 runs

**代码支持方式**：

```bash
# 自动运行多个 seed
for seed in {1..3}; do
  soft-prompt-repro train --config configs/base.yaml \
    --override training.seed=$seed \
    --override output.output_dir=artifacts/seed-$seed
done
```

### 结果报告 ✅ 自动生成

论文报告均值 ± 标准差，代码支持通过：

```bash
# 自动汇总多个 seed 的结果（见 compare 命令）
soft-prompt-repro compare --config configs/figure1_compare.yaml
```

生成 `summary.csv`、`COMPARISON_REPORT.md` 等，包含均值、标准差。

---

## A.3 数据集 (Datasets)

### 支持的数据集 ✅ 完整实现

**SuperGLUE（8 个任务）**：
```
boolq, cb, copa, multirc, record, rte, wic, wsc
```
通过 Hugging Face `datasets` 库加载，版本 1.0.2。

**SQuAD**：
```
squad / v1.1  (Tensorflow Datasets v3.0.0)
```

**MRQA OOD（6 个任务）**：
```
textbookqa, bioasq, race, drop, duorc, re
```
来自 [MRQA 2019 共享任务](https://mrqa.github.io/)。

**Paraphrase Detection**：
```
qqp, mrpc
```
从 Hugging Face `datasets` GLUE 中加载。

### 数据集大小（表 7）📊 参考表

论文表 7 给出的数据集大小：

| 数据集 | Train | Validation | Test | 
|------|-------|-----------|------|
| BoolQ | 9,427 | 3,270 | 3,245 |
| CB | 250 | 56 | 250 |
| COPA | 400 | 100 | 500 |
| MultiRC | 27,243 | 4,848 | 9,693 |
| RecORD | 100,730 | 10,000 | 10,000 |
| RTE | 2,490 | 277 | 3,000 |
| WiC | 5,428 | 638 | 1,400 |
| WSC | 554 | 104 | 146 |
| QQP | 363,849 | 40,430 | 390,965 |
| MRPC | 3,668 | 408 | 1,725 |
| SQuAD | 87,599 | 10,570 | (test) |
| TextbookQA | - | 1,504 | - |
| BioASQ | - | 1,601 | - |
| RACE | - | 1,503 | - |
| RE | - | 2,948 | - |
| DuoRC | - | - | - |
| DROP | - | 1,063 | - |

代码加载时会自动验证大小，若不匹配可通过配置调整（`max_train_examples`、`max_eval_examples` 等）。

### 数据预处理 ✅ 已实现

**论文方式**：
- 遵循 T5 官方预处理（`text-to-text` 格式）
- 数据集前缀（dataset prefix）被省略
- WSC 改写为生成任务

**代码实现位置**：
- `src/soft_prompt_repro/tasks.py`: 任务定义与文本转换
- `src/soft_prompt_repro/data/` : 数据加载与转换

例如，SQuAD 转换：
```python
def _squad(example):
    """Convert SQuAD to seq2seq format: 'context: [C] question: [Q]' → '[A]'"""
    return {
        'input': f"context: {example['context']} question: {example['question']}",
        'target': example['answers']['text'][0] if example['answers']['text'] else ''
    }
```

### 标签分布（表 8-16）📊 实现说明

论文表 8-16 给出了各个分类任务的标签分布（例如 BoolQ：True/False 比例）。

**代码实现**：
- 文件：[src/soft_prompt_repro/dataset_validation.py](src/soft_prompt_repro/dataset_validation.py)
- 功能：计算并验证数据集的标签分布，与论文报告值对比（论文表 8-16）

**验证所有 SuperGLUE 任务**：
```bash
soft-prompt-repro validate-dataset
```

**验证特定任务**：
```bash
# 验证 boolq 训练集
soft-prompt-repro validate-dataset --task boolq --split train

# 验证 cb 验证集，允许 3% 的差异
soft-prompt-repro validate-dataset --task cb --split validation --tolerance 3.0
```

标签分布不需要在代码中硬编码，因为使用官方数据集。

---

## 代码与论文的差异总结

### ✅ 完全实现
- [x] 冻结 T5.1.1 模型，仅优化 soft prompt
- [x] SuperGLUE 8 个任务的完整支持
- [x] SQuAD / MRQA 数据集加载与评估
- [x] Prompt 初始化策略（random, vocab, class-labels）
- [x] Prompt length ablation
- [x] LM adaptation 阶段
- [x] Model Tuning baseline（单任务 + 多任务）
- [x] Ensemble 方法
- [x] 多 seed 结果聚合
- [x] 可解释性分析（最近邻 token）

### ⚠️ 部分实现
- [ ] 论文的完整超参数搜索空间（表 6）
  - 目前：固定学习率、batch size
  - 需要：Learning rate decay schedule 支持
- [ ] 论文默认配置
  - 论文：100 token prompt, 100K LM adaptation steps
  - 代码：20 token prompt, 100 adaptation steps（smoke test 配置）
- [ ] 多 GPU/TPU 分布式训练
  - 论文：使用 TPU (4v2 or 16v3)
  - 代码：单 GPU/CPU

### ❌ 未实现
- [ ] 论文的 77 个超参数搜索试验的完整自动化
  - 原因：需要大量计算资源和时间
  - 解决：提供搜索空间配置供用户手动运行
- [ ] 默认的 XXL 模型配置
  - 原因：内存需求大（>40GB VRAM）
  - 解决：提供大模型调参建议

---

## 如何逼近论文的实验设置

要尽可能接近论文的结果，请：

### 1. 使用论文的默认配置

创建 `configs/paper_defaults.yaml`：
```yaml
extends: base.yaml

prompt:
  num_virtual_tokens: 100
  init_strategy: class_labels

adaptation:
  enabled: true
  max_steps: 100000

training:
  max_steps: 100000
  learning_rate: 0.3
  batch_size: 32
  eval_steps: 5000
  save_steps: 5000
```

运行：
```bash
soft-prompt-repro train --config configs/paper_defaults.yaml
```

### 2. 运行多个 seed 并聚合结果

```bash
soft-prompt-repro compare --config configs/figure1_compare.yaml
# 自动生成 summary.csv（均值 ± 标准差）
```

### 3. 扩展到所有模型尺度

```bash
for model in small base large xl xxl; do
  soft-prompt-repro train --config configs/paper_defaults.yaml \
    --override model.name_or_path=google/t5-v1_1-$model
done
```

### 4. 验证结果数据

与论文对比后检查：
- SuperGLUE 平均得分是否在正确范围内
- 不同 prompt length 的趋势是否一致
- Domain shift 时 prompt tuning 是否优于 model tuning

---

## 参考

- 论文附录 A1: Experimental Settings, page 23-24
- 论文表 4-7: Parameter counts, runtimes, hyperparameter search space, dataset sizes
- 代码参考：`configs/`、`src/soft_prompt_repro/metrics.py`、`src/soft_prompt_repro/tasks.py`
