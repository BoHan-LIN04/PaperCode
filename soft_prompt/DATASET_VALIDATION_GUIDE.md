# 数据集验证命令使用指南

本文档说明如何使用 `validate-dataset` 命令验证论文附录 A 的数据集标签分布（表 8-16）。

## 快速开始

### 验证所有 SuperGLUE 任务

```bash
soft-prompt-repro validate-dataset
```

这会逐个验证所有 8 个 SuperGLUE 任务 (boolq, cb, copa, multirc, wic, rte) 以及 2 个 GLUE 任务 (mrpc, qqp) 的标签分布。

### 验证单个任务

```bash
# 验证 boolq 任务的训练集
soft-prompt-repro validate-dataset --task boolq --split train

# 验证 cb 任务的验证集
soft-prompt-repro validate-dataset --task cb --split validation

# 增加容差到 3%（默认 2%）
soft-prompt-repro validate-dataset --task copa --tolerance 3.0
```

## 输出示例

```
================================================================================
SUPERGLUE LABEL DISTRIBUTION VALIDATION
================================================================================

Task: boolq
================================================================================

Train Split:
✓ False              : paper=37.7% computed=37.7% (diff= 0.0%)
✓ True               : paper=62.3% computed=62.3% (diff= 0.0%)

Validation Split:
✓ False              : paper=37.8% computed=37.8% (diff= 0.0%)
✓ True               : paper=62.2% computed=62.2% (diff= 0.0%)

================================================================================
Task: cb
================================================================================

Train Split:
✓ contradiction      : paper=47.6% computed=47.6% (diff= 0.0%)
✓ entailment         : paper=46.0% computed=46.0% (diff= 0.0%)
✓ neutral            : paper= 6.4% computed= 6.4% (diff= 0.0%)

Validation Split:
✓ contradiction      : paper=50.0% computed=50.0% (diff= 0.0%)
✓ entailment         : paper=41.1% computed=41.1% (diff= 0.0%)
✓ neutral            : paper= 8.9% computed= 8.9% (diff= 0.0%)

...（省略其他任务）...

================================================================================
✓ All validations passed!
================================================================================
```

## 支持的任务

| 任务 | 数据集 | 论文表 |
|-----|-------|-------|
| boolq | super_glue | 表 8 |
| cb | super_glue | 表 9 |
| copa | super_glue | 表 10 |
| multirc | super_glue | 表 11 |
| wic | super_glue | 表 12 |
| rte | super_glue | 表 14 |
| mrpc | glue | 表 15 |
| qqp | glue | 表 16 |

## Python API 使用

```python
from soft_prompt_repro.dataset_validation import (
    compute_label_distribution,
    validate_label_distribution,
    print_label_distribution,
    validate_all_superglue,
)
from soft_prompt_repro.config import DatasetConfig

# 计算标签分布
config = DatasetConfig(task_name="boolq", dataset_name="super_glue")
dist = compute_label_distribution(config, split="train")

# 美化输出
print_label_distribution(dist)

# 与论文数据对比（返回验证结果）
results = validate_label_distribution(config, split="train", tolerance=2.0)

# 验证所有任务
validate_all_superglue()
```

## 为什么需要标签分布验证？

标签分布验证是确保数据正确加载的重要步骤，因为：

1. **数据完整性**：确认下载的数据集与论文使用的版本一致
2. **预处理验证**：确认 seq2seq 转换过程没有引入偏差
3. **可重现性**：任何数据差异会影响最终的性能结果

论文附录 A.3 表 8-16 报告了每个任务的标签分布，这是验证数据加载正确性的"黄金标准"。

## 故障排除

### 如果验证失败（标记为 ✗）

可能的原因：

1. **数据集版本不同**
   - 论文使用的是 2021 年的 SuperGLUE v1.0.2
   - 解决：确认使用 `datasets` 库版本号，可能需要下载历史版本

2. **数据预处理差异**
   - 论文的 WSC 任务有特殊处理（仅使用正确标签）
   - 解决：检查 `src/soft_prompt_repro/tasks.py` 中的处理逻辑

3. **允许范围不足**
   - 默认容差为 2%，某些任务可能需要更大容差
   - 解决：增加 `--tolerance` 参数

### 如果数据无法加载

```bash
# 手动下载数据集
python -c "from datasets import load_dataset; load_dataset('super_glue', 'boolq')"
```

这会缓存数据到本地（默认 `~/.cache/huggingface/datasets/`）。

## 代码位置

- 主实现：[src/soft_prompt_repro/dataset_validation.py](../src/soft_prompt_repro/dataset_validation.py)
- CLI 集成：[src/soft_prompt_repro/cli.py](../src/soft_prompt_repro/cli.py) (validate-dataset 子命令)
- 论文参考值：[dataset_validation.py L29-48](../src/soft_prompt_repro/dataset_validation.py#L29-L48)

## 论文引用

- 论文附录 A.3: Datasets (表 8-16), page 24-25
- 论文表 7: Dataset sizes
