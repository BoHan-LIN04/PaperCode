# GoEmotions 适配说明

本目录已适配 goemotions 数据集，数据预处理脚本为 `preprocess_goemotions.py`，生成的训练/测试集为：
- data/goemotions_train.jsonl
- data/goemotions_test.jsonl

配置文件：
- configs/goemotions_soft_prompt.yaml

可直接用于 soft prompt 训练与评测。

## 训练命令示例

```bash
python src/train.py --config configs/goemotions_soft_prompt.yaml
```

## 说明
- input/target 字段已标准化。
- emotion_names 与 goemotions 标签完全对齐。
- emotion_vectors_path/metadata_path 请根据实际路径调整。

## 数据集用途说明

goemotions 数据集既可用于 soft prompt 微调训练，也常作为验证/评测集，重点在于“公开、标准、覆盖广”，适合做方法有效性和泛化能力的客观评测。

- 训练：可用作 soft prompt 微调训练集，帮助模型学习情感风格控制。
- 验证/评测：更常用作验证集或测试集，评估模型在真实、公开的情感分类任务上的泛化能力和情感可控生成效果。
- 对比实验：可与 anthropic 自建数据、emotion24 等其他情感数据集对比，检验方法的通用性和鲁棒性。

结论：推荐将 goemotions 作为主要验证/评测集，训练时可选用全部或部分数据，具体视实验设计而定。
