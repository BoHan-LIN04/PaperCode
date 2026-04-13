# anthropic-emotions-repro

`/opt/data/private/lbh/anthropic` 现在是一个最小主线研究仓库，只做两件事：

1. 用仓库内固定的 `100` 个话题和手工 prompt 模板，批量生成情绪故事与中性故事。
2. 在 `Qwen3` 指定层提取 residual stream，并构建情绪向量。

这个版本明确摒弃外部数据集，只保留最小主线，不再包含任何后续扩展分析或额外评测模块。

## 当前主线

- `prepare_topic_bank`
- `prepare_prompt_templates`
- `generate_emotion_corpus`
- `extract_residuals`
- `build_emotion_vectors`
- `build_report`

## 核心输入

- [topics_100.yaml](/opt/data/private/lbh/anthropic/datasets/topics_100.yaml)
- [story_templates.yaml](/opt/data/private/lbh/anthropic/prompts/story_templates.yaml)
- [emotion_constraints.yaml](/opt/data/private/lbh/anthropic/datasets/emotion_constraints.yaml)

## 快速开始

```bash
cd /opt/data/private/lbh/anthropic
bash scripts/bootstrap_env.sh
bash scripts/run_step.sh prepare_topic_bank configs/smoke.yaml
bash scripts/run_step.sh prepare_prompt_templates configs/smoke.yaml
bash scripts/run_step.sh generate_emotion_corpus configs/smoke.yaml
bash scripts/run_step.sh extract_residuals configs/smoke.yaml
bash scripts/run_step.sh build_emotion_vectors configs/smoke.yaml
bash scripts/run_step.sh build_report configs/smoke.yaml
```

## 目录原则

- 所有实验产物都落到单一 artifact 根目录下
- 每个步骤目录都自动生成 `README.md`、`manifest.json`、`metrics.json` 和标准子目录
- 最终报告只总结 topic bank、模板、故事语料、activation cache 和 emotion vectors
