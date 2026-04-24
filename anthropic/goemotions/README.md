# goemotions 向量构建流程（完全复刻 anthropic 主线）

本流程指导如何用 goemotions 数据集，复刻 anthropic 风格的情绪向量构建产物（emotion_vectors_raw.npy, emotion_vectors_orth.npy, vector_metadata.json），用于 soft prompt 初始化、分析等。

---
scp -r c:/Users/xiexi/PaperCode/anthropic/goemotions xinyix888@engr-liu01s.bluecat.arizona.edu:~/

## 目录结构

```
goemotions/
  data/
    goemotions_train.jsonl
    ...
  pipeline/
    extract_residuals_goemotions.py
    build_emotion_vectors_goemotions.py
  artifacts/
    activation_cache/
    emotion_vectors/
```

---

## 步骤一：准备数据

- 确保 data/goemotions_train.jsonl 为标准格式：
  - 每行：{"input": "...", "target": "标签"}
  - 标签为 goemotions 28类之一

---

## 步骤二：提取 residual（激活缓存）

```bash
cd anthropic/goemotions/pipeline
python extract_residuals_goemotions.py
```
- 产物会保存在 ../artifacts/activation_cache/
- 包括 activations.npy, sample_ids.npy, token_positions.npy, sample_index.parquet

---

## 步骤三：构建情绪向量

```bash
python build_emotion_vectors_goemotions.py
```
- 产物会保存在 ../artifacts/emotion_vectors/
- 包括 emotion_vectors_raw.npy, emotion_vectors_orth.npy, vector_metadata.json

---

## 参数说明
- 默认抽取 Qwen3-14B 最后一层 hidden state，token_pool_start=0
- neutral 去噪采用自身 neutral 类样本
- emotion_names 顺序自动与 goemotions 标签对齐

---

## 下游兼容
- 产物结构与 anthropic/artifacts/05_emotion_vectors/intermediate/ 完全一致
- 可直接用于 soft prompt 初始化、分析、可视化等

---

## 常见问题
- 如需适配验证集/其他分割，只需修改数据路径和脚本参数
- 如需更换模型或层号，修改脚本 MODEL_NAME/LAYER_IDX

---

如有问题可随时补充！