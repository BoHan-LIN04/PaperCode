# decoder-soft-prompt-repro

这个目录是一个新的 decoder-only soft prompt tuning 脚手架。

设计目标：

- 参考 `prompt-tuning` 的核心思路：冻结底座模型，只训练前插的 soft prompt
- 参考 `soft_prompt` 的工程风格：PyTorch + Transformers + YAML config + 简单 CLI
- 面向 decoder-only / causal LM，例如 Qwen、LLaMA、Mistral、GPT 类模型

它不是在 `prompt-tuning` 上硬改 T5X/JAX，而是单独实现一套更适合 causal LM 的版本。

## 当前实现范围

- 冻结 `AutoModelForCausalLM`
- 训练一个形状为 `[num_virtual_tokens, hidden_size]` 的 soft prompt
- 把 soft prompt 作为 embedding 前缀拼到输入前面
- 支持本地 JSONL 数据集训练与评估
- 支持从外部 `.pt` / `.npy` 文件初始化 prompt
- 支持从 `emotion_vectors_orth.npy` 按情绪名选择向量并直接初始化 prompt
- 支持可选投影矩阵，把 emotion vector 从源模型空间映射到目标 causal LM hidden space
- 支持通过独立脚本拟合并导出线性投影矩阵

## 数据格式

训练和验证都使用 JSONL。每行至少包含两列：

```json
{"input": "Question: ...\nAnswer:", "target": "..."}
```

默认字段名是 `input` 和 `target`，也可以在配置里改。

## 快速开始

## 最小配置模板

如果你只想先跑通主线，最少只改下面这 5 个字段：

```yaml
model:
	name_or_path: Qwen/Qwen3-14B

dataset:
	train_file: data/emotion24_train.jsonl
	eval_file: data/emotion24_eval.jsonl

prompt:
	init_strategy: emotion_vectors
	emotion_vector_route: same_model

output:
	output_dir: artifacts/my-run
```

更直接一点，推荐直接从现成配置复制：

- [configs/anthropic_qwen3_14b_emotion24_task.yaml](configs/anthropic_qwen3_14b_emotion24_task.yaml)

训练前先做一次配置预检：

```bash
decoder-soft-prompt-repro validate-config --config configs/anthropic_qwen3_14b_emotion24_task.yaml
```

这个命令会在真正训练前检查：

- route 是否和配置一致
- 数据文件路径是否存在
- emotion vector / metadata / projection 路径是否存在
- 向量 shape 是否和模型 hidden size 对得上
- tokenizer / model 是否能正常加载

### 1. 安装

```bash
cd decoder_soft_prompt
pip install -e .[dev]
```

### 2. 配置

编辑 [configs/base.yaml](configs/base.yaml)，至少改这几项：

- `model.name_or_path`
- `dataset.train_file`
- `dataset.eval_file`
- `output.output_dir`

### 3. 训练

```bash
decoder-soft-prompt-repro train --config configs/base.yaml
```

### 4. 评估已有 prompt

```bash
decoder-soft-prompt-repro eval --config configs/base.yaml --prompt-path artifacts/base/best_prompt.pt
```

## 当前实验结论


## 实验结论（Prompt Tuning 纵向搜索）

### 纵向搜索结果

- 以 emotion_vectors 初始化、24 token、lr=0.005 为基础，分别尝试 max_steps=300、500、800：
	- 300 步时，eval_loss 最优为 2.2174（step 75），此时 exact_match 仍为 0.0。
	- 500 步时，eval_loss 最优为 2.2435（step 200），未优于 300 步。
	- 800 步实验未见进一步下降，loss 有波动但无提升。

### 结论与推荐

- 300 步为推荐/默认训练预算，继续增加步数无明显收益。
- 当前最佳 checkpoint 出现在 step 75（eval_loss=2.2174）。
- best checkpoint 选择逻辑已修正为：优先 exact_match，若相同则取 eval_loss 更低者。

后续实验建议直接采用 300 步配置。

---

下面这些结论来自已经实际跑完的 `Qwen3-14B -> Qwen3-14B` 同模型路线实验，而不是只基于配置推测。

### 1. 已验证主线可跑通

下面这条链路已经完成端到端验证：

- `../anthropic/artifacts/05_emotion_vectors/intermediate/emotion_vectors_orth.npy`
- `prompt.init_strategy: emotion_vectors`
- `prompt.emotion_vector_route: same_model`
- `Qwen/Qwen3-14B` 作为 decoder-only 底座
- `bfloat16` 训练

也就是说，当前仓库的默认主线不是概念验证，而是已经能稳定完成：模型加载、emotion prompt 初始化、训练、评估、保存 `best_prompt.pt`。

### 2. 初始化策略对比结论

在相同任务、相同模型、相同训练步数下，三组初始化 baseline 的三随机种子平均 `eval loss` 如下：

- `emotion_vectors`: `2.3022`
- `random_uniform`: `2.4865`
- `sampled_vocab`: `2.5721`

当前结论：

- `emotion_vectors` 是这三个初始化里平均表现最好的方案
- `random_uniform` 偶尔能跑出不错单次结果，但均值和稳定性都不如 `emotion_vectors`
- `sampled_vocab` 在当前任务上是三者里最差的 baseline

因此，当前仓库推荐把 `emotion_vectors` 作为默认初始化策略，而不是 `random_uniform` 或 `sampled_vocab`。

### 3. Prompt 长度和学习率结论

围绕 `emotion_vectors` 初始化，又做了 prompt 长度和学习率搜索。

先看 prompt 长度：

- `num_virtual_tokens=24` 明显优于 `32` 和 `48`
- 在当前数据规模和 300 steps 设置下，更长 prompt 没有带来收益，反而更容易退化

再看学习率：

- 单个 seed 上，`learning_rate=0.01` 一度看起来最好
- 但三随机种子确认后，`learning_rate=0.005` 的均值更低，也更稳定

三随机种子确认结果：

- `tok24 + lr=0.005`: mean `2.3022`
- `tok24 + lr=0.01`: mean `2.3459`

因此，当前默认主实验配置应当选更稳的 `0.005`，而不是单次最优的 `0.01`。

### 4. 当前推荐默认实验配置

如果你不是在做新的 ablation，而是想直接复现当前最稳的主实验设置，推荐使用：

- 模型：`Qwen/Qwen3-14B`
- 路线：`same_model`
- 初始化：`emotion_vectors`
- `num_virtual_tokens=24`
- `learning_rate=0.005`
- `max_steps=300`
- `batch_size=1`
- `eval_batch_size=1`
- `model.torch_dtype=bfloat16`

默认主实验配置文件：

- [configs/anthropic_qwen3_14b_emotion24_task.yaml](configs/anthropic_qwen3_14b_emotion24_task.yaml)

这个配置现在已经固定为：`emotion_vectors + tok24 + lr0.005 + 300 steps`。

推荐命令：

```bash
decoder-soft-prompt-repro train \
	--config configs/anthropic_qwen3_14b_emotion24_task.yaml \
	--override model.torch_dtype=bfloat16
```

---

## 结果可视化与画图

本仓库已内置 `src/decoder_soft_prompt_repro/plotting.py`，可用于画主图（如 Figure 1）、ablation 曲线等。

**用法示例：**

1. 先准备 summary.csv，需包含至少如下字段：
   - `model_params` 或 `step`（横轴）
   - `mean_score` 或 `eval_loss`（纵轴）
   - `std_score`（可选，误差带）
   - `method`（可选，多曲线分组）

2. 在 notebook 或脚本中调用：

```python
from decoder_soft_prompt_repro.plotting import plot_figure_from_csv
plot_figure_from_csv("summary.csv", "output.png", title="Prompt Tuning Results")
```

3. 画图后可直接插入报告或对比不同实验。

如需批量 sweep/ablation 可视化，建议将 sweep 结果整理为统一 csv 后复用本脚本。

---

注意：

- 这个配置文件已经不再是短程 smoke 配置，而是当前推荐的正式实验默认值
- 如果你只想先做 very short smoke run，再用 override 临时把 `training.max_steps` 改小即可

### 5. 纵向搜索建议：只继续试更长训练步数

在当前实验结果下，不建议再继续扩 prompt 长度，也不建议再继续横向扫更大的学习率范围。

更合理的下一步是固定默认配置，只做少量纵向搜索，观察更长训练是否还能继续降低 loss：

- [configs/anthropic_qwen3_14b_emotion24_task.yaml](configs/anthropic_qwen3_14b_emotion24_task.yaml): 默认正式实验，`300 steps`
- [configs/anthropic_qwen3_14b_emotion24_task_500.yaml](configs/anthropic_qwen3_14b_emotion24_task_500.yaml): 纵向搜索，`500 steps`
- [configs/anthropic_qwen3_14b_emotion24_task_800.yaml](configs/anthropic_qwen3_14b_emotion24_task_800.yaml): 纵向搜索，`800 steps`

推荐命令：

```bash
decoder-soft-prompt-repro train \
	--config configs/anthropic_qwen3_14b_emotion24_task_500.yaml \
	--override model.torch_dtype=bfloat16
```

```bash
decoder-soft-prompt-repro train \
	--config configs/anthropic_qwen3_14b_emotion24_task_800.yaml \
	--override model.torch_dtype=bfloat16
```

如果 `500` 和 `800` steps 的 eval loss 继续下降，再决定是否继续把默认训练步数上调；如果收益已经明显变小，就保留 `300 steps` 作为更省算力的默认配置。

## 推荐工作流

如果你当前的目标是把 `anthropic` 目录里产出的 `emotion_vectors_orth.npy` 接到 decoder-only soft prompt 训练里，建议按下面两条路径理解：

### 默认推荐路线：Qwen3-14B -> Qwen3-14B

这是当前仓库里最推荐的主线。

原因很简单：

- `anthropic` 里的 emotion vectors 就是从 `Qwen/Qwen3-14B` 提出来的
- `decoder_soft_prompt` 这边如果也用 `Qwen/Qwen3-14B`，hidden space 是同一个空间
- 不需要投影矩阵
- 初始化误差最小，实验解释也最直接

如果你只是想先把整条链路跑通，优先走这一条，不建议一开始就做跨模型迁移。

对应配置：

- [configs/anthropic_qwen3_14b_emotion24_task.yaml](configs/anthropic_qwen3_14b_emotion24_task.yaml)

直接训练：

```bash
decoder-soft-prompt-repro train --config configs/anthropic_qwen3_14b_emotion24_task.yaml
```

评估已有 prompt：

```bash
decoder-soft-prompt-repro eval \
	--config configs/anthropic_qwen3_14b_emotion24_task.yaml \
	--prompt-path artifacts/anthropic-qwen3-14b-emotion24-task/best_prompt.pt
```

### 路径 A：直接匹配 hidden size

适用条件：

- emotion vector 的维度和底座 causal LM hidden size 完全一致
- 例如当前仓库里的 `Qwen3-14B emotion vectors -> Qwen3-14B decoder-only model`

这种情况下，不需要投影矩阵，直接把 emotion vectors 当作 prompt 初始化即可。

### 路径 B：先拟合投影矩阵

适用条件：

- emotion vector 的维度和目标模型 hidden size 不一致
- 例如当前仓库里的 `Qwen3-14B emotion vectors -> Qwen3-0.6B decoder-only model`

这种情况下，先用一批文本同时抽取源模型和目标模型的表示，拟合一个线性投影矩阵，再用这个矩阵把 emotion vectors 映射到目标模型空间。

### 可选路线：Qwen3-14B -> 投影 -> 目标模型

这是第二条路线，适合这些情况：

- 你想把 `Qwen3-14B` 的 emotion vectors 迁移到更小模型
- 你明确知道自己在做跨模型实验，而不是只想先跑通主线

这条路线的代价是：

- 需要额外拟合一个投影矩阵
- 迁移效果有不确定性，需要靠实验验证

当前仓库里已经给了一份现成模板：

- [configs/anthropic_qwen3_0_6b_projected_emotion24_task.yaml](configs/anthropic_qwen3_0_6b_projected_emotion24_task.yaml)

## 当前仓库里的对应关系

当前 `anthropic` 产物目录里已经有这几样东西：

- [../anthropic/artifacts/05_emotion_vectors/intermediate/emotion_vectors_orth.npy](../anthropic/artifacts/05_emotion_vectors/intermediate/emotion_vectors_orth.npy)
- [../anthropic/artifacts/05_emotion_vectors/intermediate/vector_metadata.json](../anthropic/artifacts/05_emotion_vectors/intermediate/vector_metadata.json)

这份 artifact 对应：

- 源模型：`Qwen/Qwen3-14B`
- 层：`26`
- 情绪数：`24`

这意味着：

- 用 `Qwen3-14B` 做 decoder-only soft prompt 训练时，可以直接初始化
- 用 `Qwen3-0.6B` 或其他 hidden size 不同的模型时，需要先做投影

## 和 emotion vector 的关系

如果你想把 emotion vector 当作 soft prompt 初始化，有两种模式：

1. 直接初始化：emotion vector 维度等于底座 causal LM hidden size
2. 投影初始化：提供一个投影矩阵，把源向量映射到目标 hidden size

例如：

- Qwen3-14B emotion vector 配 Qwen3-14B decoder-only 底座时，可以直接初始化
- 如果你的 emotion vector 来自别的模型空间，就可以加载 `emotion_vector_projection_path`

直接初始化时，配置 `prompt.init_strategy: emotion_vectors`，并提供：

- `prompt.emotion_vector_route`: `same_model` 或 `projected`
- `prompt.emotion_vectors_path`
- `prompt.emotion_vector_metadata_path`
- 可选 `prompt.emotion_vector_projection_path`
- `prompt.emotion_names`
- 可选 `prompt.emotion_vector_combination`: `repeat` / `interleave` / `mean_then_repeat`
- 可选 `prompt.emotion_vector_l2_normalize`

示例：

```yaml
prompt:
	num_virtual_tokens: 20
	init_strategy: emotion_vectors
	emotion_vector_route: same_model
	emotion_vectors_path: ../anthropic/artifacts/05_emotion_vectors/intermediate/emotion_vectors_orth.npy
	emotion_vector_metadata_path: ../anthropic/artifacts/05_emotion_vectors/intermediate/vector_metadata.json
	emotion_names: [joyful, hopeful, calm]
	emotion_vector_combination: repeat
	emotion_vector_l2_normalize: false
```

如果维度不匹配，再加：

```yaml
prompt:
	emotion_vector_route: projected
	emotion_vector_projection_path: artifacts/projections/qwen14b_to_qwen06b.npy
```

代码层面的约束现在也是显式的：

- `same_model`: 不允许设置 `emotion_vector_projection_path`，并且要求 emotion vector hidden size 与模型 hidden size 完全一致
- `projected`: 必须设置 `emotion_vector_projection_path`

投影矩阵 shape 必须是：

- `[source_hidden_size, target_hidden_size]`

## 现成配置

- [configs/anthropic_qwen3_14b_emotion_demo.yaml](configs/anthropic_qwen3_14b_emotion_demo.yaml): 直接使用当前 `anthropic` 的 Qwen3-14B emotion vectors
- [configs/anthropic_qwen3_0_6b_projected_demo.yaml](configs/anthropic_qwen3_0_6b_projected_demo.yaml): 通过投影矩阵接到 Qwen3-0.6B 的模板配置
- [configs/anthropic_qwen3_14b_emotion24_task.yaml](configs/anthropic_qwen3_14b_emotion24_task.yaml): 24 个情绪标签全部对齐到训练数据格式
- [configs/anthropic_qwen3_0_6b_projected_emotion24_task.yaml](configs/anthropic_qwen3_0_6b_projected_emotion24_task.yaml): 24 情绪标签 + 投影矩阵版

如果你只是想先验证流程，优先从这两个配置开始：

1. [configs/anthropic_qwen3_14b_emotion24_task.yaml](configs/anthropic_qwen3_14b_emotion24_task.yaml)
2. [configs/anthropic_qwen3_0_6b_projected_emotion24_task.yaml](configs/anthropic_qwen3_0_6b_projected_emotion24_task.yaml)

## 拟合投影矩阵

新增了命令：

```bash
decoder-soft-prompt-fit-projection \
	--texts-file data/emotion24_train.jsonl \
	--texts-file data/emotion24_eval.jsonl \
	--source-model Qwen/Qwen3-14B \
	--source-representation hidden \
	--source-layer-idx 26 \
	--source-token-pool-start 50 \
	--source-trust-remote-code \
	--target-model Qwen/Qwen3-0.6B \
	--target-representation embeddings \
	--target-trust-remote-code \
	--target-token-pool-start 0 \
	--output-path artifacts/projections/qwen14b_to_qwen06b.npy
```

默认会读取 `target` 字段作为文本，因为这更贴近你真正想要投影的情绪表达文本。如果你想改成别的字段，可以用 `--text-field`；如果你想把 `input` 和 `target` 拼起来做拟合，可以加 `--join-fields`。

### 推荐命令：Qwen3-14B -> Qwen3-0.6B

在 [data/emotion24_train.jsonl](data/emotion24_train.jsonl) 和 [data/emotion24_eval.jsonl](data/emotion24_eval.jsonl) 上拟合投影：

```bash
decoder-soft-prompt-fit-projection \
	--texts-file data/emotion24_train.jsonl \
	--texts-file data/emotion24_eval.jsonl \
	--source-model Qwen/Qwen3-14B \
	--source-representation hidden \
	--source-layer-idx 26 \
	--source-token-pool-start 50 \
	--source-trust-remote-code \
	--target-model Qwen/Qwen3-0.6B \
	--target-representation embeddings \
	--target-trust-remote-code \
	--output-path artifacts/projections/qwen14b_to_qwen06b.npy
```

这个命令会同时产出：

- `artifacts/projections/qwen14b_to_qwen06b.npy`
- `artifacts/projections/qwen14b_to_qwen06b.json`

其中 `.json` 会记录源模型、目标模型、层号、样本数和 projection shape。

## 直接训练

### 方案 1：直接使用 Qwen3-14B emotion vectors

```bash
decoder-soft-prompt-repro train --config configs/anthropic_qwen3_14b_emotion24_task.yaml
```

### 方案 2：先拟合投影，再训练 Qwen3-0.6B

先拟合投影：

```bash
decoder-soft-prompt-fit-projection \
	--texts-file data/emotion24_train.jsonl \
	--texts-file data/emotion24_eval.jsonl \
	--source-model Qwen/Qwen3-14B \
	--source-representation hidden \
	--source-layer-idx 26 \
	--source-token-pool-start 50 \
	--source-trust-remote-code \
	--target-model Qwen/Qwen3-0.6B \
	--target-representation embeddings \
	--target-trust-remote-code \
	--output-path artifacts/projections/qwen14b_to_qwen06b.npy
```

再训练：

```bash
decoder-soft-prompt-repro train --config configs/anthropic_qwen3_0_6b_projected_emotion24_task.yaml
```

### 评估已有 prompt

```bash
decoder-soft-prompt-repro eval \
	--config configs/anthropic_qwen3_14b_emotion24_task.yaml \
	--prompt-path artifacts/anthropic-qwen3-14b-emotion24-task/best_prompt.pt
```

## Windows PowerShell 示例

如果你在当前这台 Windows 机器上运行，建议直接用下面这种 PowerShell 写法。

先进入目录：

```powershell
Set-Location c:\Users\xiexi\PaperCode\decoder_soft_prompt
```

### 1. 拟合投影矩阵

```powershell
decoder-soft-prompt-fit-projection `
	--texts-file data/emotion24_train.jsonl `
	--texts-file data/emotion24_eval.jsonl `
	--source-model Qwen/Qwen3-14B `
	--source-representation hidden `
	--source-layer-idx 26 `
	--source-token-pool-start 50 `
	--source-trust-remote-code `
	--target-model Qwen/Qwen3-0.6B `
	--target-representation embeddings `
	--target-trust-remote-code `
	--output-path artifacts/projections/qwen14b_to_qwen06b.npy
```

如果你更想显式走当前虚拟环境里的 Python，也可以这样：

```powershell
c:/Users/xiexi/PaperCode/.venv/Scripts/python.exe -m decoder_soft_prompt_repro.projection `
	--texts-file data/emotion24_train.jsonl `
	--texts-file data/emotion24_eval.jsonl `
	--source-model Qwen/Qwen3-14B `
	--source-representation hidden `
	--source-layer-idx 26 `
	--source-token-pool-start 50 `
	--source-trust-remote-code `
	--target-model Qwen/Qwen3-0.6B `
	--target-representation embeddings `
	--target-trust-remote-code `
	--output-path artifacts/projections/qwen14b_to_qwen06b.npy
```

### 2. 直接训练 Qwen3-14B 版本

```powershell
decoder-soft-prompt-repro train --config configs/anthropic_qwen3_14b_emotion24_task.yaml
```

### 3. 训练投影后的 Qwen3-0.6B 版本

```powershell
decoder-soft-prompt-repro train --config configs/anthropic_qwen3_0_6b_projected_emotion24_task.yaml
```

### 4. 评估已有 prompt

```powershell
decoder-soft-prompt-repro eval `
	--config configs/anthropic_qwen3_14b_emotion24_task.yaml `
	--prompt-path artifacts/anthropic-qwen3-14b-emotion24-task/best_prompt.pt
```

PowerShell 里续行符是反引号 `` ` ``，不是反斜杠 `\`。如果你直接复制 Linux/bash 风格命令，最容易在这里出错。

## 24 情绪对齐数据

新增了一套更贴近当前任务的数据：

- [data/emotion24_train.jsonl](data/emotion24_train.jsonl)
- [data/emotion24_eval.jsonl](data/emotion24_eval.jsonl)

它们覆盖了当前 `anthropic` 产物里的全部 24 个情绪标签，并且使用了当前 decoder-only 训练格式：

```json
{"emotion": "hopeful", "input": "...", "target": "..."}
```

这组数据更适合当前项目的目的：

- `input` 负责给模型一个明确的风格/情绪指令
- `target` 负责提供目标语气的短文本
- `emotion` 字段保留下来，方便你后续做分组分析、采样或自定义评估

如果你后面想把它扩成更接近真实任务的数据，建议优先往这三个方向加：

1. 同一情绪下增加多个话题
2. 同一话题下增加不同长度目标文本
3. 增加 `split`, `topic`, `style`, `source` 之类的辅助字段

## 限制

当前实现是一个最小可用脚手架，不是完整训练平台。你需要注意：

- 训练器默认是简单的 decoder-only teacher forcing，不包含 RL、SFT pipeline 或复杂 prompt routing
- 投影矩阵目前是线性的 ridge-style 闭式解，不是非线性映射
- 是否真的能把 `Qwen3-14B` 的情绪方向有效迁移到更小模型，最终还是要靠实验验证
- 如果直接跑 `Qwen3-14B`，显存和加载时间都会明显高于测试环境里的小模型单测

## 命令

```bash
decoder-soft-prompt-repro train --config configs/base.yaml
decoder-soft-prompt-repro eval --config configs/base.yaml --prompt-path path/to/best_prompt.pt
```

## 文件结构

- [configs/base.yaml](configs/base.yaml): 基础实验配置
- [src/decoder_soft_prompt_repro/config.py](src/decoder_soft_prompt_repro/config.py): 配置结构与 override
- [src/decoder_soft_prompt_repro/data.py](src/decoder_soft_prompt_repro/data.py): JSONL 数据读取与 causal collator
- [src/decoder_soft_prompt_repro/prompt_tuning.py](src/decoder_soft_prompt_repro/prompt_tuning.py): decoder-only soft prompt 实现
- [src/decoder_soft_prompt_repro/training.py](src/decoder_soft_prompt_repro/training.py): 训练与评估循环
- [src/decoder_soft_prompt_repro/cli.py](src/decoder_soft_prompt_repro/cli.py): CLI 入口