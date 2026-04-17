# Current_Version.md

## 项目定位

decoder_soft_prompt 是一个面向 decoder-only 大模型（如 Qwen、LLaMA、Mistral、GPT 等）的 soft prompt tuning 复现与实验脚手架。其目标是高效、灵活地支持情绪向量初始化、prompt ablation、跨模型迁移等研究场景。

## 当前主要特性

- 冻结 AutoModelForCausalLM，仅训练 soft prompt（虚拟 token embedding）
- 支持多种初始化方式：emotion_vectors、random_uniform、sampled_vocab
- 支持直接用 .npy/.pt 文件初始化 prompt
- 支持 emotion vector 直接初始化和线性投影（跨模型 hidden size）
- 支持本地 JSONL 数据集训练与评估
- PyTorch + Transformers + YAML config + 简单 CLI
- 结果可视化脚本（plotting.py），支持主图/ablation 曲线绘制

## 当前推荐实验配置

- 模型：Qwen/Qwen3-14B
- 路线：same_model
- 初始化：emotion_vectors
- num_virtual_tokens=24
- learning_rate=0.005
- max_steps=300
- batch_size=1
- eval_batch_size=1
- model.torch_dtype=bfloat16

详见 configs/anthropic_qwen3_14b_emotion24_task.yaml

## 已验证实验结论

- emotion_vectors 初始化显著优于 random_uniform 和 sampled_vocab
- prompt 长度 24、lr=0.005 最优，长 prompt 或大 lr 反而不稳定
- 300 steps 已是较优预算，500/800 steps 收益递减
- 端到端链路（Qwen3-14B emotion vectors -> Qwen3-14B soft prompt）已稳定可用

## 主要局限与改进建议

- 只支持 teacher forcing，不含 RL/SFT/复杂 routing
- 投影矩阵仅为线性 ridge 解，未探索非线性映射
- 评估指标以 loss 为主，缺少下游任务指标
- 可视化和可解释性分析有待丰富
- 仅适配情绪风格小样本任务，泛化能力待扩展

## 推荐后续方向

- 丰富评估与可解释性分析（如 prompt token 最近邻、下游指标）
- 支持更多初始化/迁移策略（如多源融合、非线性投影）
- 扩展训练流程（RL/SFT、ensemble、超参搜索）
	- ensemble：论文只在附录简单提到 prompt ensemble 的效果提升，主线实验并不要求实现。可选扩展。
	- 超参搜索：论文所有 ablation（如 prompt 长度、初始化、学习率）都属于超参 sweep，这部分建议实现（已支持）。
- 适配多任务/多领域数据
- 工程完善（CLI、日志、分布式训练等）

---

如需详细实验流程、配置说明或代码扩展建议，请查阅 README.md 或联系维护者。
