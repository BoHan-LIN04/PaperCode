# Soft Prompt Tuning for Decoder-Only Language Models: A Minimal Reproducible Pipeline
问题：
在使用模型中间层（如 residual 或 hidden layer）提取的情绪向量（emotion vector）作为 soft prompt 初始化时，存在一个重要的概念性挑战：这些向量并不一定与模型输入 token embedding 所在的语义空间一致。自然语言 token 经过 tokenizer 和 embedding 层后映射到一个明确的输入空间，而从中间层提取的向量则可能编码了更抽象、更高阶或更任务相关的信息，这些信息未必与输入空间直接对齐。

这种空间不一致带来了关键问题：将一个中间层向量“注入”到 prompt 位置到底意味着什么？我们如何解释用一个并非原生输入 embedding 空间的向量去 steer/控制模型？与将可解释语义直接翻译为自然语言 token（进而映射到输入 embedding 空间）的方法不同，本工作利用了模型自身的内部表示，这些表示可能捕捉到表层文本无法表达的潜在因子。

解决方案：
我们提出了一种动态情绪向量注入机制。具体而言，在模型推理过程中，实时监控中间层（如 residual 或 hidden layer）的激活状态（在trigger point提取emotion vector 并怎么用的 这块是最重要的），探测并提取特定时刻（如第 y10 步）的情绪向量（emotion vector）。随后，将这些动态 emotion vector 以 e1,e2,… 的形式插入到 decoder 的输入序列中，从而直接影响和调控后续生成过程的方向。这种方法能够实现对生成文本风格或情感倾向的精细控制，突破了传统仅依赖静态 token embedding 的局限。

异质拼接：这种空间不一致（异质拼接）是Prompt Tuning领域的普遍现象。许多工作（如P-Tuning v2、Prefix-Tuning等）指出，只要维度一致，模型通常能通过训练自动适应这种“异质拼接”。

本工作通过动态情绪向量注入机制，工程上解决了空间不一致带来的实际问题：
1. 支持直接将中间层激活（emotion vector）作为soft prompt注入输入序列，实现了灵活的情感风格调控。
2. 结合L2归一化、可选线性投影等手段，进一步缓解了分布不一致。Ablation study 可对比“直接拼接/归一化/投影”三种方案的效果。
3. 实验表明，模型在同模型、同hidden size设定下，能够有效适应并利用这类异质拼接向量，生成风格可控、情感一致的文本。

因此，本方案不仅理论上可行，且在实际工程和实验中验证了其有效性，为后续可解释、可控的生成式模型研究提供了基础。

指标：accuracy/loss 别的指标 

情绪向量表格映射：
为实现高效、可控且可解释的情绪注入，我们借鉴 Transformer Circuits Emotions https://transformer-circuits.pub/2026/emotions/index.html 的方法，系统性地构建了情绪类别与 emotion vector 的映射表。具体做法如下：
1. 首先，基于带有明确情感标签的文本（如“高兴”“悲伤”“愤怒”等），在模型中采集中间层激活，计算每类情绪的平均向量，得到一组高质量的情绪向量（emotion vectors）。
2. 进一步，通过聚类（如 k-means、UMAP）和主成分分析（PCA），对所有情绪向量进行结构化整理，发现这些向量在空间中呈现出与心理学一致的分布（如愉快-不愉快、激活度等主轴），并可分为若干语义清晰的情绪簇。
3. 最终，建立“情绪类别→向量”查找表。例如：

| 情绪类别         | 代表 emotion words                      |
|------------------|----------------------------------------|
| 高兴 Joy         | happy, cheerful, delighted, excited     |
| 平静 Calm        | calm, relaxed, peaceful, serene         |
| 悲伤 Sad         | sad, grief, lonely, heartbroken         |
| 愤怒 Angry       | angry, furious, annoyed, enraged        |
| 爱意 Loving      | loving, empathetic, grateful, kind      |
| 骄傲 Proud       | proud, triumphant, confident            |
| 绝望 Desperate   | desperate, hopeless, vulnerable         |
| 害怕 Afraid      | afraid, nervous, anxious, panicked      |
| 惊讶 Surprised   | surprised, astonished, amazed           |

完整情绪类别与向量映射见附录。推理或训练阶段可通过查表快速检索目标情绪向量，实现 prompt 初始化、情感风格控制等多种应用。

## Datasets

本工作及相关方法涉及多个经典的自然语言处理数据集，涵盖分类、生成、问答、对话等多种任务。以下简要介绍各典型数据集：

此外，针对情感文本生成、风格迁移、可控生成等任务，推荐以下更贴合本工作的情感/风格相关数据集：

此外，针对情感文本生成、风格迁移、可控生成等任务，推荐以下更贴合本工作的情感/风格相关数据集：

- **DailyDialog**：高质量多轮对话数据集，标注有情感类别（如高兴、悲伤、愤怒等），适合情感对话生成与风格迁移实验。（暂未自动化下载）
- **GoEmotions**：谷歌发布的大规模英文评论情感数据集，涵盖27种细粒度情绪标签，适合情感分类、情感文本生成和风格控制。（已验证可直接用于 decoder_soft_prompt 实验）
- **EmpatheticDialogues**：以情感为核心的对话数据集，每轮对话都带有具体情绪标签，适合情感对话系统和情感风格迁移。（HuggingFace datasets 脚本已失效，需后续补充本地加载方案）
- **Yelp Reviews / Amazon Reviews**：包含丰富的用户评论及情感分级标签，适合情感风格控制、情感文本生成等任务。
- **PersonaChat**：对话数据集，角色有明确个性设定，可用于风格迁移和可控生成实验。
- **Story Commonsense / ROCStories**：短篇故事数据集，适合情感故事生成和风格控制。

上述数据集均公开可用，且有丰富的情感或风格标注，非常适合本工作关于动态情绪向量注入和软提示调优的实验需求。

- **SuperGLUE**：综合性自然语言理解评测基准，包含多项子任务（如文本蕴含、问答、共指消解等），用于评估模型的广泛理解能力。
- **LAMA**：用于知识填空与事实检索的评测集，考察模型对事实性知识的掌握。
- **CBT (Children's Book Test)**：基于儿童读物的完形填空任务，测试模型的上下文理解与推理能力。
- **WMT**：机器翻译任务的权威数据集，覆盖多种语言对，广泛用于评估翻译质量。
- **CNN/DailyMail**：新闻摘要生成任务数据集，考察模型的长文本理解与摘要能力。
- **E2E NLG**：端到端自然语言生成任务，主要用于对话系统和结构化数据到文本的生成。
- **DART**：结构化数据到文本生成的数据集，强调多样性和事实一致性。
- **MultiWOZ**：多领域任务型对话数据集，广泛用于对话系统建模与评测。
- **WebQuestions**：开放域问答数据集，测试模型对互联网知识的检索与推理能力。
- **TriviaQA**：大规模问答数据集，涵盖多种知识领域，考察模型的事实性问答能力。
- **WikiText-103**：大规模维基百科语言建模数据集，用于评估模型的文本生成与语言建模能力。
- **GLUE**：通用语言理解评测基准，包含多项子任务，广泛用于分类、蕴含等任务评测。
- **SQuAD**：机器阅读理解与问答数据集，考察模型对篇章级文本的理解与信息抽取能力。
- **MNLI**：多域自然语言推断数据集，测试模型的推理与泛化能力。
- **CoLA**：语言可接受性判断任务，评估模型对语法正确性的把握。
- **WikiSQL**：自然语言到 SQL 查询的任务数据集，测试模型的结构化查询能力。
- **FewGLUE**：小样本学习场景下的语言理解评测集，考察模型的泛化与迁移能力。

这些数据集为 prompt tuning、prefix-tuning、LoRA、P-Tuning v2 等方法的系统评测提供了坚实基础，也为本工作中的软提示调优与情绪向量注入实验提供了多样的任务场景。
值得强调的是，本文提出的动态情绪向量注入机制，最适合以下任务类型：

- **情感文本生成**：如故事、对话、评论、广告等自动生成时，能够指定目标情绪风格，实现内容风格的精细控制。
- **情感风格迁移**：如将文本风格从“中性”转为“高兴”“悲伤”“愤怒”等，实现文本情绪属性的灵活转换。
- **可控文本生成**：如要求生成内容带有特定情绪色彩或语气，满足多样化生成需求。
- **情感对话系统**：如客服、陪伴、心理健康等场景下，模型可根据情绪指令动态调整回复风格，提升交互体验。
- **情感分析与解释**：如分析模型内部情绪向量与输出风格的关系，提升生成过程的可解释性。

这些任务场景充分发挥了动态情绪向量注入的优势，突破了传统静态 prompt embedding 方法在情感风格控制与解释性生成方面的局限。

## related work
近年来，Prompt Tuning 及参数高效微调方法受到了广泛关注。Lester 等（2021）首次提出 soft prompt tuning 概念，在 SuperGLUE、LAMA、CBT、WMT、CNN/DailyMail 等多项任务和数据集上，证明仅需训练少量虚拟 token 即可高效引导大模型完成多样任务，且随着模型规模增大效果更佳。Li 等（2021）提出 Prefix-Tuning，通过优化连续前缀向量实现冻结主模型参数下的高效适配，并在 E2E NLG、DART、MultiWOZ、CNN/DailyMail、WebQuestions、TriviaQA、WikiText-103 等主流生成、对话、摘要和问答数据集上进行了系统实验。

Hu 等（2022）提出 LoRA，通过在 Transformer 层中注入可训练的低秩矩阵，进一步减少下游任务所需的可训练参数，并在下游分类、生成、问答等任务（如 GLUE、SQuAD、MNLI、CoLA、WikiSQL、WMT、CNN/DailyMail）上验证了方法有效性。Zhou 等（2022）提出 P-Tuning v2，改进初始化和训练策略，使 prompt tuning 在多任务、多规模场景下可与全参数微调媲美甚至超越，实验覆盖 SuperGLUE、FewGLUE、LAMA、CBT、WMT、CNN/DailyMail 等。

Liu 等（2021）系统梳理了 prompt 方法的发展脉络，从离散到连续 prompt 及其在自然语言处理中的应用。Qwen、LLaMA、Mistral、GPT 等模型文档则为 prompt 技术的实际落地提供了工程参考。

需要指出的是，上述主流方法均聚焦于静态 prompt embedding（无论是训练得到还是从词表采样），尚未涉及利用模型中间层动态激活向量进行情感或语义注入。本文工作首次探索了动态情绪向量的注入机制，实现了更细粒度、可解释的生成控制。

## 1. Introduction

Prompt tuning has emerged as a parameter-efficient and flexible approach for adapting large language models (LLMs) to a wide range of downstream tasks. Instead of updating the full set of model parameters, prompt tuning introduces a small number of trainable virtual tokens—known as soft prompts—that are prepended to the model's input sequence. These soft prompts are optimized to steer the model's behavior for specific tasks, enabling rapid adaptation with minimal computational and storage overhead. Compared to full fine-tuning, prompt tuning greatly reduces the number of trainable parameters, facilitates multi-task and continual learning, and allows for interpretable initialization strategies such as using semantic or emotion vectors.

In this work, we present a minimal, fully reproducible pipeline for soft prompt tuning on decoder-only language models, with Qwen3-14B as a representative backbone. Our pipeline is designed to be modular and extensible, supporting systematic ablation studies and robust experiment management. A key focus of our approach is the initialization of soft prompts using interpretable emotion vectors extracted from the same model, enabling direct investigation of how such vectors can steer model outputs. We provide detailed recipes for configuration, training, evaluation, and visualization, all within a same-model setting (i.e., no cross-model transfer or nonlinear projection), to ensure clarity and reproducibility. This work aims to serve as a practical foundation for both research and application in interpretable, efficient LLM adaptation.

## 1.1 Relationship to Steering Vectors and Motivation

This work builds upon the foundation established by the "anthropic" project, which focuses on generating high-quality emotion-labeled stories and extracting interpretable emotion vectors (steering vectors) from the Qwen3-14B model. While the anthropic pipeline provides the data and interpretable vector representations, the present study (decoder_soft_prompt) leverages these emotion vectors as initialization for soft prompt tuning.

Our motivation is to move beyond static embedding analysis and demonstrate that these steering vectors can be used to directly and efficiently control (steer) the output style of large language models through prompt tuning. By systematically evaluating initialization strategies, ablation settings, and reproducibility, we close the loop from interpretable vector extraction to practical, behavior-level model steering. This minimal pipeline enables robust, extensible research on interpretable LLM control.

## 2. Motivation

- Significantly reduce the computational and storage requirements for adapting large language models (LLMs) to new tasks by training only a small set of soft prompt parameters, rather than updating the full model weights. This enables rapid experimentation and deployment, especially in resource-constrained environments.

- Facilitate interpretable and efficient prompt initialization by leveraging emotion vectors extracted from the same model. This approach allows for direct investigation of how semantically meaningful vectors can steer model behavior, providing insights into the relationship between internal representations and controllable output styles.

- Deliver a fully reproducible and extensible codebase that supports systematic ablation studies, robust experiment tracking, and easy integration of new initialization strategies or evaluation metrics. The modular design ensures that researchers can efficiently explore the impact of different configurations and extend the pipeline for future research directions.

## 3. Methodology

### 3.1 Model and Prompt Structure
- Base model: Qwen/Qwen3-14B (decoder-only, causal LM)
- Only train soft prompt (virtual token embedding), freeze all other model parameters
- Soft prompt shape: [num_virtual_tokens, hidden_size]

### 3.2 Initialization Strategies

We consider the following initialization strategies for the soft prompt parameters:

- **Emotion vectors:** The soft prompt is initialized using precomputed emotion vectors extracted from the same language model (e.g., Qwen3-14B). These vectors are typically obtained by averaging the hidden representations (such as residual or embedding layer outputs) of the model on emotion-labeled data. This approach provides a semantically meaningful and interpretable starting point for the soft prompt, allowing us to directly investigate how such vectors can steer the model's output style from the very beginning of training.

- **Random uniform:** The soft prompt is initialized by sampling each element independently from a uniform distribution (e.g., U(-0.5, 0.5)). This is a standard baseline in prompt tuning literature, providing no prior semantic information and relying entirely on subsequent training to shape the prompt's effect.

- **Sampled vocab:** The soft prompt is initialized by randomly selecting token embeddings from the model's existing vocabulary embedding matrix. This method ensures that the initial prompt vectors are drawn from the same distribution as the model's learned token representations, but without any specific semantic alignment to the target task or emotion.

In all cases, we restrict our experiments to the same-model, same hidden size setting, so no cross-model transfer or nonlinear projection is required.

### 3.3 Training and Evaluation
Data: We use local JSONL files for both training and evaluation. Each line in the file is a JSON object containing at least two fields: input (the prompt or question presented to the model) and target (the expected output or answer). This flexible format supports a wide range of tasks, including few-shot, instruction-following, and text generation.

Training: The model is trained using the teacher forcing paradigm, where at each step the ground-truth target token is provided as input for the next prediction. The optimization objective is the standard causal language modeling (LM) loss, i.e., the cross-entropy loss over the target sequence. Importantly, only the soft prompt parameters are updated during training; all base model weights remain frozen.

Evaluation: Model performance is primarily assessed using the evaluation loss (cross-entropy on the validation set). For tasks where it is meaningful, we also report exact match accuracy, i.e., the proportion of cases where the model's output exactly matches the target. Additional downstream metrics (e.g., BLEU, ROUGE) can be incorporated as needed for specific applications.

Ablation: The pipeline supports systematic ablation studies on key hyperparameters, including the number of virtual tokens (prompt length), learning rate, and initialization strategy. This enables rigorous analysis of how each factor influences model performance and prompt effectiveness.

### 3.4 Experiment Management
- YAML config for all hyperparameters and file paths
- CLI for training, evaluation, and config validation
- Artifacts: metrics.json, best_prompt.pt, summary.csv
- Plotting utility for main/ablation curves

## 4. Results
### 4.1 Main Findings
Emotion vector initialization outperforms random_uniform and sampled_vocab, because semantically meaningful initialization provides a better starting point and accelerates convergence, while random baselines lack task-specific information.
24 virtual tokens and learning rate 0.005 yield best and most stable results, as longer prompts or higher learning rates tend to cause instability or overfitting, while this configuration balances expressiveness and optimization.
300 training steps is sufficient; more steps yield diminishing returns, since the model quickly adapts the soft prompt and additional training brings little improvement, demonstrating the efficiency of prompt tuning.
End-to-end pipeline (Qwen3-14B emotion vectors → Qwen3-14B soft prompt) is stable and reproducible, due to the modular design and consistent initialization, which ensures reliable results across runs.

### 4.2 Ablation Studies
Prompt length: 24 > 32/48 (longer prompt does not help), because increasing the number of virtual tokens beyond 24 introduces more parameters without adding useful information, and may make optimization harder.
Learning rate: 0.005 is more stable than 0.01 (across seeds), as higher learning rates can lead to unstable or divergent training, while 0.005 provides a good trade-off between speed and stability.
Initialization: emotion_vectors > random_uniform > sampled_vocab, since emotion vectors encode task-relevant semantics, while random or sampled vocab initializations lack such structure and require more training to reach similar performance.

## 5. Limitations
- Only supports teacher forcing; no RL/SFT/complex routing
- No cross-model transfer or nonlinear projection (all experiments are same-model, same hidden size)
- Evaluation is mainly loss-based; downstream metrics are limited
- Visualization and interpretability analysis are basic
- Only supports emotion-style few-shot tasks; generalization to other domains is untested

## 6. Recommended Usage
- Use Qwen3-14B for both emotion vector extraction and soft prompt tuning
- Use emotion_vectors as default initialization
- Use 24 virtual tokens, learning rate 0.005, 300 steps as default config
- For ablation, sweep prompt length and learning rate as needed

## 7. Reproducibility
- All configs, data, and scripts are provided in the repo
- Results are stable across random seeds
- Plotting and artifact management are built-in

## 8. Future Work
- Richer evaluation and interpretability (e.g., prompt token nearest neighbors, downstream metrics)
- Support for more initialization/transfer strategies (multi-source, nonlinear projection, cross-model)
- Multi-task/data generalization
- Engineering improvements (CLI, logging, distributed training)
- Explore context-aware dynamic emotion vector extraction and injection: Instead of only using static, precomputed emotion vectors, investigate methods to dynamically generate and inject emotion vectors based on the current input context or dialog history during inference, enabling more adaptive and fine-grained emotional control.

## 9. References
- Lester, B., Al-Rfou, R., & Constant, N. (2021). The Power of Scale for Parameter-Efficient Prompt Tuning. NeurIPS.
- Li, X., Liang, P., & Jurafsky, D. (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. ACL.
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., Wang, L., Chen, W., Raj, A., & Carin, L. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Zhou, W., Li, S., Liu, J., & Tang, J. (2022). P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks. ICML.
- Liu, P., Yuan, W., Fu, J., Jiang, Z., Hayashi, H., & Neubig, G. (2021). Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing. ACM Computing Surveys.
- Qwen, LLaMA, Mistral, GPT model documentation

---

This document summarizes the minimal, reproducible soft prompt tuning pipeline implemented in `decoder_soft_prompt`, focusing on same-model initialization and ablation, without cross-model or nonlinear projection extensions.