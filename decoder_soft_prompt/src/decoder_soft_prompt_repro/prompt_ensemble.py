import torch
from pathlib import Path
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

@torch.no_grad()
def prompt_ensemble_predict(
    model_name_or_path: str,
    prompt_paths: List[str],
    input_text: str,
    device: str = "cuda"
) -> str:
    """
    对同一个输入，加载多个 soft prompt，集成输出（投票/平均概率）。
    目前实现简单投票（多数表决），适用于分类/生成任务。
    """
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    results = []
    for prompt_path in prompt_paths:
        # 假设 soft prompt 是 torch tensor，形状 [num_virtual_tokens, hidden_size]
        soft_prompt = torch.load(prompt_path, map_location=device)
        # 拼接 soft prompt 到输入（需根据你的模型实现调整）
        # 这里只做伪代码，实际需结合 prompt_tuning.py 的 forward 逻辑
        # 例如：model.forward_with_soft_prompt(input_text, soft_prompt)
        # 这里用普通生成做占位
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        # 伪代码：outputs = model.generate_with_soft_prompt(**inputs, soft_prompt=soft_prompt)
        outputs = model.generate(**inputs, max_new_tokens=32)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(result)
    # 简单投票/多数表决
    from collections import Counter
    most_common = Counter(results).most_common(1)[0][0]
    return most_common

# 用法示例：
# result = prompt_ensemble_predict(
#     model_name_or_path="Qwen/Qwen3-14B",
#     prompt_paths=["artifacts/prompt1.pt", "artifacts/prompt2.pt", "artifacts/prompt3.pt"],
#     input_text="Your input here"
# )
# print(result)
