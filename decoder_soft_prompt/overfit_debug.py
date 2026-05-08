# 用极小数据做 soft prompt overfit 验证脚本（冻结主模型，仅训练 soft prompt）
# 用法：cd ~/decoder_soft_prompt && python overfit_debug.py

import json
import sys
import re
import torch
from torch.optim import AdamW

sys.path.insert(0, "src")
from decoder_soft_prompt_repro.prompt_tuning import SoftPromptCausalLM

# ===== 配置 =====
DATA_PATH = "data/emotion28_train_tagged.jsonl"
N = 10
MODEL_NAME = "Qwen/Qwen3-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_VIRTUAL_TOKENS = 28
EMOTION_VECTORS_PATH = "data/emotion_vectors_orth.npy"
EMOTION_VECTOR_METADATA_PATH = "data/vector_metadata.json"
EMOTION_NAMES = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
    'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
TRAIN_STEPS = 300
LR = 0.01

# ===== 读取数据 =====
with open(DATA_PATH, encoding="utf-8") as f:
    lines = [json.loads(line) for line in f][:N]
inputs = [item["input"] for item in lines]
targets = [item["target"] for item in lines]

# ===== 加载 SoftPromptCausalLM（主模型冻结，只训练 soft prompt） =====
print(f"[INFO] Loading SoftPromptCausalLM ({MODEL_NAME})...")
prompt_model, tokenizer = SoftPromptCausalLM.from_pretrained(
    model_name_or_path=MODEL_NAME,
    num_virtual_tokens=NUM_VIRTUAL_TOKENS,
    init_strategy="emotion_vectors",
    emotion_vector_route="same_model",
    emotion_vectors_path=EMOTION_VECTORS_PATH,
    emotion_vector_metadata_path=EMOTION_VECTOR_METADATA_PATH,
    emotion_names=EMOTION_NAMES,
    emotion_vector_combination="repeat",
    emotion_vector_l2_normalize=False,
    trust_remote_code=True,
)
prompt_model.to(DEVICE)

# 只有 soft prompt 参数参与训练，主模型冻结
trainable = list(prompt_model.trainable_parameters())
print(f"[INFO] Trainable params: {sum(p.numel() for p in trainable)}")
optimizer = AdamW(trainable, lr=LR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===== 训练循环（只训练 soft prompt） =====
print("[INFO] Start overfit training...")
prompt_model.train()
for step in range(TRAIN_STEPS):
    total_loss = 0.0
    for inp, tgt in zip(inputs, targets):
        full = inp + " " + tgt
        enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=160).to(DEVICE)
        input_len = len(tokenizer(inp, return_tensors="pt").input_ids[0])
        labels = enc.input_ids.clone()
        labels[:, :input_len] = -100
        outputs = prompt_model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if (step + 1) % 10 == 0:
        print(f"Step {step+1}/{TRAIN_STEPS}, loss={total_loss/len(inputs):.4f}")

# ===== 推理测试 =====
print("\n[INFO] Inference...")
prompt_model.eval()
preds = []
with torch.no_grad():
    for inp in inputs:
        enc = tokenizer(inp, return_tensors="pt", truncation=True, max_length=144).to(DEVICE)
        out = prompt_model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=1,
        )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        pred = ""
        for word in re.split(r'[\s,.;:!?\"\']+', decoded):
            if word.lower() in EMOTION_NAMES:
                pred = word.lower()
                break
        preds.append(pred)

print("\n[Overfit结果]")
for i, (tgt, pred) in enumerate(zip(targets, preds)):
    match = "OK" if pred == tgt else "FAIL"
    print(f"{i+1}. GT: {tgt:15s} | Pred: {pred:15s} [{match}]")

acc = sum(1 for t, p in zip(targets, preds) if t == p) / len(targets)
print(f"\n[INFO] Overfit accuracy: {acc*100:.1f}%")
