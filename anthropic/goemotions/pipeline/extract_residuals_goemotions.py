import torch
import numpy as np
import jsonlines
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# 配置
MODEL_NAME = "Qwen/Qwen3-14B"
DATA_PATH = "../data/goemotions_train.jsonl"  # 相对 pipeline 目录
OUTPUT_DIR = "../artifacts/activation_cache"
TOKEN_POOL_START = 0  # 可根据需要调整
LAYER_IDX = -1  # -1 表示最后一层

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 加载模型
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).eval().cuda()

# 读取数据
samples = []
with jsonlines.open(DATA_PATH) as reader:
    for i, obj in enumerate(reader):
        samples.append({"sample_id": i, "input": obj["input"], "emotion": obj["target"]})

# 提取 residuals
activations = []
sample_ids = []
token_positions = []

for sample in tqdm(samples, desc="extract_residuals"):    
    inputs = tokenizer(sample["input"], return_tensors="pt", truncation=True, max_length=128).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[LAYER_IDX].squeeze(0)  # [seq_len, hidden_size]
    for pos in range(TOKEN_POOL_START, hidden.shape[0]):
        activations.append(hidden[pos].to(torch.float32).cpu().numpy())
        sample_ids.append(sample["sample_id"])
        token_positions.append(pos)



# 保存 activation cache（默认 float32）
np.save(f"{OUTPUT_DIR}/activations.npy", np.stack(activations, axis=0))
np.save(f"{OUTPUT_DIR}/sample_ids.npy", np.array(sample_ids))
np.save(f"{OUTPUT_DIR}/token_positions.npy", np.array(token_positions))

# 构建 sample_index
sample_index = pd.DataFrame(samples)
sample_index["pool_start_token"] = TOKEN_POOL_START
sample_index.to_parquet(f"{OUTPUT_DIR}/sample_index.parquet")

print("Done! 已保存 activations、sample_ids、token_positions、sample_index")
