# 下载并保存GoEmotions原始数据集（27类+neutral）为jsonl
from datasets import load_dataset

# 下载GoEmotions数据集
# 包含28类（27情感+neutral）
dataset = load_dataset("go_emotions")

# 打印标签名
print("标签名:", dataset["train"].features["labels"].feature.names)

# 保存为jsonl文件
dataset["train"].to_json("data/goemotions_train.jsonl")
dataset["validation"].to_json("data/goemotions_val.jsonl")
dataset["test"].to_json("data/goemotions_test.jsonl")

print("已保存为 goemotions_train.jsonl, goemotions_val.jsonl, goemotions_test.jsonl")
