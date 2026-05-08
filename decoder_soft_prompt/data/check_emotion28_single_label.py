import json

# 检查 emotion28_train.jsonl 是否每条都是单标签
single_label_count = 0
all_count = 0
labels_set = set()

with open("decoder_soft_prompt/data/emotion28_train.jsonl", "r", encoding="utf-8") as fin:
    for line in fin:
        all_count += 1
        data = json.loads(line)
        if "target" in data:
            labels_set.add(data["target"])

print(f"emotion28_train.jsonl 总样本数: {all_count}")
print(f"标签种类: {sorted(labels_set)} (共{len(labels_set)}类)")

# 检查 goemotions_train.jsonl 单标签样本数
orig_single_label_count = 0
orig_all_count = 0
with open("decoder_soft_prompt/data/goemotions_train.jsonl", "r", encoding="utf-8") as fin:
    for line in fin:
        orig_all_count += 1
        data = json.loads(line)
        if "labels" in data and isinstance(data["labels"], list) and len(data["labels"]) == 1:
            orig_single_label_count += 1

print(f"goemotions_train.jsonl 总样本数: {orig_all_count}")
print(f"goemotions_train.jsonl 单标签样本数: {orig_single_label_count}")
