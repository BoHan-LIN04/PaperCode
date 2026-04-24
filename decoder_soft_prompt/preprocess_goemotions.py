# goemotions 数据集转标准 JSONL 脚本
# 生成 data/goemotions_train.jsonl, data/goemotions_test.jsonl, data/goemotions_validation.jsonl


from datasets import load_dataset
import json
import os

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
    'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
    'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

def main():
    ds = load_dataset("mrm8488/goemotions")
    os.makedirs("data", exist_ok=True)

    def to_jsonl(split):
        out_path = f"data/goemotions_{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for item in ds[split]:
                text = item["text"]
                # 取第一个为1的情感标签
                emotion = next((label for label in EMOTION_LABELS if item[label] == 1), "neutral")
                json.dump({"input": text, "target": emotion}, f, ensure_ascii=False)
                f.write("\n")
        print(f"Saved {out_path}")

    for split in ds.keys():
        to_jsonl(split)

if __name__ == "__main__":
    main()
