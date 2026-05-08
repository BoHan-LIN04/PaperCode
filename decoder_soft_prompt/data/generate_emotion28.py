import json
from pathlib import Path


def _to_multi_hot(label_ids, num_labels):
    vec = [0] * num_labels
    for idx in label_ids:
        if 0 <= idx < num_labels:
            vec[idx] = 1
    return vec


def process_file(input_path, single_output_path, multi_output_path, label_names):
    num_labels = len(label_names)
    single_count = 0
    multi_count = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
        open(single_output_path, "w", encoding="utf-8") as fout_single, \
        open(multi_output_path, "w", encoding="utf-8") as fout_multi:
        for line in fin:
            data = json.loads(line)
            labels = data.get("labels", [])
            text = data.get("text", "")

            if not isinstance(labels, list) or len(labels) == 0:
                continue

            label_ids = [int(i) for i in labels if isinstance(i, int) and 0 <= i < num_labels]
            if not label_ids:
                continue

            label_names_all = [label_names[i] for i in label_ids]
            primary_label = label_names_all[0]
            payload_common = {
                "input": text,
                # 为了兼容现有单标签训练代码，保留 target。
                "target": primary_label,
                # 完整多标签信息。
                "targets": label_names_all,
                "label_ids": label_ids,
                "target_multi_hot": _to_multi_hot(label_ids, num_labels),
                "is_single_label": len(label_ids) == 1,
            }

            fout_multi.write(json.dumps(payload_common, ensure_ascii=False) + "\n")
            multi_count += 1

            if len(label_ids) == 1:
                fout_single.write(json.dumps(payload_common, ensure_ascii=False) + "\n")
                single_count += 1

    return single_count, multi_count

# 28类标签
label_names = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire',
    'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
    'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

base = Path(__file__).resolve().parent

train_single, train_multi = process_file(
    str(base / "goemotions_train.jsonl"),
    str(base / "emotion28_train.jsonl"),
    str(base / "emotion28_train_multi.jsonl"),
    label_names,
)
eval_single, eval_multi = process_file(
    str(base / "goemotions_val.jsonl"),
    str(base / "emotion28_eval.jsonl"),
    str(base / "emotion28_eval_multi.jsonl"),
    label_names,
)
test_single, test_multi = process_file(
    str(base / "goemotions_test.jsonl"),
    str(base / "emotion28_test.jsonl"),
    str(base / "emotion28_test_multi.jsonl"),
    label_names,
)

print("已生成单标签与多标签两套数据：")
print(f"- emotion28_train.jsonl (single): {train_single}")
print(f"- emotion28_eval.jsonl (single): {eval_single}")
print(f"- emotion28_test.jsonl (single): {test_single}")
print(f"- emotion28_train_multi.jsonl (multi): {train_multi}")
print(f"- emotion28_eval_multi.jsonl (multi): {eval_multi}")
print(f"- emotion28_test_multi.jsonl (multi): {test_multi}")
