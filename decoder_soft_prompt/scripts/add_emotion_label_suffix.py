import json

# 批量为 input 字段加英文后缀 ' Emotion label:'
def add_emotion_label_suffix(in_path, out_path):
    label_list = '[admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral]'
    strong_suffix = f" Please classify the above text into one of the following emotions: {label_list} Emotion label:"
    with open(in_path, encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            obj = json.loads(line)
            obj['input'] = obj['input'].strip() + strong_suffix
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # 用法示例
    add_emotion_label_suffix('decoder_soft_prompt/data/emotion28_train.jsonl', 'decoder_soft_prompt/data/emotion28_train_tagged.jsonl')
    add_emotion_label_suffix('decoder_soft_prompt/data/emotion28_eval.jsonl', 'decoder_soft_prompt/data/emotion28_eval_tagged.jsonl')
    print('处理完成，已生成带英文提示的新数据集。')
