import json
import sys
from collections import Counter

def check_targets(jsonl_path, emotion_names=None, max_print=20):
        # 可视化分布
        plot_target_distribution(counter)

    def plot_target_distribution(counter):
        """
        绘制情感标签分布的美观柱状图
        依赖 matplotlib
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        labels, counts = zip(*counter.most_common())
        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, counts, color=plt.cm.tab20.colors[:len(labels)])
        plt.title('Emotion Label Distribution', fontsize=16)
        plt.xlabel('Emotion', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.gca().yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
        # 在柱子上标注数值
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{count}',
                     ha='center', va='bottom', fontsize=10, rotation=90)
        plt.tight_layout()
        plt.show()
    targets = []
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                targets.append(obj.get('target', None))
            except Exception as e:
                print(f"[ERROR] 解析行失败: {e}\n{line}")
    print(f"总样本数: {len(targets)}")
    counter = Counter(targets)
    print(f"唯一target种类数: {len(counter)}")
    print("出现频次最高的target:")
    for tgt, cnt in counter.most_common(max_print):
        print(f"  {tgt!r}: {cnt}")
    if emotion_names:
        emotion_set = set([e.strip().lower() for e in emotion_names])
        invalid = [t for t in targets if t is not None and t.strip().lower() not in emotion_set]
        print(f"不在emotion_names中的target数: {len(invalid)}")
        if invalid:
            print("示例:", invalid[:max_print])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_dataset_targets.py <jsonl_path> [emotion1 emotion2 ...]")
        sys.exit(1)
    jsonl_path = sys.argv[1]
    emotion_names = sys.argv[2:] if len(sys.argv) > 2 else None
    check_targets(jsonl_path, emotion_names)