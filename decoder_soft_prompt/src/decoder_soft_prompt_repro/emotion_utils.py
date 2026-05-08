import numpy as np
import torch
import json
from pathlib import Path


def load_emotion_vector_table(vectors_path, names_path_or_list=None, metadata_path=None, config=None):
    """
    加载 emotion_vectors_orth.npy 和 emotion_names，返回 {emotion: tensor}
    支持：
      - names_path_or_list: 直接传list或json文件
      - metadata_path: 自动从metadata json读取 emotion_names
      - config: 直接从config对象读取 emotion_names/metadata_path
    """
    vectors = np.load(vectors_path)
    names = None
    # 优先 config
    if config is not None:
        if hasattr(config, 'prompt') and hasattr(config.prompt, 'emotion_names'):
            names = config.prompt.emotion_names
        if (names is None or not names) and hasattr(config.prompt, 'emotion_vector_metadata_path'):
            metadata_path = config.prompt.emotion_vector_metadata_path
    # 其次 metadata_path
    if (names is None or not names) and metadata_path is not None:
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        names = meta["emotion_names"] if "emotion_names" in meta else meta
    # 其次 names_path_or_list
    if (names is None or not names) and names_path_or_list is not None:
        if isinstance(names_path_or_list, (str, Path)):
            with open(names_path_or_list, encoding="utf-8") as f:
                names = json.load(f)
            if isinstance(names, dict) and "emotion_names" in names:
                names = names["emotion_names"]
        else:
            names = names_path_or_list
    assert names is not None and len(vectors) == len(names), f"向量数与标签数不一致: {len(vectors)} vs {len(names)}"
    return {name.strip().lower(): torch.tensor(vec, dtype=torch.float32) for name, vec in zip(names, vectors)}


def build_token2emotion(emotion_names, synonym_dict=None, extra_map=None):
    """
    批量生成 token2emotion，支持同义词自动补全和自定义扩展
    emotion_names: ['joy', 'sadness', ...]
    synonym_dict: {'joy': ['happy', 'cheerful', ...], ...}
    extra_map: {'delighted': 'joy', ...}  # 额外补充
    """
    token2emotion = {}
    # 先主情感词自身
    for emo in emotion_names:
        token2emotion[emo] = emo
    # 同义词
    if synonym_dict:
        for emo, syns in synonym_dict.items():
            for s in syns:
                token2emotion[s] = emo
    # 额外补充
    if extra_map:
        token2emotion.update(extra_map)
    return token2emotion

def get_trigger_tokens(token2emotion):
    return set(token2emotion.keys())


def get_emotion_vector_for_token(token, emotion_vector_table, token2emotion):
    """
    根据token查找emotion vector（先查类别，再查向量表）
    """
    emo_label = token2emotion.get(token.lower(), None)
    if emo_label is None:
        return None
    return emotion_vector_table.get(emo_label, None)


# 用法示例：
# emotion_vector_table = load_emotion_vector_table('data/emotion_vectors_orth.npy', metadata_path='data/vector_metadata.json')
# synonym_dict = {'joy': ['happy', 'cheerful', 'delighted', 'excited'], ...}
# token2emotion = build_token2emotion(list(emotion_vector_table.keys()), synonym_dict)
# trigger_tokens = get_trigger_tokens(token2emotion)
# token = 'happy'
# emo_vec = get_emotion_vector_for_token(token, emotion_vector_table, token2emotion)
# if emo_vec is not None:
#     ... # 注入到模型
