from __future__ import annotations

import re
import string
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import f1_score

from .tasks import ProcessedExample


def _normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punctuation = "".join(ch for ch in lowered if ch not in string.punctuation)
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punctuation)
    return " ".join(no_articles.split())


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    ref_tokens = _normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(task_name: str, predictions: list[str], examples: list[ProcessedExample]) -> dict[str, float]:
    task_name = task_name.lower()
    if task_name == "cb":
        gold = [example.references[0] for example in examples]
        acc = float(np.mean([pred == ref for pred, ref in zip(predictions, gold)]))
        f1 = float(f1_score(gold, predictions, average="macro", labels=["entailment", "contradiction", "neutral"]))
        return {"accuracy": acc, "f1": f1, "score": (acc + f1) / 2}
    if task_name == "multirc":
        gold = [example.references[0] for example in examples]
        pred_binary = [1 if pred == "true" else 0 for pred in predictions]
        gold_binary = [1 if ref == "true" else 0 for ref in gold]
        f1a = float(f1_score(gold_binary, pred_binary))
        groups: dict[str, list[bool]] = defaultdict(list)
        for pred, ref, example in zip(predictions, gold, examples):
            groups[example.metadata["group"]].append(pred == ref)
        em = float(np.mean([all(group) for group in groups.values()])) if groups else 0.0
        return {"exact_match": em, "f1_a": f1a, "score": (em + f1a) / 2}
    if task_name == "record":
        em_scores = []
        f1_scores = []
        for prediction, example in zip(predictions, examples):
            normalized_prediction = _normalize_answer(prediction)
            exact = max(1.0 if normalized_prediction == _normalize_answer(ref) else 0.0 for ref in example.references)
            f1 = max(_token_f1(prediction, ref) for ref in example.references)
            em_scores.append(exact)
            f1_scores.append(f1)
        exact_match = float(np.mean(em_scores)) if em_scores else 0.0
        f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        return {"exact_match": exact_match, "f1": f1, "score": (exact_match + f1) / 2}
    
    # Domain Shift Tasks
    # MRQA (SQuAD, TextbookQA, BioASQ, RACE, RE, DuoRC, DROP) - Extractive QA
    if task_name in ["squad", "textbookqa", "bioasq", "race", "re", "duorc", "drop"]:
        em_scores = []
        f1_scores = []
        for prediction, example in zip(predictions, examples):
            normalized_prediction = _normalize_answer(prediction)
            # Exact match: check against all references
            exact = max(1.0 if normalized_prediction == _normalize_answer(ref) else 0.0 for ref in example.references)
            # Token-level F1: check against all references
            f1 = max(_token_f1(prediction, ref) for ref in example.references)
            em_scores.append(exact)
            f1_scores.append(f1)
        exact_match = float(np.mean(em_scores)) if em_scores else 0.0
        f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        return {"exact_match": exact_match, "f1": f1, "score": (exact_match + f1) / 2}
    
    # Paraphrase Detection (QQP, MRPC) - Binary classification
    if task_name in ["qqp", "mrpc"]:
        gold = [example.references[0] for example in examples]
        pred_binary = [1 if pred == "true" else 0 for pred in predictions]
        gold_binary = [1 if ref == "true" else 0 for ref in gold]
        accuracy = float(np.mean([pred == ref for pred, ref in zip(predictions, gold)])) if gold else 0.0
        f1 = float(f1_score(gold_binary, pred_binary)) if gold else 0.0
        return {"accuracy": accuracy, "f1": f1, "score": (accuracy + f1) / 2}
    
    gold = [example.references[0] for example in examples]
    accuracy = float(np.mean([pred == ref for pred, ref in zip(predictions, gold)])) if gold else 0.0
    return {"accuracy": accuracy, "score": accuracy}