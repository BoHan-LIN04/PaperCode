from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _clean(text: str) -> str:
    return " ".join(str(text).strip().split())


@dataclass(frozen=True)
class ProcessedExample:
    example_id: str
    source_text: str
    target_text: str
    references: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    dataset_config_name: str
    label_texts: list[str] | None
    is_classification: bool

    def process(self, example: dict[str, Any]) -> ProcessedExample:
        handler = TASK_HANDLERS[self.task_name]
        return handler(example)


def _label_to_bool_text(label: int) -> str:
    return "true" if int(label) == 1 else "false"


def _boolq(example: dict[str, Any]) -> ProcessedExample:
    source = f"passage: {_clean(example['passage'])} question: {_clean(example['question'])}"
    target = _label_to_bool_text(example["label"])
    return ProcessedExample(str(example["idx"]), source, target, [target], {})


def _cb(example: dict[str, Any]) -> ProcessedExample:
    label_map = {0: "entailment", 1: "contradiction", 2: "neutral"}
    target = label_map[int(example["label"])]
    source = f"premise: {_clean(example['premise'])} hypothesis: {_clean(example['hypothesis'])}"
    return ProcessedExample(str(example["idx"]), source, target, [target], {})


def _copa(example: dict[str, Any]) -> ProcessedExample:
    target = "choice1" if int(example["label"]) == 0 else "choice2"
    source = (
        f"premise: {_clean(example['premise'])} question: {_clean(example['question'])} "
        f"choice1: {_clean(example['choice1'])} choice2: {_clean(example['choice2'])}"
    )
    return ProcessedExample(str(example["idx"]), source, target, [target], {})


def _multirc(example: dict[str, Any]) -> ProcessedExample:
    source = (
        f"paragraph: {_clean(example['paragraph'])} question: {_clean(example['question'])} "
        f"answer: {_clean(example['answer'])}"
    )
    target = _label_to_bool_text(example["label"])
    group = f"{example['idx']['paragraph']}-{example['idx']['question']}"
    meta = {"group": group}
    return ProcessedExample(str(example["idx"]["answer"]), source, target, [target], meta)


def _record(example: dict[str, Any]) -> ProcessedExample:
    entities = " ; ".join(_clean(item) for item in example["entities"])
    source = f"passage: {_clean(example['passage'])} query: {_clean(example['query'])} entities: {entities}"
    answers = [_clean(answer) for answer in example["answers"]]
    target = answers[0]
    return ProcessedExample(str(example["idx"]), source, target, answers, {})


def _rte(example: dict[str, Any]) -> ProcessedExample:
    label_map = {0: "entailment", 1: "not_entailment"}
    target = label_map[int(example["label"])]
    source = f"premise: {_clean(example['premise'])} hypothesis: {_clean(example['hypothesis'])}"
    return ProcessedExample(str(example["idx"]), source, target, [target], {})


def _wic(example: dict[str, Any]) -> ProcessedExample:
    source = (
        f"word: {_clean(example['word'])} sentence1: {_clean(example['sentence1'])} "
        f"sentence2: {_clean(example['sentence2'])}"
    )
    target = _label_to_bool_text(example["label"])
    return ProcessedExample(str(example["idx"]), source, target, [target], {})


def _wsc(example: dict[str, Any]) -> ProcessedExample:
    source = (
        f"text: {_clean(example['text'])} span1: {_clean(example['span1_text'])} "
        f"span2: {_clean(example['span2_text'])}"
    )
    target = _label_to_bool_text(example["label"])
    return ProcessedExample(str(example["idx"]), source, target, [target], {})


# ============================================================================
# Domain Shift Experiments (Section 5)
# ============================================================================

# MRQA 2019 Shared Task - Extractive QA Domain Transfer
# Train on SQuAD, evaluate on different domains (TextbookQA, BioASQ, RACE, RE, DuoRC, DROP)

def _squad(example: dict[str, Any]) -> ProcessedExample:
    """
    SQuAD: Simple Question Answering Dataset.
    Input: question + context (passage)
    Output: answer (free-form text from passage)
    Used as in-domain training for MRQA experiments.
    """
    context = _clean(example["context"])
    question = _clean(example["question"])
    source = f"question: {question} context: {context}"
    
    # Get first answer from list
    answers = [_clean(answer["text"]) for answer in example["answers"]]
    target = answers[0] if answers else ""
    
    return ProcessedExample(
        example["id"],
        source,
        target,
        answers,  # Multiple valid answers
        {"question_id": example["id"]},
    )


def _mrqa_domain(example: dict[str, Any]) -> ProcessedExample:
    """
    Generic MRQA domain handler for: TextbookQA, BioASQ, RACE, RE, DuoRC, DROP.
    MRQA shared task format (unified across datasets).
    """
    context = _clean(example["context"])
    question = _clean(example["question"])
    source = f"question: {question} context: {context}"
    
    answers = [_clean(answer["text"]) for answer in example["answers"]]
    target = answers[0] if answers else ""
    
    return ProcessedExample(
        example["id"],
        source,
        target,
        answers,
        {"question_id": example["id"]},
    )


# QQP ↔ MRPC: Paraphrase Detection Domain Transfer

def _qqp(example: dict[str, Any]) -> ProcessedExample:
    """
    QQP: Quora Question Pairs.
    Task: Determine if two questions from Quora are semantically equivalent (duplicates).
    Train on QQP, evaluate zero-shot on MRPC (news article paraphrases).
    """
    text1 = _clean(example["question1"])
    text2 = _clean(example["question2"])
    label = int(example["label"])  # 0: not duplicate, 1: duplicate
    
    source = f"text1: {text1} text2: {text2}"
    target = "true" if label == 1 else "false"
    
    return ProcessedExample(
        str(example["idx"]),
        source,
        target,
        [target],
        {"question1": example["question1"], "question2": example["question2"]},
    )


def _mrpc(example: dict[str, Any]) -> ProcessedExample:
    """
    MRPC: Microsoft Research Paraphrase Corpus.
    Task: Determine if two sentences from news articles are paraphrases.
    Train on MRPC, evaluate zero-shot on QQP (Quora question duplicates).
    """
    text1 = _clean(example["sentence1"])
    text2 = _clean(example["sentence2"])
    label = int(example["label"])  # 0: not paraphrase, 1: paraphrase
    
    source = f"text1: {text1} text2: {text2}"
    target = "true" if label == 1 else "false"
    
    return ProcessedExample(
        str(example["idx"]),
        source,
        target,
        [target],
        {"sentence1": example["sentence1"], "sentence2": example["sentence2"]},
    )


TASK_HANDLERS = {
    "boolq": _boolq,
    "cb": _cb,
    "copa": _copa,
    "multirc": _multirc,
    "record": _record,
    "rte": _rte,
    "wic": _wic,
    "wsc": _wsc,
    # Domain Shift Tasks (Section 5)
    "squad": _squad,
    "textbookqa": _mrqa_domain,
    "bioasq": _mrqa_domain,
    "race": _mrqa_domain,
    "re": _mrqa_domain,
    "duorc": _mrqa_domain,
    "drop": _mrqa_domain,
    "qqp": _qqp,
    "mrpc": _mrpc,
}


TASK_SPECS = {
    "boolq": TaskSpec("boolq", "boolq", ["false", "true"], True),
    "cb": TaskSpec("cb", "cb", ["entailment", "contradiction", "neutral"], True),
    "copa": TaskSpec("copa", "copa", ["choice1", "choice2"], True),
    "multirc": TaskSpec("multirc", "multirc", ["false", "true"], True),
    "record": TaskSpec("record", "record", None, False),
    "rte": TaskSpec("rte", "rte", ["entailment", "not_entailment"], True),
    "wic": TaskSpec("wic", "wic", ["false", "true"], True),
    "wsc": TaskSpec("wsc", "wsc.fixed", ["false", "true"], True),
    # Domain Shift Tasks (Section 5)
    # MRQA: Extractive QA (free-form answer prediction)
    "squad": TaskSpec("squad", "squad", None, False),
    "textbookqa": TaskSpec("textbookqa", None, None, False),
    "bioasq": TaskSpec("bioasq", None, None, False),
    "race": TaskSpec("race", "high", None, False),
    "re": TaskSpec("re", None, None, False),
    "duorc": TaskSpec("duorc", "ParCC", None, False),
    "drop": TaskSpec("drop", None, None, False),
    # Paraphrase Detection (binary classification, cross-domain evaluation)
    "qqp": TaskSpec("qqp", "qqp", ["false", "true"], True),
    "mrpc": TaskSpec("mrpc", "mrpc", ["false", "true"], True),
}


def get_task_spec(task_name: str) -> TaskSpec:
    normalized = task_name.lower()
    if normalized not in TASK_SPECS:
        raise KeyError(f"Unsupported task: {task_name}")
    return TASK_SPECS[normalized]


def canonicalize_class_prediction(prediction: str, labels: list[str]) -> str:
    normalized = _clean(prediction).lower()
    if normalized in labels:
        return normalized
    for label in labels:
        if normalized.startswith(label):
            return label
    for label in labels:
        if label in normalized:
            return label
    return normalized