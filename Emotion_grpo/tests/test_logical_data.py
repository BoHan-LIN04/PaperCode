from pathlib import Path

import pandas as pd

from emotion_grpo.data import prepare_logical_qa_jsonl


def test_prepare_logical_qa_jsonl_builds_chat_records(tmp_path: Path):
    source = tmp_path / "train.jsonl"
    source.write_text(
        '{"id": 1, "question": "2+2=?", "answer": "4", "domain": "arithmetic"}\n',
        encoding="utf-8",
    )
    target = tmp_path / "prepared.jsonl"
    prepare_logical_qa_jsonl([source], target, split_name="train", overwrite=True)

    frame = pd.read_json(target, lines=True)
    assert frame.iloc[0]["messages"][1]["content"].startswith("2+2=?")
    assert frame.iloc[0]["metadata"]["ground_truth"] == "4"
    assert frame.iloc[0]["metadata"]["logical_dataset"] == tmp_path.name
