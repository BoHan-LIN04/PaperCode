from pathlib import Path

import pandas as pd

from emotion_grpo.data import convert_jsonl_to_parquet, normalize_jsonl_record


def test_normalize_jsonl_record_preserves_messages_and_metadata():
    record = normalize_jsonl_record(
        {
            "messages": [{"role": "user", "content": "classify the emotion"}],
            "metadata": {"id": "sample-1", "label": "joy"},
        },
        record_index=1,
    )
    assert record["messages"][0]["role"] == "user"
    assert record["metadata"]["label"] == "joy"


def test_jsonl_is_converted_to_verl_parquet(tmp_path: Path):
    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text(
        '{"messages":[{"role":"user","content":"emotion?"}],"metadata":{"id":"demo-1","label":"joy"}}\n',
        encoding="utf-8",
    )
    parquet_path = tmp_path / "sample.parquet"

    output_path = convert_jsonl_to_parquet(jsonl_path=jsonl_path, parquet_path=parquet_path, split_name="train")
    frame = pd.read_parquet(output_path)

    assert list(frame.columns) == ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    assert frame.iloc[0]["extra_info"]["metadata"]["label"] == "joy"
    assert frame.iloc[0]["prompt"][0]["content"] == "emotion?"

