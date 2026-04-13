import torch

from anthropic_emotions_repro.runtime import (
    extract_json_payload,
    is_readable_english_token,
    last_valid_index,
    positions_for_tensor,
    sanitize_generation_text,
    valid_token_positions,
)


def test_sanitize_generation_text_removes_think_and_role_prefix():
    text = "<think>\nreasoning\n</think>\nassistant\nFinal answer here."
    cleaned = sanitize_generation_text(text)
    assert cleaned == "Final answer here."


def test_extract_json_payload_finds_embedded_object():
    payload = extract_json_payload('junk {"score": 0.8, "reason": "ok"} tail')
    assert payload is not None
    assert payload["score"] == 0.8


def test_sanitize_generation_text_keeps_suffix_after_think_block():
    text = "-thought. assistant <think>internal plan</think> I grip the bus seat and count each stop until the station appears."
    cleaned = sanitize_generation_text(text)
    assert cleaned == "I grip the bus seat and count each stop until the station appears."


def test_valid_token_positions_handles_left_padding():
    attention = torch.tensor([0, 0, 1, 1, 1], dtype=torch.bool)
    pos = valid_token_positions(attention)
    assert pos.tolist() == [2, 3, 4]
    assert last_valid_index(attention) == 4
    target = torch.randn(5, 3)
    remapped = positions_for_tensor(pos, target)
    assert remapped.device == target.device
    assert remapped.dtype == torch.long


def test_is_readable_english_token_filters_multilingual_and_special_tokens():
    assert is_readable_english_token(" urgent") is True
    assert is_readable_english_token(" grief") is True
    assert is_readable_english_token("公众号") is False
    assert is_readable_english_token("<|endoftext|>") is False
    assert is_readable_english_token("____") is False
