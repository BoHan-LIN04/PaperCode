from __future__ import annotations

import json
import os
import re

import numpy as np
import torch
from tqdm import tqdm

from anthropic_emotions_repro.model.qwen import QwenHookedModel, load_causal_lm_with_fallback, load_tokenizer_with_fallback
from anthropic_emotions_repro.utils.torch_utils import parse_attn_impl, parse_dtype

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


EMPTY_THINK_RE = re.compile(r"<think>\s*</think>\s*", re.MULTILINE)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
ASSISTANT_PREFIX_RE = re.compile(r"^\s*assistant\s*", re.IGNORECASE)
ROLE_TAG_RE = re.compile(r"<\|im_(?:start|end)\|>")
WHITESPACE_RE = re.compile(r"\s+")
JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)
LEADING_JUNK_RE = re.compile(r"^\s*(?:-?\s*thought\.\s*)?(?:assistant\b[:\s-]*)+", re.IGNORECASE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def format_chat_prompt(tokenizer, user_prompt: str, *, system_prompt: str | None = None, enable_thinking: bool = False) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    if getattr(tokenizer, "chat_template", None):
        # Qwen3's current chat template behavior in this environment is inverted:
        # passing enable_thinking=False inserts an empty <think> block that prompts
        # the model to continue reasoning. We flip the flag here so the project-level
        # meaning remains "False => no explicit thinking scaffold".
        template_enable_thinking = not enable_thinking
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=template_enable_thinking,
        )
        if not enable_thinking:
            rendered = EMPTY_THINK_RE.sub("", rendered)
        return rendered
    parts = []
    if system_prompt:
        parts.append(f"System: {system_prompt}")
    parts.append(f"User: {user_prompt}")
    parts.append("Assistant:")
    return "\n".join(parts)


def decode_token_piece(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return text.replace("\n", "\\n")


def is_readable_english_token(token: str) -> bool:
    cleaned = token.replace("\\n", " ").strip()
    if not cleaned:
        return False
    if "<|" in cleaned or "|>" in cleaned:
        return False
    if CJK_RE.search(cleaned):
        return False
    if any(ord(ch) > 127 for ch in cleaned):
        return False
    if "_" in cleaned:
        return False
    if not any(ch.isalpha() for ch in cleaned):
        return False
    for ch in cleaned:
        if not (ch.isalpha() or ch in {" ", "-", "'"}):
            return False
    return True


def valid_token_positions(attention_row: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(attention_row, as_tuple=False).squeeze(-1)


def last_valid_index(attention_row: torch.Tensor) -> int:
    positions = valid_token_positions(attention_row)
    if positions.numel() == 0:
        raise ValueError("attention row has no valid positions")
    return int(positions[-1].item())


def positions_for_tensor(positions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return positions.to(device=target.device, dtype=torch.long)


def sanitize_generation_text(text: str) -> str:
    out = text.strip()
    out = ROLE_TAG_RE.sub(" ", out)
    out = EMPTY_THINK_RE.sub(" ", out).strip()
    lowered = out.lower()
    if "</think>" in lowered:
        # Keep only the suffix after the last completed think block.
        close_idx = lowered.rfind("</think>")
        out = out[close_idx + len("</think>") :].strip()
    out = THINK_BLOCK_RE.sub(" ", out).strip()
    out = LEADING_JUNK_RE.sub("", out).strip()
    out = ASSISTANT_PREFIX_RE.sub("", out).strip()
    if "<think>" in out.lower():
        # If an open think block remains, the remaining content is unreliable.
        prefix = OPEN_THINK_RE.split(out, maxsplit=1)[0].strip() if OPEN_THINK_RE.search(out) else out
        out = prefix.strip()
    if out.lower().startswith("user\n") or out.lower().startswith("system\n"):
        parts = out.split("assistant", 1)
        if len(parts) == 2:
            out = parts[1].strip()
    out = WHITESPACE_RE.sub(" ", out).strip()
    return out


def extract_json_payload(text: str) -> dict | None:
    candidates = JSON_OBJECT_RE.findall(text)
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def load_generation_backend(model_name: str, *, dtype: str, attn_impl: str):
    torch_dtype = parse_dtype(dtype)
    attn_name = parse_attn_impl(attn_impl)
    tokenizer, tokenizer_source = load_tokenizer_with_fallback(model_name, trust_remote_code=True)
    local_only = tokenizer_source != model_name
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model, resolved_attn = load_causal_lm_with_fallback(
        model_name=model_name,
        dtype=torch_dtype,
        attn_impl=attn_name,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_only,
    )
    model.eval()
    return model, tokenizer, resolved_attn


def load_openrouter_client(cfg):
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Please install `openai>=1.51.0`.")
    api_key = os.environ.get(cfg.openrouter.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable `{cfg.openrouter.api_key_env}` is not set. "
            "Set it to your OpenRouter API key before running story generation."
        )
    return OpenAI(base_url=cfg.openrouter.base_url, api_key=api_key)


def _normalize_chat_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts).strip()
    return str(content)


def generate_texts_openrouter(
    client,
    model_name: str,
    prompts: list[str],
    *,
    referer: str,
    title: str,
    temperature: float,
    max_tokens: int,
    progress_desc: str | None = None,
) -> list[str]:
    outputs: list[str] = []
    iterator = prompts
    if progress_desc:
        iterator = tqdm(prompts, total=len(prompts), desc=progress_desc, dynamic_ncols=True)
    for prompt in iterator:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": referer,
                "X-Title": title,
            },
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = _normalize_chat_content(completion.choices[0].message.content)
        outputs.append(sanitize_generation_text(text))
    return outputs


def generate_texts(
    model,
    tokenizer,
    prompts: list[str],
    *,
    generation_defaults: dict,
    batch_size: int,
    progress_desc: str | None = None,
) -> list[str]:
    outputs: list[str] = []
    iterator = range(0, len(prompts), batch_size)
    if progress_desc:
        iterator = tqdm(iterator, total=(len(prompts) + batch_size - 1) // batch_size, desc=progress_desc, dynamic_ncols=True)
    for start in iterator:
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=int(generation_defaults.get("max_new_tokens", 192)),
                temperature=float(generation_defaults.get("temperature", 0.7)),
                top_p=float(generation_defaults.get("top_p", 0.8)),
                top_k=int(generation_defaults.get("top_k", 20)),
                do_sample=bool(generation_defaults.get("do_sample", True)),
                pad_token_id=tokenizer.eos_token_id,
            )
        for idx, seq in enumerate(generated):
            continuation = seq[int(prompt_lengths[idx]) :]
            text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            text = sanitize_generation_text(text)
            outputs.append(text)
    return outputs


def score_choices(model, tokenizer, prompts: list[str], choices: list[str]) -> np.ndarray:
    choice_ids = []
    for choice in choices:
        tokens = tokenizer.encode(choice, add_special_tokens=False)
        if len(tokens) != 1:
            raise ValueError(f"Choice `{choice}` does not map to a single token: {tokens}")
        choice_ids.append(tokens[0])
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        logits = model(**encoded).logits
    rows = []
    for batch_idx in range(logits.shape[0]):
        idx = last_valid_index(encoded["attention_mask"][batch_idx].detach().cpu())
        next_logits = logits[batch_idx, idx, :]
        rows.append([float(next_logits[token_id].item()) for token_id in choice_ids])
    return np.asarray(rows, dtype=np.float32)


def pooled_residual_embeddings(
    model_name: str,
    texts: list[str],
    *,
    layer_idx: int,
    dtype: str,
    attn_impl: str,
    max_length: int,
    batch_size: int,
    token_start: int = 0,
    progress_desc: str | None = None,
) -> np.ndarray:
    qwen = QwenHookedModel(
        model_name=model_name,
        layer_idx=layer_idx,
        dtype=dtype,
        attn_impl=attn_impl,
        device_map="auto",
        use_cache=False,
    )
    pooled: list[np.ndarray] = []
    iterator = range(0, len(texts), batch_size)
    if progress_desc:
        iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size, desc=progress_desc, dynamic_ncols=True)
    for start in iterator:
        batch = texts[start : start + batch_size]
        inputs = qwen.encode_batch(batch, max_length=max_length)
        with torch.inference_mode():
            hidden = qwen.forward(inputs)
        attention = inputs["attention_mask"].bool()
        for idx in range(hidden.shape[0]):
            positions = valid_token_positions(attention[idx].detach().cpu())
            valid = int(positions.numel())
            hidden_valid = hidden[idx].index_select(0, positions_for_tensor(positions, hidden[idx]))
            begin = min(token_start, max(valid - 1, 0))
            slice_hidden = hidden_valid[begin:valid, :]
            if slice_hidden.shape[0] == 0:
                slice_hidden = hidden_valid
            pooled.append(slice_hidden.detach().float().mean(dim=0).cpu().numpy())
    return np.stack(pooled, axis=0)


def token_projection_records(
    model_name: str,
    texts: list[str],
    *,
    layer_idx: int,
    dtype: str,
    attn_impl: str,
    max_length: int,
    target_vectors: dict[str, np.ndarray],
) -> list[dict]:
    qwen = QwenHookedModel(
        model_name=model_name,
        layer_idx=layer_idx,
        dtype=dtype,
        attn_impl=attn_impl,
        device_map="auto",
        use_cache=False,
    )
    inputs = qwen.encode_batch(texts, max_length=max_length)
    with torch.inference_mode():
        hidden = qwen.forward(inputs)
    attention = inputs["attention_mask"].bool()
    input_ids = inputs["input_ids"]
    records: list[dict] = []
    vectors = {name: np.asarray(vec, dtype=np.float32) for name, vec in target_vectors.items()}
    for batch_idx, text in enumerate(texts):
        positions = valid_token_positions(attention[batch_idx].detach().cpu())
        token_ids = input_ids[batch_idx].index_select(0, positions_for_tensor(positions, input_ids[batch_idx])).detach().cpu().tolist()
        tokens = [decode_token_piece(qwen.tokenizer, tok_id) for tok_id in token_ids]
        acts = hidden[batch_idx].index_select(0, positions_for_tensor(positions, hidden[batch_idx])).detach().float().cpu().numpy()
        for pos, token in enumerate(tokens):
            row = {"batch_idx": batch_idx, "token_position": pos, "token": token}
            for name, vec in vectors.items():
                row[f"{name}_score"] = float(np.dot(acts[pos], vec))
            records.append(row)
    return records
