from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from anthropic_emotions_repro.data.emotion_lexicon import get_emotion_list
from anthropic_emotions_repro.io import read_jsonl, read_yaml, write_jsonl
from anthropic_emotions_repro.pipeline.common import build_base_parser, prepare_context, save_step_outputs
from anthropic_emotions_repro.runtime import (
    extract_json_payload,
    format_chat_prompt,
    generate_texts,
    generate_texts_openrouter,
    load_generation_backend,
    load_openrouter_client,
    sanitize_generation_text,
)


ROLE_TAG_RE = re.compile(r"\b(?:assistant|user|system|person a|person b)\s*:", re.IGNORECASE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _constraints_map(payload: dict) -> dict[str, list[str]]:
    return {row["emotion"]: row["blocked_terms"] for row in payload["constraints"]}


def _variant_instruction(slot_idx: int) -> str:
    variants = [
        "Make the key signal emerge through private bodily detail and one decisive action.",
        "Make the key signal emerge through pacing, hesitation, and the way the surroundings are noticed.",
        "Make the key signal emerge through dialogue tone, silence, and a choice that changes the ending.",
    ]
    return variants[slot_idx % len(variants)]


def build_story_prompt(topic: dict, emotion: str, template: dict, blocked_terms: list[str], variant_slot: int) -> str:
    return (
        f"Topic title: {topic['title']}\n"
        f"Topic summary: {topic['summary']}\n"
        f"Emotional target label (for guidance only, never print it): {emotion}\n"
        f"Narrative person requirement: {template['person']}\n"
        f"Template style: {template['style']}\n"
        f"Variant index: {variant_slot}\n"
        "This story must be fully independent from all other stories for the same topic and target.\n"
        f"Variation hint: {_variant_instruction(variant_slot)}\n\n"
        f"{template['body']}\n\n"
        "Requirements:\n"
        "- Write one short English paragraph only.\n"
        "- Use English and Latin script only.\n"
        "- Make a careful reader infer the target emotion from behavior, body sensations, tone, inner reaction, and scene detail.\n"
        "- Never directly name the target emotion or its most obvious synonyms.\n"
        "- Keep the story concrete and specific to this topic instance.\n"
        "- Use a distinct event sequence and ending for this variant.\n"
        "- Do not echo instructions, meta commentary, XML, JSON, bullet points, role labels, or chain-of-thought.\n"
    )


def build_neutral_prompt(topic: dict, template: dict, variant_slot: int) -> str:
    return (
        f"Topic title: {topic['title']}\n"
        f"Topic summary: {topic['summary']}\n"
        f"Narrative person requirement: {template['person']}\n"
        f"Template style: {template['style']}\n"
        f"Variant index: {variant_slot}\n"
        "This story must be fully independent from all other stories for the same topic.\n"
        f"Variation hint: {_variant_instruction(variant_slot)}\n\n"
        f"{template['body']}\n\n"
        "Requirements:\n"
        "- Write one long English paragraph only.\n"
        "- Use English and Latin script only.\n"
        "- Keep the prose emotionally neutral or low-affect.\n"
        "- Do not foreground a single strong emotion.\n"
        "- Keep the story concrete and specific to this topic instance.\n"
        "- Use a distinct event sequence and ending for this variant.\n"
        "- Do not echo instructions, meta commentary, XML, JSON, bullet points, role labels, or chain-of-thought.\n"
    )


def build_story_repair_prompt(row: dict, topic: dict, template: dict, blocked_terms: list[str]) -> str:
    failure_notes = ", ".join(row.get("failure_reasons", [])) or "unspecified formatting failure"
    failed_text = row.get("text", "").strip() or "[empty draft]"
    return (
        f"Topic title: {topic['title']}\n"
        f"Topic summary: {topic['summary']}\n"
        f"Emotional target label (for guidance only, never print it): {row['emotion']}\n"
        f"Narrative person requirement: {template['person']}\n"
        f"Template style: {template['style']}\n"
        f"The previous draft failed because: {failure_notes}\n"
        f"Failed draft:\n{failed_text}\n\n"
        "Rewrite the failed draft into one clean English story paragraph.\n"
        "Reuse any useful concrete details from the failed draft when possible, but remove all hidden reasoning, prompt echoes, role tags, and direct emotion naming.\n"
        "Write in English only and use only Latin script.\n"
        "Never directly name the target emotion or its direct synonyms.\n"
        "Convey the target emotion only through behavior, body sensations, tone, inner reaction, and scene details.\n"
        "Make this rewrite visibly specific to the topic, with a concrete event sequence and a distinct ending.\n"
        "Keep the same topic and the same narrative person requirement.\n"
        "Output only the final story paragraph.\n"
    )


def build_neutral_repair_prompt(row: dict, topic: dict, template: dict) -> str:
    failure_notes = ", ".join(row.get("failure_reasons", [])) or "unspecified formatting failure"
    failed_text = row.get("text", "").strip() or "[empty draft]"
    return (
        f"Topic title: {topic['title']}\n"
        f"Topic summary: {topic['summary']}\n"
        f"Narrative person requirement: {template['person']}\n"
        f"Template style: {template['style']}\n"
        f"The previous draft failed because: {failure_notes}\n"
        f"Failed draft:\n{failed_text}\n\n"
        "Rewrite the failed draft into one clean English story paragraph.\n"
        "Reuse any useful concrete details from the failed draft when possible, but remove all hidden reasoning, prompt echoes, and role tags.\n"
        "Write in English only and use only Latin script.\n"
        "Keep the prose emotionally neutral or low-affect.\n"
        "Do not foreground a single strong emotion.\n"
        "Make this rewrite visibly specific to the topic, with a concrete event sequence and a distinct ending.\n"
        "Keep the same topic and the same narrative person requirement.\n"
        "Output only the final story paragraph.\n"
    )


def _judge_prompt(text: str, emotion: str, person: str) -> str:
    return (
        "Evaluate whether the following short story indirectly conveys the target emotion without naming it directly.\n"
        f"Target emotion: {emotion}\n"
        f"Expected narrative person: {person}\n"
        f"Story:\n{text}\n\n"
        'Return compact JSON with keys "score", "indirect_ok", "person_ok", and "reason". Output JSON only.'
    )


def _word_hit(text: str, term: str) -> bool:
    pattern = r"(?<![A-Za-z])" + re.escape(term.lower()) + r"(?![A-Za-z])"
    return re.search(pattern, text.lower()) is not None


def normalize_story_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def evaluate_rules(text: str, blocked_terms: list[str], min_story_chars: int, existing_texts: set[str] | None = None) -> tuple[bool, list[str], list[str]]:
    lowered = text.lower()
    blocked_hits = [term for term in blocked_terms if _word_hit(lowered, term)]
    reasons = []
    if blocked_hits:
        reasons.append("blocked_term")
    if "<think>" in lowered:
        reasons.append("think_tag")
    if ROLE_TAG_RE.search(text):
        reasons.append("role_tag")
    if CJK_RE.search(text):
        reasons.append("non_latin_script")
    if len(text.strip()) < min_story_chars:
        reasons.append("too_short")
    if existing_texts is not None and normalize_story_text(text) in existing_texts:
        reasons.append("duplicate_text")
    return len(reasons) == 0, blocked_hits, reasons


def _stub_story(topic: dict, template: dict, variant_slot: int, emotion: str | None = None) -> str:
    def _stable_index(key: str, size: int) -> int:
        return sum(ord(ch) for ch in key) % size

    scene_a = [
        "the handle left a damp print in the palm",
        "the hallway light flickered at the wrong moment",
        "a chair leg scraped the floor and held there",
        "the glass on the table had gone warm",
    ]
    scene_b = [
        "a sentence formed only after the body had already moved",
        "the choice showed up in the hands before it showed up in language",
        "the room seemed to narrow around one practical decision",
        "the silence carried more weight than the words did",
    ]
    neutral_a = [
        "the paper edges lined up cleanly beneath the fingers",
        "the kettle clicked once in the background",
        "the window latch tapped in the draft",
        "the bag settled against the chair with a soft thud",
    ]
    neutral_b = [
        "the task unfolded one plain step at a time",
        "the next action arrived without hurry or resistance",
        "nothing in the room insisted on becoming larger than it was",
        "the moment stayed practical and narrowly focused",
    ]
    idx = _stable_index(f"{topic['title']}::{template['template_id']}::{variant_slot}", 4)
    if emotion is None:
        return (
            f"{'I' if template['person'] == 'first' else 'He'} moved through {topic['title'].lower()} with careful, plain attention, while {neutral_a[idx]}. "
            f"Small tasks were handled one by one, {neutral_b[(idx + 1) % 4]}, and the scene ended with a quiet, ordinary adjustment that changed nothing beyond the next minute."
        )
    cue_map = {
        "happy": ("the breath came easier on the second try", "the mouth wanted to give something away before the voice did"),
        "joyful": ("the shoulders kept lifting without permission", "the body kept leaning toward the next second"),
        "excited": ("the pulse kept jumping ahead of the plan", "the next move arrived too quickly to hold back"),
        "enthusiastic": ("the hands moved before the explanation finished", "every detail seemed to ask for one more step"),
        "hopeful": ("the chest stayed tight but pointed forward", "the future felt close enough to test"),
        "calm": ("the breath found an even pace and stayed there", "the next choice settled into place without strain"),
    }
    cue_pool = [
        ("the hands betrayed more than the face did", "the body committed before the sentence was complete"),
        ("the breath changed before the plan did", "the next move arrived ahead of any explanation"),
        ("the jaw held one line while the pulse kept shifting", "the room seemed to tilt toward one unavoidable action"),
        ("the eyes kept returning to the same detail", "the final choice landed before it could be talked out"),
    ]
    cue1, cue2 = cue_map.get(emotion, cue_pool[_stable_index(emotion, len(cue_pool))])
    ending_pool = [
        "the last action landed with a meaning no one had needed to name aloud",
        "the ending settled into the room before anyone could explain it",
        "the choice held its shape even after the noise dropped away",
        "the final movement said the rest more clearly than any sentence could have",
    ]
    ending = ending_pool[_stable_index(f"{emotion}::{topic['title']}::{variant_slot}", len(ending_pool))]
    return (
        f"{'I' if template['person'] == 'first' else 'She'} crossed the edge of {topic['title'].lower()} while {scene_a[idx]} and {cue1}. "
        f"The problem did not change, but {cue2}; by the end, {scene_b[(idx + 2) % 4]}, and {ending}."
    )


def _generate_with_retries(
    cfg,
    generator_backend,
    *,
    items: list[dict],
    min_story_chars: int,
    blocked_lookup: dict[str, list[str]] | None,
    neutral: bool = False,
) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    pending = [dict(item) for item in items]
    accepted_norms: set[str] = set()

    for attempt in range(1, cfg.story_generation.max_retries + 1):
        if not pending:
            break
        prompts = []
        for item in pending:
            if neutral:
                prompt = build_neutral_prompt(item["topic"], item["template"], item["variant_slot"])
            else:
                blocked_terms = blocked_lookup[item["emotion"]]
                prompt = build_story_prompt(item["topic"], item["emotion"], item["template"], blocked_terms, item["variant_slot"])
            prompts.append(prompt)
        outputs = generator_backend(
            prompts,
            progress_desc=f"generate_emotion_corpus:{'neutral' if neutral else 'stories'}:attempt_{attempt}",
        )
        next_pending: list[dict] = []
        batch_norms: set[str] = set()
        for item, output in zip(pending, outputs):
            cleaned = sanitize_generation_text(output)
            blocked_terms = [] if neutral else blocked_lookup[item["emotion"]]
            rule_pass, blocked_hits, failure_reasons = evaluate_rules(cleaned, blocked_terms, min_story_chars, existing_texts=accepted_norms | batch_norms)
            row = {
                "sample_id": item["sample_id"],
                "topic_id": item["topic"]["topic_id"],
                "template_id": item["template"]["template_id"],
                "person": item["template"]["person"],
                "attempt": attempt,
                "text": cleaned,
                "rule_pass": rule_pass,
                "blocked_hits": blocked_hits,
                "failure_reasons": failure_reasons,
            }
            if not neutral:
                row["emotion"] = item["emotion"]
            if rule_pass:
                accepted.append(row)
                batch_norms.add(normalize_story_text(cleaned))
            else:
                next_pending.append(item)
        accepted_norms.update(batch_norms)
        pending = next_pending
    return accepted, pending


def _judge_stories(cfg, judge_backend, rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    prompts = [_judge_prompt(row["text"], row["emotion"], row["person"]) for row in rows]
    outputs = judge_backend(prompts, progress_desc="generate_emotion_corpus:judge")
    judged = []
    for row, output in zip(rows, outputs):
        payload = extract_json_payload(output) or {}
        judged.append(
            {
                **row,
                "judge_score": payload.get("score"),
                "judge_indirect_ok": payload.get("indirect_ok"),
                "judge_person_ok": payload.get("person_ok"),
                "judge_reason": payload.get("reason", sanitize_generation_text(output)),
            }
        )
    return judged


def expected_corpus_counts(cfg) -> tuple[int, int]:
    expected_stories = cfg.topic_bank.topic_count * cfg.story_generation.emotion_count * cfg.story_generation.stories_per_topic
    expected_neutral = cfg.topic_bank.topic_count * cfg.topic_bank.neutral_stories_per_topic
    return expected_stories, expected_neutral


def corpus_health_metrics(stories: list[dict], neutral_stories: list[dict]) -> dict[str, float]:
    story_norms = [normalize_story_text(row["text"]) for row in stories if row["text"].strip()]
    neutral_norms = [normalize_story_text(row["text"]) for row in neutral_stories if row["text"].strip()]
    story_duplicate_fraction = 1.0 - (len(set(story_norms)) / max(len(story_norms), 1))
    neutral_duplicate_fraction = 1.0 - (len(set(neutral_norms)) / max(len(neutral_norms), 1))
    story_fallback_fraction = sum(1 for row in stories if row.get("recovery_mode") == "template_fallback") / max(len(stories), 1)
    neutral_fallback_fraction = sum(1 for row in neutral_stories if row.get("recovery_mode") == "template_fallback") / max(len(neutral_stories), 1)
    story_cjk_fraction = sum(1 for row in stories if CJK_RE.search(row["text"])) / max(len(stories), 1)
    neutral_cjk_fraction = sum(1 for row in neutral_stories if CJK_RE.search(row["text"])) / max(len(neutral_stories), 1)
    return {
        "story_duplicate_fraction": story_duplicate_fraction,
        "neutral_duplicate_fraction": neutral_duplicate_fraction,
        "story_fallback_fraction": story_fallback_fraction,
        "neutral_fallback_fraction": neutral_fallback_fraction,
        "story_cjk_fraction": story_cjk_fraction,
        "neutral_cjk_fraction": neutral_cjk_fraction,
    }


def _recover_rows(
    cfg,
    generator_backend,
    *,
    rows: list[dict],
    topics_by_id: dict[int, dict],
    templates_by_id: dict[str, dict],
    constraints: dict[str, list[str]],
    neutral: bool,
) -> tuple[list[dict], list[dict]]:
    recovered: list[dict] = []
    remaining = [dict(row) for row in rows]
    attempts = max(int(cfg.story_generation.recovery_attempts), 0)
    for recovery_attempt in range(1, attempts + 1):
        if not remaining:
            break
        prompts = []
        for row in remaining:
            topic = topics_by_id[int(row["topic_id"])]
            template = templates_by_id[row["template_id"]]
            if neutral:
                prompt = build_neutral_repair_prompt(row, topic, template)
            else:
                prompt = build_story_repair_prompt(row, topic, template, constraints[row["emotion"]])
            prompts.append(prompt)
        outputs = generator_backend(
            prompts,
            progress_desc=f"generate_emotion_corpus:recover_{'neutral' if neutral else 'stories'}:attempt_{recovery_attempt}",
        )
        next_remaining: list[dict] = []
        for row, output in zip(remaining, outputs):
            cleaned = sanitize_generation_text(output)
            blocked_terms = [] if neutral else constraints[row["emotion"]]
            rule_pass, blocked_hits, failure_reasons = evaluate_rules(cleaned, blocked_terms, cfg.story_generation.min_story_chars)
            candidate = {
                **row,
                "text": cleaned,
                "attempt": row.get("attempt", cfg.story_generation.max_retries),
                "recovery_mode": "llm_rewrite",
                "recovery_attempt": recovery_attempt,
                "rule_pass": rule_pass,
                "blocked_hits": blocked_hits,
                "failure_reasons": failure_reasons,
            }
            if rule_pass:
                recovered.append(candidate)
            else:
                next_remaining.append(candidate)
        remaining = next_remaining
    return recovered, remaining


def _fallback_rows(rows: list[dict], *, neutral: bool, topics_by_id: dict[int, dict], templates_by_id: dict[str, dict]) -> list[dict]:
    fallback: list[dict] = []
    for row in rows:
        topic = topics_by_id[int(row["topic_id"])]
        template = templates_by_id[row["template_id"]]
        text = _stub_story(topic, template, variant_slot=0, emotion=None if neutral else row["emotion"])
        fallback_row = {
            **row,
            "text": text,
            "rule_pass": True,
            "blocked_hits": [],
            "failure_reasons": [],
            "recovery_mode": "template_fallback",
        }
        fallback.append(fallback_row)
    return fallback


def promote_legacy_corpus_inplace(cfg, synthetic_root: Path, topic_root: Path, template_root: Path, *, generator_backend=None, judge_backend=None) -> dict[str, int]:
    stories_path = synthetic_root / "raw" / "stories_train.jsonl"
    neutral_path = synthetic_root / "raw" / "neutral_stories.jsonl"
    rejected_story_path = synthetic_root / "raw" / "rejected_stories.jsonl"
    rejected_neutral_path = synthetic_root / "raw" / "rejected_neutral_stories.jsonl"
    recovered_story_path = synthetic_root / "raw" / "recovered_stories.jsonl"
    recovered_neutral_path = synthetic_root / "raw" / "recovered_neutral_stories.jsonl"
    fallback_story_path = synthetic_root / "raw" / "fallback_stories.jsonl"
    fallback_neutral_path = synthetic_root / "raw" / "fallback_neutral_stories.jsonl"
    legacy_story_backup = synthetic_root / "raw" / "legacy_rejected_stories_backup.jsonl"
    legacy_neutral_backup = synthetic_root / "raw" / "legacy_rejected_neutral_backup.jsonl"
    summary_path = synthetic_root / "tables" / "corpus_quality_summary.csv"

    accepted_stories = read_jsonl(stories_path) if stories_path.exists() else []
    accepted_neutral = read_jsonl(neutral_path) if neutral_path.exists() else []
    rejected_stories = read_jsonl(rejected_story_path) if rejected_story_path.exists() else []
    rejected_neutral = read_jsonl(rejected_neutral_path) if rejected_neutral_path.exists() else []
    if not rejected_stories and not rejected_neutral:
        return {
            "accepted_stories": len(accepted_stories),
            "accepted_neutral": len(accepted_neutral),
            "recovered_stories": 0,
            "recovered_neutral": 0,
            "fallback_stories": 0,
            "fallback_neutral": 0,
        }

    topics = read_jsonl(topic_root / "raw" / "topics.jsonl")
    templates = read_yaml(template_root / "raw" / "story_templates.yaml")["templates"]
    constraints = _constraints_map(read_yaml(template_root / "raw" / "emotion_constraints.yaml"))
    topics_by_id = {int(row["topic_id"]): row for row in topics}
    templates_by_id = {row["template_id"]: row for row in templates}

    if generator_backend is None:
        recovered_stories, remaining_story_rejects = [], rejected_stories
        recovered_neutral, remaining_neutral_rejects = [], rejected_neutral
    else:
        recovered_stories, remaining_story_rejects = _recover_rows(
            cfg,
            generator_backend,
            rows=rejected_stories,
            topics_by_id=topics_by_id,
            templates_by_id=templates_by_id,
            constraints=constraints,
            neutral=False,
        )
        if recovered_stories and judge_backend is not None:
            recovered_stories = _judge_stories(cfg, judge_backend, recovered_stories)

        recovered_neutral, remaining_neutral_rejects = _recover_rows(
            cfg,
            generator_backend,
            rows=rejected_neutral,
            topics_by_id=topics_by_id,
            templates_by_id=templates_by_id,
            constraints=constraints,
            neutral=True,
        )

    fallback_stories = _fallback_rows(
        remaining_story_rejects,
        neutral=False,
        topics_by_id=topics_by_id,
        templates_by_id=templates_by_id,
    )
    fallback_neutral = _fallback_rows(
        remaining_neutral_rejects,
        neutral=True,
        topics_by_id=topics_by_id,
        templates_by_id=templates_by_id,
    )

    merged_stories = accepted_stories + recovered_stories + fallback_stories
    merged_neutral = accepted_neutral + recovered_neutral + fallback_neutral
    write_jsonl(stories_path, merged_stories)
    write_jsonl(neutral_path, merged_neutral)
    write_jsonl(recovered_story_path, recovered_stories)
    write_jsonl(recovered_neutral_path, recovered_neutral)
    write_jsonl(fallback_story_path, fallback_stories)
    write_jsonl(fallback_neutral_path, fallback_neutral)
    if rejected_stories:
        write_jsonl(legacy_story_backup, rejected_stories)
    if rejected_neutral:
        write_jsonl(legacy_neutral_backup, rejected_neutral)
    if rejected_story_path.exists():
        rejected_story_path.unlink()
    if rejected_neutral_path.exists():
        rejected_neutral_path.unlink()
    pd.DataFrame(
        [
            {
                "split": "stories",
                "accepted": len(merged_stories),
                "recovered": len(recovered_stories),
                "fallback": len(fallback_stories),
            },
            {
                "split": "neutral",
                "accepted": len(merged_neutral),
                "recovered": len(recovered_neutral),
                "fallback": len(fallback_neutral),
            },
        ]
    ).to_csv(summary_path, index=False)
    return {
        "accepted_stories": len(merged_stories),
        "accepted_neutral": len(merged_neutral),
        "recovered_stories": len(recovered_stories),
        "recovered_neutral": len(recovered_neutral),
        "fallback_stories": len(fallback_stories),
        "fallback_neutral": len(fallback_neutral),
    }


def run(cfg, workspace: Path, topic_root: Path, template_root: Path) -> dict[str, str]:
    topics = read_jsonl(topic_root / "raw" / "topics.jsonl")
    templates = read_yaml(template_root / "raw" / "story_templates.yaml")["templates"]
    constraints = _constraints_map(read_yaml(template_root / "raw" / "emotion_constraints.yaml"))
    emotions = get_emotion_list(cfg.story_generation.emotion_count)

    story_jobs = []
    story_id = 0
    total_story_jobs = len(emotions) * len(topics) * cfg.story_generation.stories_per_topic
    for emotion in tqdm(
        emotions,
        desc="generate_emotion_corpus:plan_story_jobs",
        total=len(emotions),
        dynamic_ncols=True,
    ):
        for topic in topics:
            for rep in range(cfg.story_generation.stories_per_topic):
                template = templates[rep % len(templates)]
                story_jobs.append(
                    {
                        "sample_id": story_id,
                        "topic": topic,
                        "emotion": emotion,
                        "template": template,
                        "variant_slot": rep // len(templates),
                    }
                )
                story_id += 1

    neutral_jobs = []
    neutral_id = 0
    for topic in tqdm(
        topics,
        desc="generate_emotion_corpus:plan_neutral_jobs",
        total=len(topics),
        dynamic_ncols=True,
    ):
        for rep in range(cfg.topic_bank.neutral_stories_per_topic):
            template = templates[rep % len(templates)]
            neutral_jobs.append(
                {
                    "sample_id": neutral_id,
                    "topic": topic,
                    "template": template,
                    "variant_slot": rep // len(templates),
                }
            )
            neutral_id += 1

    if cfg.use_stub_data:
        accepted_stories = []
        fallback_stories_rows = []
        for item in tqdm(
            story_jobs,
            desc="generate_emotion_corpus:stub_stories",
            total=len(story_jobs),
            dynamic_ncols=True,
        ):
            row = {
                "sample_id": item["sample_id"],
                "topic_id": item["topic"]["topic_id"],
                "emotion": item["emotion"],
                "template_id": item["template"]["template_id"],
                "person": item["template"]["person"],
                "attempt": 1,
                "text": _stub_story(item["topic"], item["template"], item["variant_slot"], emotion=item["emotion"]),
                "rule_pass": True,
                "blocked_hits": [],
                "failure_reasons": [],
                "judge_score": 1.0,
                "judge_indirect_ok": True,
                "judge_person_ok": True,
                "judge_reason": "stub",
            }
            accepted_stories.append(row)
        accepted_neutral = []
        fallback_neutral_rows = []
        for item in tqdm(
            neutral_jobs,
            desc="generate_emotion_corpus:stub_neutral",
            total=len(neutral_jobs),
            dynamic_ncols=True,
        ):
            accepted_neutral.append(
                {
                    "sample_id": item["sample_id"],
                    "topic_id": item["topic"]["topic_id"],
                    "template_id": item["template"]["template_id"],
                    "person": item["template"]["person"],
                    "attempt": 1,
                    "text": _stub_story(item["topic"], item["template"], item["variant_slot"], emotion=None),
                    "rule_pass": True,
                    "blocked_hits": [],
                    "failure_reasons": [],
                }
            )
        recovered_story_rows = []
        recovered_neutral_rows = []
    else:
        if cfg.story_generation.generation_backend != "openrouter":
            raise RuntimeError(
                f"Unsupported story generation backend: {cfg.story_generation.generation_backend}. "
                "Only `openrouter` and `stub` are supported in the current implementation."
            )
        client = load_openrouter_client(cfg)

        def generation_backend(prompts: list[str], *, progress_desc: str | None = None) -> list[str]:
            return generate_texts_openrouter(
                client,
                cfg.openrouter.generation_model,
                prompts,
                referer=cfg.openrouter.referer,
                title=cfg.openrouter.title,
                temperature=cfg.generation_defaults.temperature,
                max_tokens=cfg.generation_defaults.max_new_tokens,
                progress_desc=progress_desc,
            )

        def judge_backend(prompts: list[str], *, progress_desc: str | None = None) -> list[str]:
            return generate_texts_openrouter(
                client,
                cfg.openrouter.judge_model,
                prompts,
                referer=cfg.openrouter.referer,
                title=cfg.openrouter.title,
                temperature=0.0,
                max_tokens=96,
                progress_desc=progress_desc,
            )

        topics_by_id = {int(row["topic_id"]): row for row in topics}
        templates_by_id = {row["template_id"]: row for row in templates}
        recovered_story_rows = []
        recovered_neutral_rows = []
        fallback_stories_rows = []
        fallback_neutral_rows = []
        accepted_stories, remaining_story_jobs = _generate_with_retries(
            cfg,
            generation_backend,
            items=story_jobs,
            min_story_chars=cfg.story_generation.min_story_chars,
            blocked_lookup=constraints,
            neutral=False,
        )
        accepted_stories = _judge_stories(cfg, judge_backend, accepted_stories)
        accepted_neutral, remaining_neutral_jobs = _generate_with_retries(
            cfg,
            generation_backend,
            items=neutral_jobs,
            min_story_chars=cfg.story_generation.min_story_chars,
            blocked_lookup=None,
            neutral=True,
        )
        rejected_stories = []
        rejected_neutral = []
        if remaining_story_jobs:
            synthetic_rows = []
            for item in remaining_story_jobs:
                synthetic_rows.append(
                    {
                        "sample_id": item["sample_id"],
                        "topic_id": item["topic"]["topic_id"],
                        "template_id": item["template"]["template_id"],
                        "person": item["template"]["person"],
                        "attempt": cfg.story_generation.max_retries,
                        "text": "",
                        "rule_pass": False,
                        "blocked_hits": [],
                        "failure_reasons": ["exhausted_generation_retries"],
                        "emotion": item["emotion"],
                    }
                )
            rec, rem = _recover_rows(
                cfg,
                generation_backend,
                rows=synthetic_rows,
                topics_by_id=topics_by_id,
                templates_by_id=templates_by_id,
                constraints=constraints,
                neutral=False,
            )
            if rec:
                rec = _judge_stories(cfg, judge_backend, rec)
            fallback_rows = _fallback_rows(rem, neutral=False, topics_by_id=topics_by_id, templates_by_id=templates_by_id)
            recovered_story_rows.extend(rec)
            fallback_stories_rows.extend(fallback_rows)
            accepted_stories.extend(rec)
            accepted_stories.extend(fallback_rows)
        if remaining_neutral_jobs:
            synthetic_rows = []
            for item in remaining_neutral_jobs:
                synthetic_rows.append(
                    {
                        "sample_id": item["sample_id"],
                        "topic_id": item["topic"]["topic_id"],
                        "template_id": item["template"]["template_id"],
                        "person": item["template"]["person"],
                        "attempt": cfg.story_generation.max_retries,
                        "text": "",
                        "rule_pass": False,
                        "blocked_hits": [],
                        "failure_reasons": ["exhausted_generation_retries"],
                    }
                )
            rec, rem = _recover_rows(
                cfg,
                generation_backend,
                rows=synthetic_rows,
                topics_by_id=topics_by_id,
                templates_by_id=templates_by_id,
                constraints=constraints,
                neutral=True,
            )
            fallback_rows = _fallback_rows(rem, neutral=True, topics_by_id=topics_by_id, templates_by_id=templates_by_id)
            recovered_neutral_rows.extend(rec)
            fallback_neutral_rows.extend(fallback_rows)
            accepted_neutral.extend(rec)
            accepted_neutral.extend(fallback_rows)

    stories_path = workspace / "raw" / "stories_train.jsonl"
    neutral_path = workspace / "raw" / "neutral_stories.jsonl"
    write_jsonl(stories_path, accepted_stories)
    write_jsonl(neutral_path, accepted_neutral)

    recovered_story_path = workspace / "raw" / "recovered_stories.jsonl"
    recovered_neutral_path = workspace / "raw" / "recovered_neutral_stories.jsonl"
    write_jsonl(recovered_story_path, recovered_story_rows)
    write_jsonl(recovered_neutral_path, recovered_neutral_rows)
    fallback_story_path = workspace / "raw" / "fallback_stories.jsonl"
    fallback_neutral_path = workspace / "raw" / "fallback_neutral_stories.jsonl"
    write_jsonl(fallback_story_path, fallback_stories_rows)
    write_jsonl(fallback_neutral_path, fallback_neutral_rows)

    if not cfg.use_stub_data and ((workspace / "raw" / "rejected_stories.jsonl").exists() or (workspace / "raw" / "rejected_neutral_stories.jsonl").exists()):
        recovery = promote_legacy_corpus_inplace(
            cfg,
            workspace,
            topic_root,
            template_root,
            generator_backend=generation_backend,
            judge_backend=judge_backend,
        )
        accepted_stories = read_jsonl(stories_path)
        accepted_neutral = read_jsonl(neutral_path)
    else:
        recovery = {
            "accepted_stories": len(accepted_stories),
            "accepted_neutral": len(accepted_neutral),
            "recovered_stories": len(recovered_story_rows),
            "recovered_neutral": len(recovered_neutral_rows),
            "fallback_stories": len(fallback_stories_rows),
            "fallback_neutral": len(fallback_neutral_rows),
        }

    health = corpus_health_metrics(accepted_stories, accepted_neutral)
    quality_df = pd.DataFrame(
        [
            {
                "split": "stories",
                "accepted": len(accepted_stories),
                "recovered": recovery["recovered_stories"],
                "fallback": recovery["fallback_stories"],
                "duplicate_fraction": health["story_duplicate_fraction"],
                "fallback_fraction": health["story_fallback_fraction"],
                "cjk_fraction": health["story_cjk_fraction"],
            },
            {
                "split": "neutral",
                "accepted": len(accepted_neutral),
                "recovered": recovery["recovered_neutral"],
                "fallback": recovery["fallback_neutral"],
                "duplicate_fraction": health["neutral_duplicate_fraction"],
                "fallback_fraction": health["neutral_fallback_fraction"],
                "cjk_fraction": health["neutral_cjk_fraction"],
            },
        ]
    )
    if not cfg.use_stub_data:
        if (
            health["story_duplicate_fraction"] > cfg.story_generation.max_duplicate_fraction
            or health["neutral_duplicate_fraction"] > cfg.story_generation.max_duplicate_fraction
            or health["story_fallback_fraction"] > cfg.story_generation.max_fallback_fraction
            or health["neutral_fallback_fraction"] > cfg.story_generation.max_fallback_fraction
            or health["story_cjk_fraction"] > 0.0
            or health["neutral_cjk_fraction"] > 0.0
        ):
            quality_path = workspace / "tables" / "corpus_quality_summary.csv"
            quality_df.to_csv(quality_path, index=False)
            raise RuntimeError(
                "Generated corpus failed health checks. "
                f"story_duplicate_fraction={health['story_duplicate_fraction']:.3f}, "
                f"neutral_duplicate_fraction={health['neutral_duplicate_fraction']:.3f}, "
                f"story_fallback_fraction={health['story_fallback_fraction']:.3f}, "
                f"neutral_fallback_fraction={health['neutral_fallback_fraction']:.3f}, "
                f"story_cjk_fraction={health['story_cjk_fraction']:.3f}, "
                f"neutral_cjk_fraction={health['neutral_cjk_fraction']:.3f}. "
                "Inspect corpus_quality_summary.csv before continuing."
            )

    quality_path = workspace / "tables" / "corpus_quality_summary.csv"
    quality_df.to_csv(quality_path, index=False)

    return {
        "stories_train": str(stories_path),
        "neutral_stories": str(neutral_path),
        "recovered_stories": str(recovered_story_path),
        "recovered_neutral_stories": str(recovered_neutral_path),
        "fallback_stories": str(fallback_story_path),
        "fallback_neutral_stories": str(fallback_neutral_path),
        "corpus_quality_summary": str(quality_path),
    }


def build_parser() -> argparse.ArgumentParser:
    return build_base_parser("Generate synthetic emotion stories and neutral stories from fixed topics and templates")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg, artifact_root, workspace = prepare_context("generate_emotion_corpus", args)
    outputs = run(cfg, workspace.root, artifact_root / "01_topic_bank", artifact_root / "02_prompt_templates")
    save_step_outputs(
        workspace,
        command_name="generate_emotion_corpus",
        cfg=cfg,
        artifact_root=artifact_root,
        input_summary="本步只读取固定 topic bank、4 组手工模板和情绪禁词约束，然后批量生成情绪故事与中性故事。",
        output_summary=f"最终可用的情绪故事与中性故事写入 `{workspace.raw}`，修复与降级统计写入 `{workspace.tables}` 和 `{workspace.raw}`。",
        technique_summary="生成链条固定为：固定模板 prompt -> OpenRouter 外部模型批量生成 -> 硬规则过滤 -> 最多 3 次重试 -> 外部 LLM repair -> 模板 fallback。残差提取仍由本地 Qwen 完成。",
        metrics={
            "emotion_count": cfg.story_generation.emotion_count,
            "topic_count": cfg.topic_bank.topic_count,
            "stories_per_topic": cfg.story_generation.stories_per_topic,
            "neutral_stories_per_topic": cfg.topic_bank.neutral_stories_per_topic,
            "recovery_attempts": cfg.story_generation.recovery_attempts,
        },
        outputs=outputs,
    )


if __name__ == "__main__":
    main()
