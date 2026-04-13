from anthropic_emotions_repro.pipeline.generate_emotion_corpus import (
    build_story_prompt,
    build_story_repair_prompt,
    build_neutral_prompt,
    evaluate_rules,
    normalize_story_text,
)


def test_story_prompt_contains_topic_emotion_person_and_block_rules():
    topic = {"topic_id": 1, "title": "Late rent notice", "summary": "A tenant receives a final notice."}
    template = {"template_id": "first_introspective", "person": "first", "style": "introspective", "body": "Write a first-person story."}
    prompt = build_story_prompt(topic, "desperate", template, ["desperate", "frantic"], variant_slot=0)
    assert "Late rent notice" in prompt
    assert "desperate" in prompt
    assert "first" in prompt
    assert "Emotional target label (for guidance only, never print it): desperate" in prompt
    assert "Use English and Latin script only." in prompt
    assert "Never directly name the target emotion or its most obvious synonyms." in prompt
    assert "Use a distinct event sequence and ending for this variant." in prompt


def test_neutral_prompt_contains_english_and_uniqueness_constraints():
    topic = {"topic_id": 1, "title": "Late rent notice", "summary": "A tenant receives a final notice."}
    template = {"template_id": "first_introspective", "person": "first", "style": "introspective", "body": "Write a first-person story."}
    prompt = build_neutral_prompt(topic, template, variant_slot=1)
    assert "Use English and Latin script only." in prompt
    assert "Keep the prose emotionally neutral or low-affect." in prompt
    assert "Use a distinct event sequence and ending for this variant." in prompt


def test_rule_filter_blocks_direct_emotion_terms():
    text = "I felt desperate and frantic as I opened the letter."
    passed, hits, reasons = evaluate_rules(text, ["desperate", "frantic"], min_story_chars=20)
    assert passed is False
    assert "desperate" in hits
    assert "blocked_term" in reasons


def test_rule_filter_blocks_cjk_and_duplicates():
    existing = {normalize_story_text("A narrow hallway forced the choice into his hands.")}
    passed, hits, reasons = evaluate_rules("走进房间。", [], min_story_chars=2, existing_texts=existing)
    assert passed is False
    assert "non_latin_script" in reasons
    passed2, hits2, reasons2 = evaluate_rules("A narrow hallway forced the choice into his hands.", [], min_story_chars=2, existing_texts=existing)
    assert passed2 is False
    assert "duplicate_text" in reasons2


def test_story_repair_prompt_reuses_failed_draft_and_constraints():
    row = {
        "emotion": "desperate",
        "text": "assistant <think>plan</think> I stared at the notice.",
        "failure_reasons": ["think_tag", "too_short"],
    }
    topic = {"topic_id": 1, "title": "Late rent notice", "summary": "A tenant receives a final notice."}
    template = {"template_id": "first_introspective", "person": "first", "style": "introspective", "body": "Write a first-person story."}
    prompt = build_story_repair_prompt(row, topic, template, ["desperate", "frantic"])
    assert "Failed draft:" in prompt
    assert "assistant <think>plan</think> I stared at the notice." in prompt
    assert "The previous draft failed because: think_tag, too_short" in prompt
    assert "Write in English only and use only Latin script." in prompt
    assert "Never directly name the target emotion or its direct synonyms." in prompt
