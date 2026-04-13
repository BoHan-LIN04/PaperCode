from pathlib import Path

from anthropic_emotions_repro.constants import PROJECT_ROOT
from anthropic_emotions_repro.data.emotion_lexicon import get_emotion_list
from anthropic_emotions_repro.io import read_yaml


def test_topics_100_yaml_has_exactly_100_topics():
    payload = read_yaml(PROJECT_ROOT / "datasets" / "topics_100.yaml")
    topics = payload["topics"]
    assert len(topics) == 100
    assert len({row["topic_id"] for row in topics}) == 100


def test_story_templates_cover_first_and_third_person():
    payload = read_yaml(PROJECT_ROOT / "prompts" / "story_templates.yaml")
    templates = payload["templates"]
    assert len(templates) == 4
    persons = {row["person"] for row in templates}
    assert persons == {"first", "third"}


def test_emotion_constraints_cover_all_emotions():
    payload = read_yaml(PROJECT_ROOT / "datasets" / "emotion_constraints.yaml")
    constraints = payload["constraints"]
    mapping = {row["emotion"]: row["blocked_terms"] for row in constraints}
    emotions = get_emotion_list(171)
    assert set(emotions).issubset(mapping.keys())
    for emotion in emotions:
        assert emotion in mapping[emotion]
        assert len(mapping[emotion]) >= 2
