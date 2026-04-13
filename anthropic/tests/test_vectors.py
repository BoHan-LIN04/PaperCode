from anthropic_emotions_repro.data.emotion_lexicon import get_emotion_list


def test_emotion_lexicon_counts():
    assert len(get_emotion_list(12)) == 12
    assert len(get_emotion_list(171)) == 171
