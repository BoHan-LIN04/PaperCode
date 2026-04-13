from anthropic_emotions_repro.config import load_config


def test_smoke_config_loads_minimal_fields():
    cfg = load_config("configs/smoke.yaml")
    assert cfg.model_name == "Qwen/Qwen3-0.6B"
    assert cfg.use_stub_data is True
    assert cfg.story_generation.emotion_count == 12
    assert cfg.topic_bank.topic_count == 8
    assert not hasattr(cfg, "public_corpora")
