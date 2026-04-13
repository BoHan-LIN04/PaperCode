from anthropic_emotions_repro.cli import COMMANDS
from anthropic_emotions_repro.constants import STEP_ORDER


def test_old_commands_removed():
    removed = {
        "prepare_public_corpora",
        "run_validation",
        "run_preference_eval",
        "run_geometry_eval",
        "run_dialogue_role_eval",
        "run_blackmail_like_eval",
        "run_reward_hacking_eval",
        "run_sycophancy_eval",
        "run_posttraining_compare",
        "render_figures",
        "export_a_result",
    }
    assert removed.isdisjoint(COMMANDS.keys())


def test_step_order_is_minimal_pipeline():
    assert STEP_ORDER == [
        "00_env",
        "01_topic_bank",
        "02_prompt_templates",
        "03_synthetic_emotion_corpus",
        "04_activation_cache",
        "05_emotion_vectors",
        "report",
    ]
