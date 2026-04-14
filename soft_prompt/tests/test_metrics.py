from soft_prompt_repro.metrics import compute_metrics
from soft_prompt_repro.tasks import ProcessedExample


def test_record_metrics_choose_best_reference():
    examples = [
        ProcessedExample("1", "src", "New York", ["New York", "NYC"], {}),
        ProcessedExample("2", "src", "red", ["red"], {}),
    ]
    metrics = compute_metrics("record", ["NYC", "blue"], examples)

    assert metrics["exact_match"] == 0.5
    assert 0.49 < metrics["f1"] < 0.51


def test_multirc_metrics_include_question_level_em():
    examples = [
        ProcessedExample("1", "src", "true", ["true"], {"group": "a"}),
        ProcessedExample("2", "src", "false", ["false"], {"group": "a"}),
        ProcessedExample("3", "src", "true", ["true"], {"group": "b"}),
    ]
    metrics = compute_metrics("multirc", ["true", "true", "true"], examples)

    assert metrics["exact_match"] == 0.5
    assert 0.79 < metrics["f1_a"] < 0.81