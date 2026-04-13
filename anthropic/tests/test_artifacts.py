from pathlib import Path

from anthropic_emotions_repro.artifacts import create_step_workspace


def test_workspace_contains_standard_subdirs(tmp_path: Path):
    ws = create_step_workspace(tmp_path / "01_topic_bank")
    assert ws.raw.exists()
    assert ws.intermediate.exists()
    assert ws.tables.exists()
    assert ws.figures.exists()
    assert ws.logs.exists()
