import os
import subprocess
from pathlib import Path


def test_smoke_pipeline_prepare_to_vectors(tmp_path: Path):
    root = Path("/opt/data/private/lbh/anthropic")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    python_bin = "/opt/data/private/multifuse/bin/python"

    for command in [
        "prepare_topic_bank",
        "prepare_prompt_templates",
        "generate_emotion_corpus",
        "extract_residuals",
        "build_emotion_vectors",
    ]:
        subprocess.run(
            [
                python_bin,
                "-m",
                "anthropic_emotions_repro.cli",
                command,
                "--config",
                str(root / "configs" / "smoke.yaml"),
                "--artifact-root",
                str(tmp_path / "artifacts"),
            ],
            check=True,
            env=env,
        )

    assert (tmp_path / "artifacts" / "05_emotion_vectors" / "intermediate" / "emotion_vectors_orth.npy").exists()
