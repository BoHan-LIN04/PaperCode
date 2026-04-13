from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from anthropic_emotions_repro.artifacts import StepWorkspace, create_step_workspace, ensure_step_placeholder
from anthropic_emotions_repro.config import RunConfig, load_config, resolve_artifact_root
from anthropic_emotions_repro.constants import STEP_ORDER, STEP_TO_DIR
from anthropic_emotions_repro.io import ensure_dir
from anthropic_emotions_repro.utils import set_seed


def build_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--artifact-root", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def prepare_context(command_name: str, args: argparse.Namespace) -> tuple[RunConfig, Path, StepWorkspace]:
    cfg = load_config(args.config)
    if args.artifact_root is not None:
        cfg.artifact_root = args.artifact_root
    if args.wandb_mode is not None:
        cfg.wandb_mode = args.wandb_mode
    if args.seed is not None:
        cfg.seed = args.seed
    artifact_root = resolve_artifact_root(cfg)
    ensure_dir(artifact_root)
    for step_name in STEP_ORDER:
        ensure_step_placeholder(artifact_root / step_name, step_name)
    workspace = create_step_workspace(artifact_root / STEP_TO_DIR[command_name])
    workspace.write_config(cfg.to_dict())
    set_seed(cfg.seed)
    return cfg, artifact_root, workspace


def standard_manifest(command_name: str, cfg: RunConfig, artifact_root: Path, outputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "completed",
        "command": command_name,
        "model_name": cfg.model_name,
        "seed": cfg.seed,
        "wandb_mode": cfg.wandb_mode,
        "artifact_root": str(artifact_root),
        "outputs": outputs,
    }


def save_step_outputs(
    workspace: StepWorkspace,
    *,
    command_name: str,
    cfg: RunConfig,
    artifact_root: Path,
    input_summary: str,
    output_summary: str,
    technique_summary: str,
    metrics: dict[str, Any],
    outputs: dict[str, Any],
) -> None:
    workspace.write_readme(
        step_title=command_name,
        input_summary=input_summary,
        output_summary=output_summary,
        technique_summary=technique_summary,
    )
    workspace.write_manifest(standard_manifest(command_name, cfg, artifact_root, outputs))
    workspace.write_metrics(metrics)


def read_prompt_spec(project_root: Path, relative_path: str) -> dict[str, Any]:
    from anthropic_emotions_repro.io import read_yaml

    return read_yaml(project_root / relative_path)
