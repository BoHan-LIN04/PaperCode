from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic_emotions_repro.io import ensure_dir, write_json, write_yaml
from anthropic_emotions_repro.readme_templates import step_readme


@dataclass
class StepWorkspace:
    root: Path
    raw: Path
    intermediate: Path
    tables: Path
    figures: Path
    logs: Path

    @property
    def readme_path(self) -> Path:
        return self.root / "README.md"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def config_path(self) -> Path:
        return self.root / "config.resolved.yaml"

    def write_readme(self, step_title: str, input_summary: str, output_summary: str, technique_summary: str) -> None:
        self.readme_path.write_text(
            step_readme(step_title, input_summary, output_summary, technique_summary),
            encoding="utf-8",
        )

    def write_manifest(self, payload: dict[str, Any]) -> None:
        write_json(self.manifest_path, payload)

    def write_metrics(self, payload: dict[str, Any]) -> None:
        write_json(self.metrics_path, payload)

    def write_config(self, payload: dict[str, Any]) -> None:
        write_yaml(self.config_path, payload)


def create_step_workspace(root: str | Path) -> StepWorkspace:
    root_path = ensure_dir(root)
    raw = ensure_dir(root_path / "raw")
    intermediate = ensure_dir(root_path / "intermediate")
    tables = ensure_dir(root_path / "tables")
    figures = ensure_dir(root_path / "figures")
    logs = ensure_dir(root_path / "logs")
    return StepWorkspace(root=root_path, raw=raw, intermediate=intermediate, tables=tables, figures=figures, logs=logs)


def ensure_step_placeholder(root: str | Path, step_title: str) -> StepWorkspace:
    workspace = create_step_workspace(root)
    if not workspace.readme_path.exists():
        workspace.write_readme(
            step_title=step_title,
            input_summary="该步骤目录已预创建，等待实际运行后写入正式输入说明。",
            output_summary="该步骤目录已预创建，等待实际运行后写入正式输出说明。",
            technique_summary="该占位文件用于保证整棵 artifact 树结构稳定、完整、可浏览。",
        )
    if not workspace.manifest_path.exists():
        workspace.write_manifest({"status": "placeholder", "step_title": step_title})
    if not workspace.metrics_path.exists():
        workspace.write_metrics({"status": "placeholder"})
    if not workspace.config_path.exists():
        workspace.write_config({"status": "placeholder"})
    return workspace
