from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from anthropic_emotions_repro.constants import PROJECT_ROOT
from anthropic_emotions_repro.io import read_yaml, write_yaml
from anthropic_emotions_repro.pipeline.common import build_base_parser, prepare_context, save_step_outputs


def run(cfg, workspace: Path) -> dict[str, str]:
    templates_path = PROJECT_ROOT / cfg.story_generation.template_path
    constraints_path = PROJECT_ROOT / cfg.story_generation.constraints_path
    templates = read_yaml(templates_path)
    constraints = read_yaml(constraints_path)

    frozen_templates = []
    for row in tqdm(
        templates["templates"],
        desc="prepare_prompt_templates:freeze_templates",
        total=len(templates["templates"]),
        dynamic_ncols=True,
    ):
        frozen_templates.append(row)

    frozen_constraints = []
    for row in tqdm(
        constraints["constraints"][: cfg.story_generation.emotion_count],
        desc="prepare_prompt_templates:freeze_constraints",
        total=cfg.story_generation.emotion_count,
        dynamic_ncols=True,
    ):
        frozen_constraints.append(row)

    out_templates = workspace / "raw" / "story_templates.yaml"
    out_constraints = workspace / "raw" / "emotion_constraints.yaml"
    write_yaml(out_templates, {"templates": frozen_templates})
    write_yaml(out_constraints, {"constraints": frozen_constraints})
    return {"story_templates": str(out_templates), "emotion_constraints": str(out_constraints)}


def build_parser() -> argparse.ArgumentParser:
    return build_base_parser("Prepare the fixed prompt templates and blocked-term constraints")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg, artifact_root, workspace = prepare_context("prepare_prompt_templates", args)
    outputs = run(cfg, workspace.root)
    save_step_outputs(
        workspace,
        command_name="prepare_prompt_templates",
        cfg=cfg,
        artifact_root=artifact_root,
        input_summary="本步读取仓库内固定的 4 组故事模板和完整情绪禁词约束文件。",
        output_summary=f"模板和禁词约束会冻结到 `{workspace.raw}` 里，供故事生成阶段直接消费。",
        technique_summary="模板和禁词约束都版本化在仓库内，不依赖任何外部数据源或在线组装逻辑。",
        metrics={"template_count": 4, "emotion_count": cfg.story_generation.emotion_count},
        outputs=outputs,
    )


if __name__ == "__main__":
    main()
