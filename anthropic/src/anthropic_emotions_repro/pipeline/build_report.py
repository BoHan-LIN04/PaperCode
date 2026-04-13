from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from anthropic_emotions_repro.constants import STEP_ORDER
from anthropic_emotions_repro.io import read_json, write_json
from anthropic_emotions_repro.pipeline.common import build_base_parser, prepare_context, save_step_outputs


def run(workspace: Path, artifact_root: Path) -> dict[str, str]:
    parity_rows = []
    for step in tqdm(STEP_ORDER, desc="build_report:scan_steps", total=len(STEP_ORDER), dynamic_ncols=True):
        if step == "00_env":
            continue
        manifest_path = artifact_root / step / "manifest.json"
        ready = False
        status = "missing"
        if manifest_path.exists():
            payload = read_json(manifest_path)
            status = payload.get("status", "completed")
            ready = status != "placeholder"
        parity_rows.append({"step_dir": step, "exists": ready, "status": status, "manifest": str(manifest_path)})

    parity_index_path = workspace / "parity_index.json"
    write_json(parity_index_path, parity_rows)

    md_lines = [
        "# Minimal Pipeline Report",
        "",
        "## Steps",
        "",
    ]
    for row in parity_rows:
        md_lines.append(f"- `{row['step_dir']}`: {'ready' if row['exists'] else row['status']}")
    md_lines.extend(["", "## Summary", "", "- Topic bank, prompt templates, synthetic corpus, activation cache, and emotion vectors are the only retained stages."])
    report_md = workspace / "report.md"
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"parity_index": str(parity_index_path), "report_md": str(report_md)}


def build_parser() -> argparse.ArgumentParser:
    return build_base_parser("Build a compact report for the minimal pipeline")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg, artifact_root, workspace = prepare_context("build_report", args)
    outputs = run(workspace.root, artifact_root)
    save_step_outputs(
        workspace,
        command_name="build_report",
        cfg=cfg,
        artifact_root=artifact_root,
        input_summary="本步遍历最小主线的各步骤目录，检查 manifest 是否存在并生成汇总报告。",
        output_summary=f"parity index 与 Markdown 汇总写入 `{workspace.root}`。",
        technique_summary="报告现在只覆盖最小主线，不再引用任何 held-out、评测或论文扩展模块。",
        metrics={"step_count": len(STEP_ORDER) - 1},
        outputs=outputs,
    )


if __name__ == "__main__":
    main()
