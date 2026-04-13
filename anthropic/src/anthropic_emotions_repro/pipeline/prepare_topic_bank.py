from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from anthropic_emotions_repro.constants import PROJECT_ROOT
from anthropic_emotions_repro.io import read_yaml, write_jsonl, write_yaml
from anthropic_emotions_repro.pipeline.common import build_base_parser, prepare_context, save_step_outputs


def run(cfg, workspace: Path) -> dict[str, str]:
    source_path = Path(cfg.topic_bank.source_path)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    payload = read_yaml(source_path)
    topics = []
    for row in tqdm(
        payload["topics"][: cfg.topic_bank.topic_count],
        desc="prepare_topic_bank:freeze_topics",
        total=cfg.topic_bank.topic_count,
        dynamic_ncols=True,
    ):
        topics.append(row)

    yaml_path = workspace / "raw" / "topics.yaml"
    jsonl_path = workspace / "raw" / "topics.jsonl"
    write_yaml(yaml_path, {"topics": topics})
    write_jsonl(jsonl_path, topics)
    return {"topics_yaml": str(yaml_path), "topics_jsonl": str(jsonl_path)}


def build_parser() -> argparse.ArgumentParser:
    return build_base_parser("Prepare the fixed in-repo topic bank")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg, artifact_root, workspace = prepare_context("prepare_topic_bank", args)
    outputs = run(cfg, workspace.root)
    save_step_outputs(
        workspace,
        command_name="prepare_topic_bank",
        cfg=cfg,
        artifact_root=artifact_root,
        input_summary="本步直接读取仓库内固定的 topics_100.yaml，不访问任何外部数据集。",
        output_summary=f"冻结后的 topic bank 同时写入 `{workspace.raw}` 下的 YAML 和 JSONL。",
        technique_summary="topic bank 只做裁剪和冻结，不做在线采样；smoke 模式通过 topic_count 取前 N 条。",
        metrics={"topic_count": cfg.topic_bank.topic_count},
        outputs=outputs,
    )


if __name__ == "__main__":
    main()
