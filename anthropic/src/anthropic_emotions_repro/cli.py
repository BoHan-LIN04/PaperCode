from __future__ import annotations

import argparse
import sys

from anthropic_emotions_repro.pipeline import (
    build_emotion_vectors,
    build_report,
    extract_residuals,
    generate_emotion_corpus,
    prepare_prompt_templates,
    prepare_topic_bank,
    run_vector_analysis,
)


COMMANDS = {
    "prepare_topic_bank": prepare_topic_bank.main,
    "prepare_prompt_templates": prepare_prompt_templates.main,
    "generate_emotion_corpus": generate_emotion_corpus.main,
    "extract_residuals": extract_residuals.main,
    "build_emotion_vectors": build_emotion_vectors.main,
    "build_report": build_report.main,
    "run_vector_analysis": run_vector_analysis.main,
}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Unified CLI for the minimal anthropic-emotions-repro project")
    parser.add_argument("command", choices=sorted(COMMANDS))
    known, remaining = parser.parse_known_args(argv)
    COMMANDS[known.command](remaining)


def prepare_topic_bank_main() -> None:
    prepare_topic_bank.main()


def prepare_prompt_templates_main() -> None:
    prepare_prompt_templates.main()


def generate_emotion_corpus_main() -> None:
    generate_emotion_corpus.main()


def extract_residuals_main() -> None:
    extract_residuals.main()


def build_emotion_vectors_main() -> None:
    build_emotion_vectors.main()


def build_report_main() -> None:
    build_report.main()


def run_vector_analysis_main() -> None:
    run_vector_analysis.main()


if __name__ == "__main__":
    main()
