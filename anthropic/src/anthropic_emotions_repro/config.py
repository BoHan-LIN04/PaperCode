from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from anthropic_emotions_repro.constants import PROJECT_ROOT
from anthropic_emotions_repro.io import read_yaml


@dataclass
class GenerationDefaults:
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_new_tokens: int = 192
    do_sample: bool = True


@dataclass
class RuntimeConfig:
    dtype: str = "bfloat16"
    attn_impl: str = "sdpa"
    generation_batch_size: int = 4
    extraction_batch_size: int = 4
    judge_batch_size: int = 4
    max_length: int = 256


@dataclass
class OpenRouterConfig:
    enabled: bool = True
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    generation_model: str = "openai/gpt-4o-mini-2024-07-18"
    judge_model: str = "openai/gpt-4o-mini-2024-07-18"
    referer: str = "https://localhost"
    title: str = "anthropic-emotions-repro"
    timeout_seconds: float = 120.0


@dataclass
class TopicBankConfig:
    source_path: str = "datasets/topics_100.yaml"
    topic_count: int = 100
    neutral_stories_per_topic: int = 4


@dataclass
class StoryGenerationConfig:
    template_path: str = "prompts/story_templates.yaml"
    constraints_path: str = "datasets/emotion_constraints.yaml"
    emotion_count: int = 171
    stories_per_topic: int = 12
    max_retries: int = 3
    min_story_chars: int = 120
    qc_emotions: int = 30
    qc_samples_per_emotion: int = 10
    forbid_direct_terms: bool = True
    recovery_attempts: int = 2
    max_duplicate_fraction: float = 0.15
    max_fallback_fraction: float = 0.35
    generation_backend: str = "openrouter"
    judge_backend: str = "openrouter"


@dataclass
class VectorExtractionConfig:
    token_pool_start: int = 50
    neutral_pca_variance: float = 0.5


@dataclass
class SmokeConfig:
    enabled: bool = False


@dataclass
class RunConfig:
    project_name: str = "anthropic-emotions-repro"
    artifact_root: str | None = None
    model_name: str = "Qwen/Qwen3-14B"
    language: str = "en"
    wandb_mode: str = "offline"
    seed: int = 42
    enable_thinking_quant: bool = False
    use_stub_data: bool = False
    main_layer: int = 26
    layer_sweep: list[int] = field(default_factory=lambda: [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34])
    generation_defaults: GenerationDefaults = field(default_factory=GenerationDefaults)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    topic_bank: TopicBankConfig = field(default_factory=TopicBankConfig)
    story_generation: StoryGenerationConfig = field(default_factory=StoryGenerationConfig)
    vector_extraction: VectorExtractionConfig = field(default_factory=VectorExtractionConfig)
    smoke: SmokeConfig = field(default_factory=SmokeConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_artifact_root"] = str(resolve_artifact_root(self))
        return payload


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def _build_config(payload: dict[str, Any]) -> RunConfig:
    return RunConfig(
        project_name=payload.get("project_name", "anthropic-emotions-repro"),
        artifact_root=payload.get("artifact_root"),
        model_name=payload.get("model_name", "Qwen/Qwen3-14B"),
        language=payload.get("language", "en"),
        wandb_mode=payload.get("wandb_mode", "offline"),
        seed=int(payload.get("seed", 42)),
        enable_thinking_quant=bool(payload.get("enable_thinking_quant", False)),
        use_stub_data=bool(payload.get("use_stub_data", False)),
        main_layer=int(payload.get("main_layer", 26)),
        layer_sweep=[int(x) for x in payload.get("layer_sweep", [26])],
        generation_defaults=GenerationDefaults(**payload.get("generation_defaults", {})),
        runtime=RuntimeConfig(**payload.get("runtime", {})),
        openrouter=OpenRouterConfig(**payload.get("openrouter", {})),
        topic_bank=TopicBankConfig(**payload.get("topic_bank", {})),
        story_generation=StoryGenerationConfig(**payload.get("story_generation", {})),
        vector_extraction=VectorExtractionConfig(**payload.get("vector_extraction", {})),
        smoke=SmokeConfig(**payload.get("smoke", {})),
    )


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    default_payload = read_yaml(PROJECT_ROOT / "configs" / "default.yaml")
    override_payload = read_yaml(config_path)
    merged = _merge_dicts(default_payload, override_payload)
    return _build_config(merged)


def resolve_artifact_root(cfg: RunConfig) -> Path:
    if cfg.artifact_root:
        return Path(cfg.artifact_root)
    model_slug = cfg.model_name.split("/")[-1].lower()
    root_name = (
        f"model={model_slug}"
        f"__lang={cfg.language}"
        f"__emotions={cfg.story_generation.emotion_count}"
        f"__topics={cfg.topic_bank.topic_count}"
        f"__stories={cfg.story_generation.stories_per_topic}"
        f"__layer={cfg.main_layer}"
    )
    return PROJECT_ROOT / "artifacts" / root_name
