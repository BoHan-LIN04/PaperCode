# Emotion_grpo

`Emotion_grpo` is a standalone VERL-based research scaffold for GRPO experiments with intrinsic reward.
It now includes both a deterministic random stub reward and an emotion-vector reward bridge that can reuse
the final `emotion_vectors_orth.npy` artifacts produced by `/opt/data/private/lbh/PaperCode/anthropic`.

## What is included

- A clean project layout under `src/emotion_grpo/`
- A wrapper CLI: `python -m emotion_grpo.cli.train --config-name single_gpu_demo`
- JSONL `chat + metadata` input that is converted to VERL parquet automatically
- A VERL-compatible custom reward adapter
- A pluggable `IntrinsicRewardProvider` interface
- A reproducible `RandomIntrinsicRewardProvider`
- An `EmotionVectorRewardProvider` that converts Qwen hidden states into scalar RL reward
- Demo configs, scripts, and minimal tests

## Project layout

```text
Emotion_grpo/
├── configs/
├── data/demo/
├── scripts/
├── src/emotion_grpo/
└── tests/
```

## Install

The default Python environment is `/opt/data/private/lbh/emorlenv/bin/python`.

```bash
cd /opt/data/private/lbh/PaperCode/Emotion_grpo
./scripts/install.sh
```

This script installs:

- This project in editable mode
- A local `codetiming` compatibility shim for proxy-restricted environments
- VERL pinned to commit `1a771692d441ec249bf434060fe6c9859ab28e19`

## Data format

The public input format is JSONL. Each line must contain:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Describe the emotion in the sentence."}
  ],
  "metadata": {
    "id": "demo-1",
    "difficulty": "easy",
    "ground_truth": "joyful",
    "label": "joyful"
  }
}
```

`metadata` is optional and kept stable as the long-term extension field. The wrapper converts this into VERL's
parquet schema and stores the original `messages` and `metadata` inside `extra_info`. The converter prefers
`metadata.ground_truth` and falls back to `metadata.label` when populating the reward target.

## Run

Prepare parquet and launch the default single-GPU demo:

```bash
./scripts/train_single_gpu.sh
```

Run the smaller smoke-test config:

```bash
./scripts/smoke_test.sh
```

Run the emotion-vector reward demo backed by the Qwen3-0.6B artifact from `anthropic`:

```bash
./scripts/train_emotion_vector_qwen3_0_6b.sh
```

Preview the generated VERL command without launching training:

```bash
PYTHONPATH=src /opt/data/private/lbh/emorlenv/bin/python -m emotion_grpo.cli.train \
  --config-name single_gpu_demo \
  --dry-run
```

Preview the emotion-vector reward launch command:

```bash
PYTHONPATH=src /opt/data/private/lbh/emorlenv/bin/python -m emotion_grpo.cli.train \
  --config-name emotion_vector_qwen3_0_6b_demo \
  --dry-run
```

## Reward extension point

Implement a new provider class and switch it in config:

```python
from emotion_grpo.rewards.base import IntrinsicRewardProvider


class MyRewardProvider(IntrinsicRewardProvider):
    def score_batch(self, batch_records, generations, metadata):
        return [0.0 for _ in generations]
```

Then update:

```yaml
reward:
  provider_cls: your_package.your_module.MyRewardProvider
  provider_kwargs:
    some_arg: some_value
```

The stable Python interface is:

```python
IntrinsicRewardProvider.score_batch(batch_records, generations, metadata) -> list[float]
```

## Emotion Vector Reward

The `EmotionVectorRewardProvider` loads:

- `05_emotion_vectors/intermediate/emotion_vectors_orth.npy`
- `05_emotion_vectors/intermediate/vector_metadata.json`
- optionally `04_activation_cache/intermediate/neutral_story_embeddings.npy`

It then:

1. runs a frozen scorer model over the generated response,
2. extracts the hidden state at the configured transformer layer,
3. pools tokens starting from `token_pool_start`,
4. optionally projects out the neutral PCA subspace,
5. compares the pooled embedding to the target emotion vector,
6. converts the similarity or margin into a scalar reward.

The shipped demo config uses:

- anthropic artifact: `/opt/data/private/lbh/PaperCode/anthropic/artifacts/model=qwen3-0.6b__lang=en__emotions=12__topics=8__stories=2__layer=8`
- scorer model: `Qwen/Qwen3-0.6B`
- reward mode: cosine-margin against the target emotion

Make sure your dataset labels match the available artifact emotions. The bundled vector demo uses `hopeful`,
`joyful`, `calm`, and `grateful`, all of which exist in the Qwen3-0.6B vector set.

## Tests

```bash
PYTHONPATH=src /opt/data/private/lbh/emorlenv/bin/python -m pytest tests -q
```
