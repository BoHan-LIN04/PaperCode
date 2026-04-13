import numpy as np

from anthropic_emotions_repro.data.activation_cache import ActivationCacheReader, ActivationCacheWriter


def test_activation_cache_roundtrip(tmp_path):
    writer = ActivationCacheWriter(tmp_path, num_tokens=4, hidden_size=3)
    writer.write_batch(
        activations=np.ones((4, 3), dtype=np.float32),
        sample_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        token_ids=np.array([1, 2, 3, 4], dtype=np.int32),
        token_positions=np.array([0, 1, 0, 1], dtype=np.int32),
    )
    writer.flush()
    writer.write_metadata({"layer_idx": 8, "model_name": "stub"})

    reader = ActivationCacheReader(tmp_path)
    assert reader.activations.shape == (4, 3)
    assert int(reader.sample_ids[0]) == 0
