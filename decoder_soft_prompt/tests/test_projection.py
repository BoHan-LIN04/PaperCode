from __future__ import annotations

import numpy as np

from decoder_soft_prompt_repro.projection import fit_linear_projection


def test_fit_linear_projection_recovers_simple_linear_map():
    source = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float32,
    )
    true_projection = np.asarray(
        [
            [2.0, 3.0],
            [5.0, 7.0],
        ],
        dtype=np.float32,
    )
    target = source @ true_projection

    fitted = fit_linear_projection(source, target, ridge_alpha=1e-8)

    assert np.allclose(fitted, true_projection, atol=1e-4)