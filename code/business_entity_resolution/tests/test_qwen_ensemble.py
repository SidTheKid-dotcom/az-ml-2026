"""CPU tests for the C2 + Qwen ensemble helpers (synthetic arrays only)."""
from __future__ import annotations

import numpy as np

from ber import qwen_ensemble as qe
from ber.pair_tree_c2 import c2_blend, c2_features


def test_blend3_without_qwen_is_the_c2_blend():
    rng = np.random.default_rng(0)
    c2, q, base = rng.random(50).astype(np.float32), rng.random(50).astype(np.float32), rng.random(50)
    base[:10] = 0.001
    c2[3] = np.nan
    for w in (0.5, 0.9, 1.0):
        expect = c2_blend(np.where(np.isnan(c2), -1.0, c2), base, w)
        assert np.allclose(qe.blend3(c2, q, base, w, 0.0), expect)
    mixed = qe.blend3(c2, q, base, 0.6, 0.25)
    gated = base >= qe.GATE
    assert np.all(mixed[~gated] == 0)
    s_c2 = np.nan_to_num(c2, nan=0.0)
    assert np.allclose(mixed[gated], (0.6 * s_c2 + 0.25 * q + 0.15 * base)[gated], atol=1e-6)


def test_q_features_reuse_the_c2_block_and_add_agreement():
    left = np.array([0, 0, 1, 1, 2])
    right = np.array([0, 1, 1, 2, 3])
    base = np.array([.5, .5, .5, .5, .005])
    c2 = np.array([.9, .1, .8, np.nan, np.nan], dtype=np.float32)
    q = np.array([.7, .2, .6, .4, np.nan], dtype=np.float32)
    x = qe.q_features(left, right, c2, q, base, 3, 4)
    assert x.shape == (5, len(qe.Q_FEATURES))
    assert np.allclose(x[:, :8], c2_features(left, right, q, base, 3, 4))
    s_c2 = np.where(np.isnan(c2), -1.0, c2)
    s_q = np.where(np.isnan(q), -1.0, q)
    assert np.allclose(x[:, 8], (s_c2 + s_q) / 2) and np.allclose(x[:, 10], s_c2 - s_q)
    assert x[1, 5] == np.float32(.6)  # record 1's best rival claimant under Qwen is row 2


def test_blend_weights_stay_on_the_simplex():
    weights = qe.blend_weights()
    assert weights and all(a + b <= 1 + 1e-9 and a > 0 and b >= 0 for a, b in weights)
    assert (0.9, 0.0) in weights and (0.3, 0.6) in weights
