"""Tests for WorldModelTool's vectorized k-NN predictor.

Runs on the fallback (statistical) encoder — no torch/checkpoint required.
The key test proves the vectorized predict() is numerically equivalent to the
original Python-loop reference implementation.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cogniarc.world_model import WorldModelTool, WorldModelConfig


def _reference_predict(wm, observation, action):
    """Original O(N*D) loop implementation — ground truth for equivalence."""
    current_latent = wm.encode(observation)
    candidates = [(ml, mn) for ml, ma, mn in wm.memory if ma == action]
    if not candidates:
        return current_latent.copy(), 0.0
    distances = []
    for mem_latent, _ in candidates:
        dot = np.dot(current_latent, mem_latent)
        norm = np.linalg.norm(current_latent) * np.linalg.norm(mem_latent) + 1e-8
        distances.append(1.0 - dot / norm)
    k = min(wm.config.knn_k, len(candidates))
    top = np.argsort(distances)[:k]
    weights = 1.0 / (np.array([distances[i] for i in top]) + 1e-8)
    weights = weights / weights.sum()
    predicted = np.zeros_like(current_latent)
    for idx, w in zip(top, weights):
        predicted += w * candidates[idx][1]
    confidence = 1.0 / (1.0 + distances[top[0]])
    return predicted, float(confidence)


def _populated_wm(seed=0, n=200):
    wm = WorldModelTool(config=WorldModelConfig(checkpoint_path="/does/not/exist"))
    assert not wm.available  # fallback encoder
    rng = np.random.default_rng(seed)
    for _ in range(n):
        a = int(rng.integers(1, 5))
        wm.remember(rng.normal(size=768), a, rng.normal(size=768))
    return wm


def test_empty_memory_returns_zero_confidence():
    wm = WorldModelTool(config=WorldModelConfig(checkpoint_path="/does/not/exist"))
    grid = np.zeros((8, 8), dtype=int)
    pred, conf = wm.predict(grid, action=1)
    assert conf == 0.0
    assert pred.shape == (768,)


def test_predict_shape_and_confidence_range():
    wm = _populated_wm()
    grid = (np.arange(64).reshape(8, 8) % 10)
    pred, conf = wm.predict(grid, action=2)
    assert pred.shape == (768,)
    assert 0.0 <= conf <= 1.0


def test_vectorized_matches_reference():
    wm = _populated_wm(seed=42, n=300)
    grid = (np.arange(64).reshape(8, 8) % 10)
    for action in (1, 2, 3, 4):
        pred_v, conf_v = wm.predict(grid, action)
        pred_r, conf_r = _reference_predict(wm, grid, action)
        assert np.allclose(pred_v, pred_r, atol=1e-9), f"action {action}: prediction mismatch"
        assert abs(conf_v - conf_r) < 1e-9, f"action {action}: confidence mismatch"


def test_action_with_no_matching_transitions():
    wm = WorldModelTool(config=WorldModelConfig(checkpoint_path="/does/not/exist"))
    wm.remember(np.ones(768), action=1, next_latent=np.ones(768))
    grid = np.zeros((8, 8), dtype=int)
    pred, conf = wm.predict(grid, action=6)  # no action-6 transitions stored
    assert conf == 0.0
