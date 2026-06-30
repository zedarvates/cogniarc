#!/usr/bin/env python3
"""Empirical comparison: hand-written logic rules vs the micro-NN classifiers.

Both are evaluated on a FRESH synthetic test set drawn from the *same* generator
the NNs were trained on (micro_nn/train_domain.py, train_action.py). Since the
labels in that generator are produced by explicit rules, a rule classifier that
encodes those rules should match or beat the NN — this script proves it.

Run:  python scripts/benchmark_rules_vs_nn.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cogniarc.micro_predictors import (
    DomainPredictor, ActionPredictor, domain_rule, action_success_rule,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data generators — copied verbatim from micro_nn/train_*.py (label = the rule).
# A different seed is used so this is a held-out test set, not training data.
# ─────────────────────────────────────────────────────────────────────────────
def generate_domain_data(n_samples=4000, seed=2024):
    rng = np.random.default_rng(seed)
    n = n_samples // 4
    X = np.zeros((n_samples, 6)); y = np.zeros(n_samples, dtype=int)
    for cls in range(4):
        s, e = cls * n, cls * n + n
        if cls == 0:    # MOVEMENT
            X[s:e, 0] = rng.uniform(0.5, 1.0, n); X[s:e, 1] = rng.uniform(0.5, 1.0, n)
            X[s:e, 2] = rng.uniform(0.1, 0.4, n); X[s:e, 3] = rng.uniform(0.1, 0.5, n)
            X[s:e, 4] = rng.uniform(0.2, 0.5, n); X[s:e, 5] = rng.uniform(0.005, 0.05, n)
        elif cls == 1:  # ROTATION
            X[s:e, 0] = rng.uniform(0.3, 0.6, n); X[s:e, 1] = X[s:e, 0] * rng.uniform(0.9, 1.1, n)
            X[s:e, 2] = rng.uniform(0.2, 0.6, n); X[s:e, 3] = rng.uniform(0.3, 0.7, n)
            X[s:e, 4] = rng.uniform(0.5, 0.9, n); X[s:e, 5] = rng.uniform(0.03, 0.10, n)
        elif cls == 2:  # TRANSFORM
            X[s:e, 0] = rng.uniform(0.1, 0.35, n); X[s:e, 1] = rng.uniform(0.1, 0.35, n)
            X[s:e, 2] = rng.uniform(0.5, 1.0, n); X[s:e, 3] = rng.uniform(0.5, 1.0, n)
            X[s:e, 4] = rng.uniform(0.1, 0.5, n); X[s:e, 5] = rng.uniform(0.05, 0.25, n)
        else:           # HYBRID
            X[s:e, 0] = rng.uniform(0.2, 0.7, n); X[s:e, 1] = rng.uniform(0.2, 0.7, n)
            X[s:e, 2] = rng.uniform(0.3, 0.7, n); X[s:e, 3] = rng.uniform(0.3, 0.7, n)
            X[s:e, 4] = rng.uniform(0.2, 0.7, n); X[s:e, 5] = rng.uniform(0.02, 0.12, n)
        y[s:e] = cls
    X += rng.normal(0, 0.015, X.shape); X = np.clip(X, 0, 1)
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def generate_action_data(n_samples=4000, seed=2024):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_samples, 8)); y = np.zeros(n_samples)
    for i in range(n_samples):
        px, py = rng.uniform(0, 1, 2); tx, ty = rng.uniform(0, 1, 2)
        dx, dy = tx - px, ty - py
        dist = np.sqrt(dx**2 + dy**2)
        action = rng.integers(1, 5)
        wall = rng.choice([0, 1], p=[0.7, 0.3])
        stagnation = rng.integers(0, 15)
        near_target = 1 if dist < 0.15 else 0
        steps = rng.integers(0, 200)
        if near_target and action in [1, 2, 3, 4]:
            success = 1 if rng.random() > 0.3 else 0
        elif wall and stagnation > 5: success = 0
        elif wall: success = 0
        elif stagnation > 8: success = 0
        elif action == 1 and dx > 0: success = 1
        elif action == 2 and dy > 0: success = 1
        elif action == 3 and dx < 0: success = 1
        elif action == 4 and dy < 0: success = 1
        else: success = 0 if rng.random() > 0.3 else 1
        X[i] = [dx, dy, dist, (action - 1) / 3.0, wall, stagnation / 15.0, near_target, steps / 200.0]
        y[i] = success
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Hand-written logic rules (if / and / or) — encode the same decision boundaries.
# ─────────────────────────────────────────────────────────────────────────────
# Rules under test live in cogniarc.micro_predictors (single source of truth):
#   domain_rule(f) -> (name, conf)   ;   action_success_rule(f) -> 0.0/1.0
def domain_rule_idx(f):
    return DomainPredictor.DOMAINS.index(domain_rule(f)[0])


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  LOGIC RULES vs MICRO-NN  (same held-out synthetic test set)")
    print("=" * 64)

    # ── Domain (4-class) ──
    Xd, yd = generate_domain_data()
    dom_nn = DomainPredictor(mode="nn")
    if dom_nn.available:
        nn_pred = np.array([DomainPredictor.DOMAINS.index(dom_nn.predict(x)[0]) for x in Xd])
        nn_acc = (nn_pred == yd).mean()
    else:
        nn_acc = float("nan")
    rule_pred = np.array([domain_rule_idx(x) for x in Xd])
    rule_acc = (rule_pred == yd).mean()
    print(f"\nDOMAIN  ({len(yd)} samples, 4 classes)")
    print(f"  Micro-NN  : {nn_acc:6.1%}")
    print(f"  Logic rule: {rule_acc:6.1%}   (delta {rule_acc - nn_acc:+.1%})")

    # ── Action (binary) ──
    Xa, ya = generate_action_data()
    act_nn = ActionPredictor(mode="nn")
    if act_nn.available:
        nn_pred = np.array([1 if act_nn.predict(x) >= 0.5 else 0 for x in Xa])
        nn_acc_a = (nn_pred == ya).mean()
    else:
        nn_acc_a = float("nan")
    rule_pred = np.array([1 if action_success_rule(x) >= 0.5 else 0 for x in Xa])
    rule_acc_a = (rule_pred == ya).mean()
    print(f"\nACTION  ({len(ya)} samples, binary; ~contains injected label noise)")
    print(f"  Micro-NN  : {nn_acc_a:6.1%}")
    print(f"  Logic rule: {rule_acc_a:6.1%}   (delta {rule_acc_a - nn_acc_a:+.1%})")
    print()


if __name__ == "__main__":
    main()
