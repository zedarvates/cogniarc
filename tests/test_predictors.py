"""Tests for the rule-first predictors (DomainPredictor, ActionPredictor).

Validates that mode="rule" is the default, always available, matches the same
interface as the NN, and beats the >=90% accuracy bar on the synthetic
distribution the NNs were trained on.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.micro_predictors import (
    DomainPredictor, ActionPredictor, domain_rule, action_success_rule, DOMAINS,
)

# Reuse the held-out generators from the benchmark for ground truth.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from benchmark_rules_vs_nn import generate_domain_data, generate_action_data


# ── Interface / defaults ──────────────────────────────────────────────────────
def test_domain_rule_first_by_default():
    p = DomainPredictor()
    assert p.mode == "rule"
    assert p.available is True  # rules need no checkpoint


def test_action_rule_first_by_default():
    p = ActionPredictor()
    assert p.mode == "rule"
    assert p.available is True


def test_domain_predict_returns_known_label():
    p = DomainPredictor()
    name, conf = p.predict(np.array([1.0, 1.0, 0.3, 0.35, 0.45, 0.02]))  # LS20-ish
    assert name in DOMAINS
    assert name == "movement"
    assert 0.0 <= conf <= 1.0


def test_action_predict_returns_probability():
    p = ActionPredictor()
    prob = p.predict(np.array([0.5, 0.0, 0.5, 0.0, 0, 0.0, 0, 0.1]))  # action1, dx>0
    assert prob == 1.0  # moving right toward target with no wall


def test_action_wall_blocks():
    p = ActionPredictor()
    prob = p.predict(np.array([0.5, 0.0, 0.5, 0.0, 1, 0.0, 0, 0.1]))  # wall=1
    assert prob == 0.0


# ── Accuracy bars ─────────────────────────────────────────────────────────────
def test_domain_rule_accuracy_above_90():
    X, y = generate_domain_data()
    pred = np.array([DOMAINS.index(domain_rule(x)[0]) for x in X])
    acc = (pred == y).mean()
    assert acc >= 0.90, f"domain rule accuracy {acc:.1%} < 90%"


def test_action_rule_accuracy_above_90():
    X, y = generate_action_data()
    pred = np.array([1 if action_success_rule(x) >= 0.5 else 0 for x in X])
    acc = (pred == y).mean()
    assert acc >= 0.90, f"action rule accuracy {acc:.1%} < 90%"


# ── Rule beats NN on the same data (the whole point) ──────────────────────────
def test_rule_beats_nn_when_available():
    X, y = generate_domain_data()
    rule_acc = (np.array([DOMAINS.index(domain_rule(x)[0]) for x in X]) == y).mean()
    nn = DomainPredictor(mode="nn")
    if not nn.available:
        return  # NN checkpoint absent — nothing to compare, rule bar already covered
    nn_acc = (np.array([DOMAINS.index(nn.predict(x)[0]) for x in X]) == y).mean()
    assert rule_acc >= nn_acc, f"rule {rule_acc:.1%} should beat nn {nn_acc:.1%}"
