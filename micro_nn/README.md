---
license: mit
language: en
tags:
- arc-agi-3
- micro-nn
- rust
- numpy
- tiny-ml
- domain-classification
- action-prediction
- embedded
- token-free
pipeline_tag: text-classification
---

# CogniARC Nano-NN — Micro Neural Networks for ARC-AGI-3

Tiny feedforward neural networks trained in pure NumPy, deployed in Rust (394KB binary, <1ms inference). Zero LLM tokens — replace trivial agent decisions with deterministic 5µs classifiers.

**Pattern:** train in Python (numpy) → export JSON → infer in Rust (serde only)

## Models

| Model | Architecture | Accuracy | Use Case |
|-------|-------------|----------|----------|
| **Domain Classifier** | 6→12→4 (relu+softmax) | 75% synth, 3/4 games | Classify ARC game type: movement/rotation/transform/hybrid |
| **Action Predictor** | 8→16→1 (relu+sigmoid) | 77.6% test | Predict if an action will succeed (replaces V-JEPA 6s → 5µs) |

## Domain Classifier

```python
# 6 input features from game grid
features = [grid_w/64, grid_h/64, n_colors/10, entropy, spatial_var, objects_ratio]
# → ["movement", "rotation", "transform", "hybrid"]
```

| Feature | Description |
|---------|-------------|
| grid_w | Grid width (normalized 0-1, 1=64px) |
| grid_h | Grid height (normalized) |
| n_colors | Unique colors / 10 |
| entropy | Shannon entropy of color distribution |
| spatial_var | Spread of colored cells |
| objects_ratio | Connected components / total cells |

## Action Predictor

```python
# 8 input features from game state + action
features = [dx, dy, dist, action_norm, wall_between, stagnation, near_target, steps]
# → success probability 0-1
```

| Feature | Description |
|---------|-------------|
| dx, dy | Normalized direction to target |
| dist | Distance to target (normalized) |
| action_norm | Action 1-4 → 0-1 |
| wall_between | 1 if wall color between player and target |
| stagnation | Stagnation counter / 15 |
| near_target | 1 if adjacent to target |
| steps | Steps taken / 200 |

## Usage (Python)

```python
import json, numpy as np

# Load model
with open('domain_classifier.json') as f:
    data = json.load(f)

# Forward pass (numpy)
x = np.array([1.0, 1.0, 0.3, 0.35, 0.45, 0.02])  # LS20 features
for w, b, act in zip(data['weights'], data['biases'], data['activations']):
    w = np.array(w).reshape(data['layers'][i+1], data['layers'][i])
    x = x @ w.T + np.array(b)
    x = np.maximum(0, x) if act == 'relu' else x
# Softmax
ex = np.exp(x - np.max(x))
probs = ex / ex.sum()
print(f"Domain: {['movement','rotation','transform','hybrid'][np.argmax(probs)]}")
```

## Usage (Rust)

```bash
# Build
cd micro_nn && cargo build --release

# Predict domain
./target/release/domain-classifier domain_classifier.json 1.0 1.0 0.3 0.35 0.45 0.02
# → movement (conf=1.0000)

# Predict action success
./target/release/domain-classifier action_predictor.json 0.3 0.0 0.3 0.33 0 0.0 0 0.1
# → 0.544 (success probable)
```

Same binary, different JSON — architecture auto-detected from layer dimensions.

## Training

```bash
# Domain classifier
python3 train_domain.py

# Action predictor
python3 train_action.py
```

Pure NumPy, 0 ML dependencies. Mini-batch SGD, Xavier init, ~500 epochs.

## Tiered Escalation (Agent Pipeline)

```
Input → [Micro-NN, 5µs, 0 tokens]
           ↓ confidence < 0.7
        [World Model V-JEPA, 6s, 0 tokens]
           ↓ still uncertain
        [LLM, 1s, ~200 tokens]
```

80% of decisions at 5µs, 0 tokens.

## Related

- [CogniARC](https://github.com/zedarvates/cogniarc) — Full ARC-AGI-3 agent
- [botte-nano-nn](https://huggingface.co/zedgamer/botte-nano-nn) — Same pattern for agent pipeline decisions
- [V-JEPA 2.1](https://github.com/facebookresearch/jepa) — Encoder architecture used in WorldModelTool
