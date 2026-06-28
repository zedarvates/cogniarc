# CogniARC 🧠✨

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![ARC-AGI-3](https://img.shields.io/badge/ARC--AGI--3-human--skills-orange.svg)]()

**ARC-AGI-3 Cognitive Architecture** — 6 human drives, 9 reasoning modes, SkillDAG, SocraticCritic, **V-JEPA World Model**, and human-like skill acquisition from zero.

> **Discover, simulate, then solve.** World model as a tool, not the architecture.

---

## 🧭 Two Complementary Tracks

| Track | Repository | Focus |
|-------|------------|-------|
| **Cognitive Solver** | `cogniarc/` (this repo) | ARC-AGI-3 puzzle solving via 6 drives + 9 reasoning modes + SocraticCritic + World Model + Perception Stack |
| **Human Skills** | `arc-human-skills/` | Learn to **read, write, paint like a human from zero** — Watch tutorials → Practice in MS Paint → Self-evaluate → Transfer skills |

Both share the **SkillDAG architecture** (atomic skills + topological dependencies) for composable, transferable learning.

---

## 🏗️ Cognitive Architecture (Solver Track)

```text
CogniARC Solver
├── ScientificState          — structured hypothesis/evidence/assumptions
├── SocraticCritic           — 6 Socratic operations (midwifery)
├── ReasonModeManager        — 9 reasoning modes with automatic selection
├── WorldModelTool 🆕        — V-JEPA 2.1 encoder + k-NN predictor
│   └── "If I do X, what happens?" — simulate without executing
├── Drives (6)
│   ├── novelty, simplicity, doubt, pleasure, caution, impulse
├── Reasoning Modes (9)
│   ├── EXPLORATION, PATHFINDING, ROTATION, TRANSFORMATION
│   ├── GOAL_INFERENCE, CAUSAL, COUNTERFACTUAL, ANALOGICAL, SOCRATIC
├── Perception Stack
│   ├── TemporalReasoner     — ⏱️ time as change (no clock)
│   ├── SpatialReasoner      — 🗺️ space as relations (no ruler)
│   ├── AttentionModel       — focus follows changes
│   └── SymbolicInference    — perception → SkillDAG
└── Dynamic Workflows (6)
    ├── Classify and Act, Fan Out & Synthesize, Adversarial Verification
    ├── Generate and Filter, Tournament, Loop Until Done
```

---

## 🌍 World Model Tool (NEW — June 2026)

> *"The world model is a tool, not the entire architecture. You plug it into your harness."*
> — Inspired by "Einstein World Models" (Discover AI, 2026)

The `WorldModelTool` lets ScientistAgent simulate "what happens if I take action X?" without executing the action in the real environment.

### Architecture

```
Observation (ARC grid) → V-JEPA 2.1 ViT-B/16 (80M, pretrained) → 768-dim latent
                                        ↓
                              k-NN Predictor (k=3)
                              "Which past transition with same action is most similar?"
                                        ↓
                           (predicted_latent, confidence 0-1)
```

### Usage

```python
from cogniarc import ScientistAgent

# Agent with world model enabled
agent = ScientistAgent('ls20-9607627b', enable_world_model=True)

# Every step() automatically records transitions
agent.step(1)  # Move right → latent_before + action + latent_after memorized

# Simulate without executing
predicted, confidence = agent._world_model_simulate(action=1)
print(f"Confidence: {confidence:.0%}")

# Report
print(agent._world_model_report())
# → "World model: 47 transitions memorized"
```

### Key Features
- **Pretrained encoder:** V-JEPA 2.1 ViT-B/16 (80M params, 384px, RoPE) — zero-shot understanding of visual scenes
- **k-NN predictor:** Learns from real experience — no training needed
- **Graceful degradation:** Falls back to statistical encoding if V-JEPA checkpoint unavailable
- **Memory:** Up to 10,000 transitions, automatic eviction
- **Token-free:** World model queries cost 0 LLM tokens
- **Harness-compatible:** The world model is an optional tool — agent works without it

---

## ⏱️ Temporal Inference

> **Time is not an absolute measure. It's the perception of changes between states.**

The `temporal_inference.py` module implements clockless temporal reasoning:
"time" is modeled as a sequence of **DELTAS** (differences between observed states)
and **METACHANGES** (relations between those deltas).

| Pattern | Description | Detection |
|---------|-------------|-----------|
| **CONSTANT** | Same change repeats | Equal magnitudes |
| **ACCELERATING** | Change amplifies | Increasing magnitude |
| **DECELERATING** | Change attenuates | Decreasing magnitude |
| **OSCILLATING** | Change reverses | Added pixels = removed |
| **WAVE** | Change moves | Center of mass shifts |
| **STASIS** | Stable state | Magnitude ~0 |

```python
from cogniarc import TemporalReasoner

r = TemporalReasoner(frames=[grid1, grid2, grid3])
pattern = r.analyze()
print(f"Pattern: {pattern.type.value} (confidence: {pattern.confidence:.0%})")
```

```bash
python -m cogniarc.temporal_inference
```

---

## 🗺️ Spatial Inference

> **Space is not a grid of coordinates. It's a set of relations between objects.**

The `spatial_inference.py` module models space as a **graph of regions**:
no absolute ruler — only `LEFT_OF`, `CONTAINS`, `TOUCHING`, `ALIGNED_H` relations.

**Global patterns:** `SYMMETRY_H`, `SYMMETRY_V`, `GRID`, `CASCADE`, `RAY`, `CHAIN`, `RING`, `CLUSTER`

```python
from cogniarc import SpatialReasoner

sr = SpatialReasoner(grid)
regions = sr.segment()       # -> list[Region]
relations = sr.relate()      # -> list[Relation]
pattern = sr.analyze()       # -> SpatialPattern
```

```bash
python -m cogniarc.spatial_inference
```

---

## 📊 Benchmark Results (ARC-AGI-3)

Results collected on **ls20-9607627b** (navigation game with obstacles).

| Game | Level | Attempts | Solved | Rate | Avg Steps | Avg Time |
|------|-------|----------|--------|------|-----------|----------|
| `ls20-9607627b` | L1 | 76 | **55** | **72%** | 556 | ~0.02s |
| `ls20-9607627b` | L2 | 22 | 0 | 0% | 753 | ~0.03s |

### Key Metrics

| Metric | Value |
|--------|-------|
| **LLM tokens consumed** | **0** per game |
| Total solve time | 2.53s (98 games) |
| Architecture | BFS + deterministic transforms |
| L1 efficiency | 72% success in ~0.02s |
| L2 challenge | Unsolved — needs advanced spatial reasoning |

> **0 tokens used.** The agent solves grids via BFS exploration + transform mapping,
> without any LLM calls. The Perception Stack + World Model target L2 and complex games by guiding search.

### Performance Evolution

| Date | Version | Modules | L1 Resolution |
|------|---------|---------|---------------|
| 2026-06-14 | v1 (simple BFS) | arc_agent.py | ❌ Failed |
| 2026-06-15 | v2 (BFS + transforms) | +transforms.py | ✅ 72% |
| 2026-06-25 | v3 (Perception) | +temporal, spatial, attention, symbolic | 🚧 In progress |
| 2026-06-27 | v3.1 (AHOIS) | +ScientificState, SocraticCritic, 9 modes | 🚧 In progress |
| 2026-06-28 | v3.2 (World Model) 🆕 | +WorldModelTool (V-JEPA 2.1) | 🚧 In progress |

---

## 🎨 Human Skills Track (arc-human-skills)

> **Learn to READ, WRITE, and PAINT like a human — from absolute zero — using Windows Paint, video tutorials, and iterative self-evaluation.**

### Five Learning Levels (Drawing Fundamentals)

| Level | Skills | Description |
|-------|--------|-------------|
| **0 — Line Control** | 13 | Horizontal, vertical, 4 diagonals, pressure fade in/out/wave |
| **1 — 2D Primitives** | 9 | Cross, square, rectangle, X, diamond, △, hexagon, octagon, circle |
| **2 — 3D Wireframes** | 8 | Cube/box (iso/1pt/2pt), pyramid, prism, cylinder, cone |
| **3 — Perspective** | 7 | Horizon/VPs, grids (ground/wall), ellipses, shadows, measuring |
| **4 — Construction** | 3 | Still life, interior corner, building exterior |

**Total: 39 drawing skills** + 12 writing + 4 reading + 12 painting + 7 transfer = **74 atomic skills** in unified DAG.

### Key Features
- **Geometric evaluation** — Angle tolerance ±3°, length ±5%, closure <5px (no vision needed for basics)
- **Real Paint automation** — pywinauto on Windows, headless fallback on Linux
- **Cross-domain transfer** — Strokes→Letters, Primitives→Shapes, Perspective→Scenes
- **SkillDAG mastery** — Skills unlock via 5 attempts with avg ≥80%

---

## 🚀 Quick Start

### CogniARC Solver (This Repo)
```bash
cd ~/projects/cogniarc
pip install -e .
python -m cogniarc.arc_agent --task <arc_task.json>

# With world model (requires V-JEPA checkpoint)
python -c "
from cogniarc import ScientistAgent
agent = ScientistAgent('ls20-9607627b', enable_world_model=True)
"
```

### Windows (Full Training — ODIN-PC)
```cmd
cd C:\Users\redga\projects\arc-human-skills
run_windows.bat
```

### Linux/WSL (Headless Testing)
```bash
cd ~/projects/arc-human-skills
python -m arc_human_skills.trainer --headless --max-sessions 1 --domains drawing
```

---

## 📋 Requirements

| Component | Purpose |
|-----------|---------|
| **Python 3.11+** | Runtime |
| **V-JEPA 2.1 checkpoint** | World model encoder (~320MB, auto-downloaded) |
| **PyTorch + torchvision** | V-JEPA inference (CPU OK, GPU recommended) |
| **LocalAI on EUREKAI (192.168.1.47:8080)** | qwen3.6-27b (vision), whisper-1 (STT), tts-1 (TTS) |
| **Qdrant on EUREKAI (192.168.1.47:6333)** | Vector embeddings |

---

## 📁 Project Structure

```
cogniarc/                          # Cognitive solver (this repo)
├── cogniarc/
│   ├── arc_agent.py               # Main ARC solver entry point
│   ├── scientist_agent.py         # 🧠 Discover → simulate → solve loop
│   ├── world_model.py             # 🆕 V-JEPA 2.1 encoder + k-NN predictor
│   ├── scientific_state.py        # Structured hypothesis/evidence tracking
│   ├── socratic_critic.py         # 6 Socratic operations for hypothesis validation
│   ├── cognitive_player.py        # 6 cognitive drives + game interface
│   ├── pathfinding.py             # A* navigation with walkable overrides
│   ├── skill_tree.py              # Cross-game skill transfer
│   ├── temporal_inference.py      # ⏱️ Time as change patterns
│   ├── spatial_inference.py       # 🗺️ Space as region relations
│   ├── attention.py               # Focus follows changes
│   ├── symbolic_inference.py      # Perception → SkillDAG bridge
│   ├── skill_dag/                 # SkillDAG v2 (atomic skills)
│   ├── benchmark_tracker.py       # JSONL experiment tracking
│   └── goal_inference.py          # Goal hypothesis from observation
├── tests/                         # 31 passing
└── README.md

arc-human-skills/                  # Human skills (separate repo)
├── arc_human_skills/
│   ├── drawing_fundamentals/      # Levels 0-4 + geometric evaluators
│   │   ├── line_control.py        # Level 0: motor control
│   │   ├── primitives_2d.py       # Level 1: 2D shapes
│   │   ├── wireframe_3d.py        # Level 2: 3D wireframes
│   │   ├── perspective.py         # Level 3: perspective
│   │   ├── construction.py        # Level 4: scenes
│   │   ├── eval_utils.py          # 🆕 Extract + evaluate drawn strokes
│   │   └── curriculum.py          # Orchestrator + 74-skill SkillDAG
│   ├── reading/                   # Letter recognition + Qdrant
│   ├── writing/                   # Zaner-Bloser strokes + letters
│   ├── painting/                  # Shapes + Bob Ross landscapes
│   ├── paint_automation.py        # Windows Paint control
│   └── trainer.py                 # Main training loop
├── tests/                         # 70 passed, 10 skipped (Linux)
└── README.md
```

---

## 🔗 Related Projects

| Repo | Description |
|------|-------------|
| [hermes-agent](https://github.com/nous-research/hermes-agent) | Hermes Agent framework |
| [arc-human-skills](https://github.com/zedarvates/arc-human-skills) | Human skills track (drawing/writing/reading/painting) |
| [hermes-fusion](https://github.com/zedarvates/hermes-fusion) | Multi-LLM fusion engine (Rust + Python) |
| [turboquant](https://github.com/zedarvates/turboquant) | Autonomous trading agent |
| [ultimate-odycer](https://github.com/zedarvates/ultimate-odycer) | MMORPG server |

---

## 📚 Documentation

- [World Model Tool](./cogniarc/world_model.py) — V-JEPA encoder + k-NN predictor
- [Einstein World Models (video)](https://youtu.be/tv17bmE2FNY) — World models as tools, not architectures
- [Socratic Agents (AHOIS)](https://arxiv.org/abs/2606.26722) — Paper inspiring SocraticCritic
- [V-JEPA 2.1](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) — Encoder architecture
- [World Models: 5 Approaches](https://themesis.com/2026/01/07/world-models-five-competing-approaches) — Competitive landscape

---

## 📜 License

MIT — See `LICENSE` for details.

---

**Built for ARC-AGI-3** — Advancing cognitive generalization through world models, socratic reasoning, and human-like skill acquisition.
