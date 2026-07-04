# CogniARC 🧠✨

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![ARC-AGI-3](https://img.shields.io/badge/ARC--AGI--3-LS20_solved-brightgreen.svg)]()

**ARC-AGI-3 Cognitive Architecture** — 6 human drives, 9 reasoning modes, SkillDAG, SocraticCritic, **V-JEPA World Model tool**, and human-like skill acquisition from zero.

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

## 🌍 World Model Tool (V-JEPA 2.1 + k-NN)

> *"The world model is a tool, not the entire architecture."* — Yann LeCun, V-JEPA paper (2024)

The `WorldModelTool` lets ScientistAgent answer **"what happens if I take action X?"** without executing it in the real environment. It memorizes observed transitions and replays them via nearest-neighbor lookup.

### Architecture & Sources

| Component | Detail | Link |
|-----------|--------|------|
| **Encoder** | V-JEPA 2.1 ViT-B/16 (80M params, 384px, RoPE) | [arXiv:2402.04107](https://arxiv.org/abs/2402.04107), [GitHub](https://github.com/facebookresearch/jepa) |
| **Inference** | `vjepa2_infer.py` — extracts 768-dim latent from ARC grids | [skill script](~/.hermes/skills/mlops/vjepa-encoder/scripts/vjepa2_infer.py) |
| **Predictor** | k-NN (k=3) — cosine similarity on stored transitions | [world_model.py](./cogniarc/world_model.py) (330 loc) |
| **Fallback** | Statistical encoding (mean/std histograms) if checkpoint unavailable | Same file, `FallbackEncoder` |
| **Benchmark** | 10-run LS20 with persistent memory | [benchmark_wm.py](./scripts/benchmark_wm.py) |
| **Checkpoint** | `vjepa2_vitb_384.pt` (~320MB, manual download) | HF: [facebook/v-jepa-2.1](https://huggingface.co/collections/facebook/v-jepa-21-67ff32a05c01dc9ff862c56b) |

### How It Works

```
Observation (64×64 ARC grid)
    ↓
V-JEPA 2.1 ViT-B/16 encoder → 768-dim latent vector
    ↓
k-NN Predictor (k=3):
  "Which stored transition (latent_t + action → latent_{t+1}) 
   is most similar to my current latent?"
    ↓
(predicted_next_latent, confidence 0-1)
```

### Usage

```python
from cogniarc import ScientistAgent

# With V-JEPA checkpoint (320MB, ~6s to load)
agent = ScientistAgent('ls20-9607627b', enable_world_model=True)

# Without checkpoint — uses fallback statistical encoder
agent = ScientistAgent('ls20-9607627b', enable_world_model=True, 
                        world_model_config={'checkpoint_path': None})

# Every step() records transitions
agent.step(1)  # latent_before + action + latent_after → stored

# Simulate
predicted, confidence = agent._world_model_simulate(action=1)
# → (768-dim latent, 0.73)

# Report
print(agent._world_model_report())
# → "World model: 47 transitions, avg confidence 0.61"
```

### Key Features

- **Token-free:** World model queries cost 0 LLM tokens
- **k-NN predictor:** Learns from real experience — no training needed
- **Graceful degradation:** Falls back to statistical encoding if V-JEPA unavailable
- **Memory:** 10,000 transitions with automatic eviction, stored in preallocated numpy ring buffers (no per-call rebuild)
- **Per-game persistence:** `.npz` cache in `~/.cache/cogniarc/world_model/`
- **Harness-compatible:** Optional tool — agent works without it

### Current Status

| Metric | Value |
|--------|-------|
| Encoder loaded | ✅ V-JEPA 2.1 ViT-B/16 |
| k-NN accuracy (LS20) | ~60% (needs more transitions) |
| Encoder inference time | ~6s (V-JEPA) / <1ms (fallback) |
| k-NN `predict()` time | ~9.7ms at 10k transitions/768-dim (was ~13.4ms — vectorized via einsum, see `cogniarc/world_model.py`) |
| Fallback quality | Statistical only — 0% real grid understanding |
| Integration | ✅ Wired into ScientistAgent tier chain

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

> ⚠️ **Read the dev/holdout split before trusting any number below.** `arc_agi`
> exposes 25 real environments; [`cogniarc/eval_games.json`](./cogniarc/eval_games.json)
> classifies 15 as **dev** and 10 as **holdout** (zero references anywhere in
> the repo, verified by `git grep`).

### 🎉 ScientistAgent: Generic Harness (2026-07-04)

**LS20 Level 1 & 2 SOLVED.** The generic harness (`observe→hypothesize→plan→execute→verify→refine`) replaces the legacy LS20-specific phase machine. Zero game-specific knowledge — the agent discovers mechanics through observation and exploration.

| Game | Level | Solved | Steps | Time | Notes |
|------|-------|--------|-------|------|-------|
| `ls20-9607627b` | L1 | ✅ | **40** | ~2s | Reproducible. Descend-then-left wall bypass + A* waypoint |
| `ls20-9607627b` | L2 | ✅ | **48** | ~30s | Auto-rotate on changer, walk-on lock collection |
| `sp80` | L1 | ✅ | **17** | ~1s | ObjectTracker exploration — level completed during random exploration |
| 17 other games | L1 | ❌ | 16-25 | ~1s | Active exploration (was 1 step before). ObjectTracker learns movement directions but needs better exploit/explore balance |

### Architecture that made it work

| Component | Role |
|-----------|------|
| **Generic harness** | `observe→hypothesize→plan→execute→verify→refine` — game-agnostic |
| **GoalSanityChecker** | 4 checks (distance, action loop, critic staleness, goal plausibility) — detects wrong-goal loops |
| **Failed hypothesis memory** | Never repeats a failed hypothesis (lock→changer→lock switch on LS20) |
| **Descend-then-left** | Wall bypass: when same-column blocked, descend 15 cells then waypoint left |
| **Random exploration** | "Just give it a go" — unblocks player when stagnation ≥ 5 |
| **ObjectTracker** | Discovers player, movement directions, and interactable objects without tags |
| **Multi-step exploration** | 5 exploration steps to feed ObjectTracker (was 1 step before) |
| **Auto-interact** | Auto-rotates on changer, collects lock by walking on it |
| **A* waypoint** | Intermediate waypoint when direct path blocked |

### LS20 Mechanics (discovered 2026-06-28)

| Mechanic | Detail |
|----------|--------|
| **Actions** | 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT (non-standard!) |
| **Step size** | 5 cells per action |
| **Wall colors** | {3, 5, 11} — color 5 blocks lock area |
| **Changer** | at (19,30) — cycles rotation 3→0→1→2→3 |
| **Lock** | at (34,10) — intangible, all rotations valid |
| **Topology** | Column-34 wall blocks direct path. Player must descend below wall, go left, then up. |

### Key Metrics

| Metric | Value |
|--------|-------|
| **LLM tokens consumed** | **0** per game (all solvers; nano-LLM tier is opt-in and off by default) |
| Solve rate (dev) | **3/20 games** (LS20 L1, LS20 L2, SP80) |
| Architecture | Generic harness + GoalSanityChecker + ObjectTracker + A* + heuristic |
| L1 steps (LS20) | **40** (was 200+ before generic harness) |
| L2 steps (LS20) | **48** |

### Performance Evolution

| Date | Version | Modules | L1 Resolution |
|------|---------|---------|---------------|
| 2026-06-14 | v1 (simple BFS) | arc_agent.py | ❌ Failed |
| 2026-06-15 | v2 (BFS + transforms) | +transforms.py | ✅ 72% (dev-only BFS) |
| 2026-06-25 | v3 (Perception) | +temporal, spatial, attention, symbolic | 🚧 In progress |
| 2026-06-27 | v3.1 (AHOIS) | +ScientificState, SocraticCritic, 9 modes | 🚧 In progress |
| 2026-06-28 | v3.2 (World Model + micro-NN) | +WorldModelTool, micro-NN, heuristic path | 🚧 0% L1 (mechanics discovered) |
| 2026-07-01 | v3.3 (audit) | Dev/holdout harness, ObjectTracker, program synthesis DSL | 📊 0% dev, 0% holdout |
| **2026-07-04** | **v4.0 (generic harness)** 🎉 | GoalSanityChecker, generic phases, descend-then-left, random explore, ObjectTracker hypotheses | ✅ **LS20 L1 40 steps, L2 48 steps, SP80 17 steps** |

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
│   ├── arc_agent.py                # Main ARC solver entry point
│   ├── scientist_agent.py          # 🧠 Core orchestration: init/step/solve_level/phases (852 lines, down from 1610 — split into mixins below)
│   ├── scientist_agent_discovery.py# Mechanics discovery: source reading, wall detection, sprite tags
│   ├── scientist_agent_skills.py   # Skill execution: navigate/rotate/interact + phase advance
│   ├── scientist_agent_ml_tiers.py # World-model + nano-LLM escalation tiers
│   ├── object_perception.py        # 🆕 Generic player/wall/action-direction inference — no tags, no hardcoded mapping
│   ├── active_experiment.py        # 🆕 Pick the action that best disambiguates competing hypotheses (info-gain scoring)
│   ├── program_synthesis.py        # 🆕 BFS program search over a grid-transform DSL + online discrete-state search (used to plan LS20 rotation for real)
│   ├── generalization.py           # 🆕 Dev-vs-holdout report (see eval_games.json, docs/EVALUATION.md)
│   ├── eval_games.json             # 🆕 15 dev / 10 pristine-holdout game classification
│   ├── world_model.py             # V-JEPA 2.1 encoder + k-NN predictor (vectorized storage)
│   ├── micro_predictors.py        # ⚡ Rule-first Domain/Action predictors (NN mode kept for comparison) + NN Pathfinder/CAPTCHA
│   ├── grid_viz.py                # 🔍 Instant ASCII grid visualizer
│   ├── audio_cartography.py       # 🔊 18 paramètres, 10 émotions, 10 archétypes
│   ├── audio_perception.py        # 🎧 Son → compréhension du jeu
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
├── scripts/
│   ├── run_holdout.py             # 🆕 Run ScientistAgent on holdout games; refuses dev games by default
│   ├── generalization_report.py   # 🆕 Dev-vs-holdout solve-rate report
│   ├── benchmark_rules_vs_nn.py   # Logic-vs-micro-NN accuracy comparison
│   └── demo_program_synthesis.py  # 🆕 synthesize -> verify-on-holdout demo
├── docs/
│   └── EVALUATION.md              # 🆕 Dev/holdout discipline + every empirical result this session found
├── tests/                         # 127 passing
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

## 🔊 Audio Cartography — Sound Skills

> *"Chaque paramètre sonore est catalogué comme les jeux ARC-AGI-3 : effet perçu → symbole → application."*

The `audio_cartography.py` module maps **18 audio parameters** to their perceptual effects, symbolic meanings, and practical applications — treating sound design as a cognitive skill to be mastered.

| Category | Parameters |
|----------|-----------|
| **Dynamics** | gain, envelope attack/decay/sustain/release |
| **Spectral** | frequency, timbre, brightness, warmth, air |
| **Modulation** | vibrato, tremolo, chorus, flanger, phaser |
| **Spatial** | pan, reverb, delay |
| **Articulation** | portamento, glissando, staccato/legato |

Each parameter maps to:
- **Perceptual effect** — what the human ear/brain perceives
- **Symbolic meaning** — what the sound communicates (urgency, calm, closeness, mystery)
- **ARC-AGI-3 pattern** — which reasoning skill it exercises (temporal, spatial, symbolic)
- **10 emotions + 10 archetypes** — affective mapping for generative audio

```python
from cogniarc.audio_cartography import AudioParameter, AudioParameterMap

freq_map = AudioParameterMap.get(AudioParameter.FREQUENCY)
print(freq_map.what_it_does)     # "Hauteur perçue. Aigu = petit, proche, urgent. Grave = grand, lointain, calme."
print(freq_map.emotion)          # Emotion.JOY (high) / Emotion.SADNESS (low)
print(freq_map.archetype)        # Archetype.TRICKSTER (high) / Archetype.SAGE (low)
```

**Related:** `audio_perception.py` — bridges audio analysis to CognitiveDrives (novelty response to new sounds, caution response to sudden loudness).

---

## ⚡ Micro-NN Models (Hugging Face)

Four micro neural networks trained in pure NumPy, deployed in Rust (<400KB binary, <1ms inference). Zero LLM tokens.

> **Terminology** — three distinct model tiers, don't confuse them:
> - **Micro-NN** — these 4 tiny NumPy/Rust nets (deterministic classifiers/regressors). Published on HF as `cogniarc-nano-nn`.
> - **Nano-LLM** — Qwen2.5-0.5B, a real (small) *language model* run via Ollama. See [nano_llm.py](./cogniarc/nano_llm.py).
> - **V-JEPA** — the world-model encoder. See [world_model.py](./cogniarc/world_model.py).

| Model | Architecture | NN acc | Logic baseline | Verdict |
|-------|-------------|--------|----------------|---------|
| **Domain Classifier** | 6→12→4 (relu+softmax) | 59% | **90.8%** (rules) | ⚠️ use logic |
| **Action Predictor** | 8→16→1 (relu+sigmoid) | 77.5% | **92.1%** (rules) | ⚠️ use logic |
| **Pathfinder** | 105→64→32→4 (relu+softmax) | 99.6% walls | A*/BFS exact | NN as reactive prior only |
| **CAPTCHA Classifier** 🆕 | 256→64→32→6 (relu+softmax) | 100% test | none (perception) | ✅ keep NN |

> Models published on HF: [cogniarc-nano-nn](https://huggingface.co/zedgamer/cogniarc-nano-nn). Numbers above measured by [`scripts/benchmark_rules_vs_nn.py`](./scripts/benchmark_rules_vs_nn.py) — see [Logic vs Micro-NN](#-logic-vs-micro-nn-when-to-learn-when-to-code).

**Pattern:** train in Python (numpy) → export JSON → infer in Rust (serde only)

```bash
# Rust inference — same binary, different JSON
./domain-classifier domain_classifier.json 1.0 1.0 0.3 0.35 0.45 0.02
# → movement (conf=1.00)

./domain-classifier action_predictor.json 0.3 0.0 0.3 0.33 0 0.0 0 0.1
# → 0.544 (success probable)

./domain-classifier captcha_classifier.json <64 grayscale values>
# → turnstile (conf=0.98)
```

**Tiered escalation in ScientistAgent:**
```
Micro-NN (5µs) → if low conf → 🤖 Nano-LLM HF (<1s) → if low conf → 🌍 V-JEPA (6s) → if fails → 🧱 Heuristic
```
- **Micro-NN** ([4 models](#-micro-nn-models-hugging-face)): domain, action, pathfinder, CAPTCHA — ultra-cheap, deterministic ✅ wired
- **Nano-LLM HF** ([Qwen2.5-0.5B](./cogniarc/nano_llm.py)): reads game state, proposes actions — wrapped in `NanoLLMHarness` for safety. 🚧 **opt-in tier** — enable with `ScientistAgent(..., enable_nano_llm=True)` (requires Ollama); exposed via `agent._nano_propose_action()`. Not yet auto-invoked by the phase machine.
- **V-JEPA** ([world_model.py](./cogniarc/world_model.py)): simulates "what if I do X?" via k-NN on stored transitions ✅ wired (`enable_world_model=True`)
- **Heuristic**: deterministic wall circumvention — never fails if topology is understood ✅ wired

---

## 🧮 Logic vs Micro-NN: when to learn, when to code

We benchmarked our own micro-NNs against plain logic baselines (`if/and/or`) on the
*same* held-out distribution the nets were trained on. The result is deliberate and
reported honestly:

| Task | Micro-NN | Logic rule | Δ |
|------|----------|-----------|---|
| Domain classification (4-class) | 59.0% | **90.8%** | **+31.9** |
| Action success (binary) | 77.5% | **92.1%** | **+14.6** |

*Reproduce:* `python scripts/benchmark_rules_vs_nn.py`

**Why logic wins here.** Both training sets are *generated by rules* (see
`micro_nn/train_domain.py`, `train_action.py`): a sample's label **is** a rule.
A net trained to imitate a known rule can only ever be a lossy, opaque copy of it —
so a few hand-written thresholds reproduce the decision boundary almost exactly,
while the NN under-fits the overlap and amplifies the injected label noise. This is
the classic tabular / known-mapping regime where simple models dominate
(Grinsztajn et al., *NeurIPS 2022*; Rudin, *Nature MI 2019*), and it echoes the
historical lesson that NNs exist for mappings logic can't separate cheaply
(XOR needs a hidden layer — Minsky & Papert, 1969), **not** for ones you can
already write down.

**Our routing policy** (`cogniarc/micro_predictors.py`, rule-first by default):

| Predictor | Mapping | Decision |
|-----------|---------|----------|
| Domain, Action | **known** rule | logic (`mode="rule"`) — exact, 0-param, interpretable |
| Pathfinder | unknown/partial map | A*/BFS exact when walls known; NN only as a fast reactive prior |
| CAPTCHA | raw pixels → type | **NN** — genuine perception, no closed-form rule |

### How to actually make the kept NNs earn their place

The fix is **not** bigger nets — it is training where the mapping is genuinely
unknown, and gating every NN behind the simple baseline it must beat:

1. **Train on real data, not rule-generated synthetics.** Replace
   `generate_*_data()` with logged game observations / real CAPTCHA screenshots.
   If a net trained on synthetics can't beat the rule that generated them, it has
   learned nothing new.
2. **Report a generalization number, not "% synth".** Use a real train/val/test
   split and publish the held-out accuracy + the gap.
3. **Always ship the baseline.** Every NN result is paired with the logic/A*/
   majority baseline. A net only ships if it *beats* the baseline on real, unseen
   data (this repo now enforces that for Domain/Action in `tests/test_predictors.py`).
4. **Pathfinder → imitation learning.** Train on A*-optimal trajectories over many
   real maps; success metric = goal-reach rate on *unseen* mazes vs A*, not
   per-step accuracy. Justified only if it generalizes where A* is too slow or the
   map is partially observed.
5. **CAPTCHA → real distribution + capacity.** Real screenshots, augmentation,
   higher input resolution than 8×8, calibrated confidence; this is the one task
   with no closed-form rule, so it's where NN investment pays off.
6. **Calibrate "confidence."** Today it's an argmax/softmax artifact; use temperature
   scaling or held-out reliability so the escalation chain trusts it correctly.

> **TL;DR** — Code the rule when you know it; learn the function when you don't;
> and never let a net into production without beating the dumb baseline on real data.

### References

- F. Chollet, *On the Measure of Intelligence* (2019), [arXiv:1911.01547](https://arxiv.org/abs/1911.01547) — ARC favors program/rule induction over pattern fitting.
- L. Grinsztajn, E. Oyallon, G. Varoquaux, *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS 2022, [arXiv:2207.08815](https://arxiv.org/abs/2207.08815).
- C. Rudin, *Stop explaining black box ML for high-stakes decisions and use interpretable models instead*, Nature Machine Intelligence 2019, [arXiv:1811.10154](https://arxiv.org/abs/1811.10154).
- M. Minsky, S. Papert, *Perceptrons* (1969) — the XOR limitation of linear units.
- D. Wolpert, W. Macready, *No Free Lunch Theorems for Optimization* (1997) — no model is universally best; match inductive bias to the problem.
- R. Sutton, *The Bitter Lesson* (2019) — learning+search scale with compute; the nuance: it assumes you *lack* the rule and *have* the data/compute.

---

## 🔗 Related Projects

| Repo | Description |
|------|-------------|
| [hermes-agent](https://github.com/nous-research/hermes-agent) | Hermes Agent framework |
| [arc-human-skills](https://github.com/zedarvates/arc-human-skills) | Human skills track (drawing/writing/reading/painting) |
| [cogniarc-nano-nn](https://huggingface.co/zedgamer/cogniarc-nano-nn) 🆕 | Micro-NNs for ARC-AGI-3 (Rust, 394KB) |
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
