# CogniARC 🧠✨

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
[![ARC-AGI-3](https://img.shields.io/badge/ARC--AGI--3-human--skills-orange.svg)]()

**ARC-AGI-3 Cognitive Architecture** — 6 human drives, 9 reasoning modes, SkillDAG orchestration, and **human-like skill acquisition from zero** (drawing fundamentals → writing → reading → painting).

> **Discovers then solves.** No brute force, no templates — an architecture that generalizes.

---

## 🧭 Two Complementary Tracks

| Track | Repository | Focus |
|-------|------------|-------|
| **Cognitive Solver** | `cogniarc/` (this repo) | ARC-AGI-3 puzzle solving via 6 drives + 9 reasoning modes + dynamic workflows |
| **Human Skills** | `arc-human-skills/` | Learn to **read, write, paint like a human from zero** — Watch tutorials → Practice in MS Paint → Self-evaluate → Transfer skills |

Both share the **SkillDAG architecture** (atomic skills + topological dependencies) for composable, transferable learning.

---

## 🏗️ Cognitive Architecture (Solver Track)

```text
CogniARC Solver
├── Drives (6)
│   ├── Curiosity       — Explore the unknown
│   ├── Pattern_Match   — Recognize visual patterns
│   ├── Causal          — Understand cause-effect
│   ├── Efficiency      — Seek simplest solution
│   ├── Memory          — Retain past patterns
│   └── Verify          — Validate and self-correct
├── Reasoning Modes (9)
│   ├── Symbolic        — Symbol manipulation
│   ├── Spatial         — Spatial reasoning
│   ├── Temporal        — Sequential reasoning
│   ├── Analogical      — Analogy & transfer
│   ├── Causal          — Causal chains
│   ├── Compositional   — Pattern composition
│   ├── Relational      — Object relations
│   ├── Probabilistic   — Uncertainty estimation
│   └── Meta            — Reasoning about reasoning
└── Dynamic Workflows (6)
    ├── Classify and Act         — Type classification → strategy
    ├── Fan Out & Synthesize     — Parallel hypotheses
    ├── Adversarial Verification — Skeptic agent review
    ├── Generate and Filter      — N candidates → best
    ├── Tournament               — Pairwise comparison
    └── Loop Until Done          — Iterate to verification
```

---

## 🎨 Human Skills Track (arc-human-skills)

> **Learn to READ, WRITE, and PAINT like a human — from absolute zero — using Windows Paint, video tutorials, and iterative self-evaluation.**

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SKILL DAG ORCHESTRATOR                    │
│  (Topological scheduling, prerequisites, mastery tracking)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ DRAWING │    │ WRITING │    │ READING │
   └────┬────┘    └────┬────┘    └────┬────┘
        │              │              │
        └──────────────┼──────────────┘
                       ▼
            ┌─────────────────────┐
            │     PAINTING        │
            │  (Bob Ross style)   │
            └─────────────────────┘
```

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

- **Embodied practice** — Real drawing in MS Paint via pywinauto automation
- **Self-supervision** — LocalAI vision (qwen3.6-27b) evaluates own output
- **Geometric evaluation** — Angle tolerance ±3°, length ±5%, closure <5px (no vision needed for basics)
- **Cross-domain transfer** — Strokes→Letters, Primitives→Shapes, Perspective→Scenes
- **Continuous learning** — SkillDAG unlocks skills via mastery (5 attempts, avg ≥80%)
- **Tutorial-driven** — YouTube → Whisper transcription → Key frame extraction → Practice

---

## 🚀 Quick Start

### Windows (Full Training — ODIN-PC)
```cmd
cd C:\Users\redga\projects\arc-human-skills
run_windows.bat
# Or PowerShell with options:
.\run_windows.ps1 -Mode train -MaxSessions 10 -Duration 30 -Domains drawing,writing,reading
```

### Linux/WSL (Headless Testing)
```bash
cd ~/projects/arc-human-skills
python -m arc_human_skills.trainer --headless --max-sessions 1 --domains drawing
python -m arc_human_skills.benchmark
```

### CogniARC Solver (This Repo)
```bash
cd ~/projects/cogniarc
pip install -e .
python -m cogniarc.arc_agent --task <arc_task.json>
```

---

## 📋 Requirements

| Component | Purpose |
|-----------|---------|
| **Windows 10/11** | MS Paint automation (pywinauto) |
| **Python 3.11+** | Runtime |
| **LocalAI on EUREKAI (192.168.1.47:8080)** | qwen3.6-27b (vision), whisper-1 (STT), tts-1 (TTS) |
| **Qdrant on EUREKAI (192.168.1.47:6333)** | Vector embeddings for letter templates |
| **NVIDIA GPU** | For LocalAI inference |

---

## 📁 Project Structure

```
cogniarc/                          # Cognitive solver (this repo)
├── cogniarc/
│   ├── arc_agent.py               # Main ARC solver entry point
│   ├── scientist_agent.py         # Hypothesis generation (legacy)
│   ├── skill_dag/                 # SkillDAG v2 (17 atomic skills)
│   ├── triarchic_engine.py        # Sternberg WICS model
│   ├── representation_engine.py   # Visual representation
│   ├── cognitive_player.py        # Game playing interface
│   └── *_inference.py             # Transform, goal, domain, causal
├── tests/
└── README.md

arc-human-skills/                  # Human skills (separate repo)
├── arc_human_skills/
│   ├── config.py                  # Typed configuration
│   ├── paint_automation.py        # Windows Paint control
│   ├── video_tutorial.py          # YouTube → Whisper → keyframes
│   ├── trainer.py                 # Main training loop (CLI)
│   ├── benchmark.py               # ARC-style evaluation
│   ├── drawing_fundamentals/      # Levels 0-4 (NEW)
│   │   ├── line_control.py        # Level 0: motor control
│   │   ├── primitives_2d.py       # Level 1: 2D shapes
│   │   ├── wireframe_3d.py        # Level 2: 3D wireframes
│   │   ├── perspective.py         # Level 3: perspective
│   │   ├── construction.py        # Level 4: scenes
│   │   └── curriculum.py          # Orchestrator + SkillDAG
│   ├── reading/                   # Letter recognition
│   ├── writing/                   # Stroke patterns + letters
│   ├── painting/                  # Shapes + Bob Ross landscapes
│   ├── skill_dag/                 # Unified manifest (74 skills)
│   └── eval_tasks/                # Benchmark tasks
├── tests/                         # 68 passing (10 skipped on Linux)
├── run_windows.bat / .ps1         # Windows launchers
└── README.md
```

---

## 🔗 Related Projects

| Repo | Description |
|------|-------------|
| [hermes-brain](https://github.com/zedarvates/hermes-brain) | Hermes Agent architecture |
| [arc-human-skills](https://github.com/zedarvates/arc-human-skills) | Human skills track (drawing/writing/reading/painting) |
| [hermes-fusion](https://github.com/zedarvates/hermes-fusion) | Multi-LLM fusion engine (4 providers, 4 strategies) |


---

## 📚 Documentation

- [ARC-AGI-3 Human Skills Guide](https://github.com/zedarvates/arc-human-skills/blob/master/WINDOWS_RUNNER.md)
- [SkillDAG Architecture](https://github.com/zedarvates/arc-human-skills/tree/master/arc_human_skills/skill_dag)
- [Drawing Fundamentals Curriculum](https://github.com/zedarvates/arc-human-skills/tree/master/arc_human_skills/drawing_fundamentals)

---

## 📜 License

MIT — See `LICENSE` for details.

---

**Built for ARC-AGI-3** — Advancing human-like skill acquisition and cognitive generalization.


---

[![Donate](https://img.shields.io/badge/☕%20Soutenir-BTC%20%7C%20ETH-orange)](DONATE.md)