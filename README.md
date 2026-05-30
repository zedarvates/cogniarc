# CogniArc — Cognitive ARC-AGI-3 Agent Framework

**Discover. Reason. Solve.** — A human-inspired agent framework for the [ARC-AGI-3](https://arcprize.org) benchmark.

Built during the ARC-AGI-3 competition (May 2026), CogniArc approaches AI reasoning NOT as raw pattern matching, but through the lens of human cognitive science: 6 psychological drives, 9 historical reasoning modes, and a scientist-like discover-then-solve loop.

---

## Why "Cognitive"?

Most ARC solvers brute-force: BFS over 1000 grid states, hoping to stumble on the solution. CogniArc reasons **like a human**:

- **Explores with curiosity**, not exhaustive search
- **Doubts itself** and scraps bad plans
- **Finds joy** in symmetry and completion
- **Gets tired** and switches to intuition
- **Forgets** old states to force abstraction
- **Prefers simplicity** over complexity

> "To make an agent smarter, make it more human." — Not AGI, just good cognitive science.

---

## Architecture

```
cogniarc/
├── scientist_agent.py      # Main loop: Discover → Solve → PKM Memory
├── cognitive_player.py     # 6 human-like drives (novelty, doubt, joy, etc.)
├── cognitive_agent.py      # Drive integration + solver orchestration
├── domain_classifier.py    # Movement / Drawing / Transform / Interaction
├── physics_engine.py       # Deterministic state → next_state model
├── skill_tree.py           # RPG-style cumulative learning across levels
├── goal_inference.py       # Surprise, rarity, completion heuristics
├── stagnation_detector.py  # Anti-infinite-loop, detects local minima
├── transform_inference.py  # Grid transformation mapping + internal BFS
└── arc_agent.py            # Orchestrator v3 with real-step caching

games/
├── ls20_solver.py          # LS20-specific solver (Level 1 solved, 33 steps)
└── re86_explorer.py        # re86 exploration (drawing/painting paradigm)
```

### The 6 Cognitive Drives

| Drive | Mechanism | Why It Helps |
|-------|-----------|--------------|
| **Novelty** | High score for unseen states | Explores instead of optimizing blindly |
| **Simplicity** | Short plans > long plans | Finds elegance before brute-forcing |
| **Doubt** | If confident BUT stuck → scrap everything | Avoids GPT-5.5's mistake #3 |
| **Joy** | Attracted to symmetry, completion, order | Guesses "final" states intuitively |
| **Memory(7)** | Forgets old states (LRU) | Forces abstraction, not memorization |
| **Fatigue** | 50 steps of planning → switch to intuition | Like a human who "feels" the answer |

### The 9 Reasoning Modes

Inspired by the historical evolution of human thought (from myths to Descartes):

1. **Narrative-Symbolic** — creative hypotheses via analogies
2. **First-Principle** — find the dominant variable
3. **Linear Causality** — trace cause→effect chains (debugging)
4. **Conceptual Logic** (Socrates/Plato/Aristotle) — define before arguing
5. **Axiom + Observation** (Euclid/Archimedes) — build models, test on cases
6. **Structured Debate** (Scholastic) — thesis vs antithesis
7. **Simulation** (Galileo) — test one variable at a time
8. **Empirical Induction** (Bacon) — accumulate facts, detect bias
9. **Analysis-Synthesis** (Descartes) — divide→solve→verify pipeline

---

## Getting Started

### Prerequisites

```bash
pip install arc-agi
```

### Run the Scientist Agent

```python
from cogniarc.scientist_agent import ScientistAgent

agent = ScientistAgent('ls20')  # or 're86', 'ft09', 'sp80'
agent.run()
```

### Solve LS20 Level 1

```python
from cogniarc.ls20_solver import solve_level1

steps, score = solve_level1()
# → Level 1 solved in 33 steps
```

---

## Results (May 2026)

| Game | Paradigm | Status |
|------|-----------|--------|
| LS20 | Sprite maze + matching | Level 1 solved (33 steps, rotation 270°→0°) |
| LS20 Level 2 | — | Blocked by game design (player trapped after L1) |
| re86 | Drawing/painting | Mechanics fully decoded, exploration in progress |

---

## Key Innovation: Token Efficiency

Instead of sending 64×64 pixel grids (4096 tokens) to the LLM:

```
Grid → numpy preprocessor → structured properties → LLM
  4096 tokens                "pos=(34,10), rot=3, colors=[5,8,9]"  (15 tokens)
```

**800× token reduction.** With a planned Hailo-8 integration for semantic state classification (not naive YOLO on pixels), this could reach **2000× reduction**.

---

## Inspired By

- [ARC-AGI-3](https://three.arcprize.org) — François Chollet's benchmark for AI reasoning
- [Pokémon Player](https://github.com/nousresearch/hermes-agent) — 2-4 step observe→decide→act loop
- [GRAM](https://arxiv.org) — Recursive reasoning in fixed memory (O(1))
- [Opus 4.8 Swarm](https://anthropic.com) — 1000 agents in parallel, architect over coder

---

## Contributing

This is an active research project. PRs welcome for:

- New game solvers (ft09, sp80, tb00, ...)
- Hailo-8 integration (grid state classification)
- Multi-agent tournament (thesis vs antithesis)
- GRAM-style recursive reasoning loops

---

## License

MIT — Zed Art Vates, 2026.
