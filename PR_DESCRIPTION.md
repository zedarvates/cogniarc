# CogniARC — with physics world model integration

This PR adds the **World Model Physics Engine** as a sub-module of CogniARC's
existing WorldModelTool infrastructure.

## What's added

```
cogniarc/world_model/physics/        ← New physics engine (20 domains)
├── __init__.py
├── simulator/
│   ├── physics.py                   v2 — forces, collisions SAT, fluides
│   └── physics_v3.py                v3 — causal, composés, énergie, agents
├── models/
│   └── trainer.py                   Small world model training
├── tools/                           ★ 10 reasoning tools
│   ├── advanced_physics.py          Élasticité, résonance, chaos, ondes...
│   ├── discrete_classifier.py       8 états de mouvement
│   ├── kinematic_engine.py          Mobilité, workspace, transmissions
│   ├── mass_gravity.py              Masse, poids, orbites
│   ├── momentum_inertia.py          Élan, collisions, inertie rotationnelle
│   ├── relation_engine.py           Tensions, contraintes, molécules
│   ├── scene_graph.py               Graphe ASCII + DOT
│   ├── spatial_reasoning.py         Perception/occlusion/pathfinding
│   ├── spatial_zoning.py            Inside/outside/near/far
│   └── torque_experts.py            Couples + 10 micro-NN experts
├── tests/
│   └── test_smoke.py                8 tests (8/8 pass)
├── visualizer/                      Three.js interactive views
├── constants.py
├── requirements.txt
└── README.md
```

## Architecture

CogniARC's existing `world_model.py` uses V-JEPA for representation learning.
The new `physics/` module adds **deterministic physical simulation** as a
complement — the agent can query both the learned predictor (V-JEPA) and
the physics engine (Newtonian) depending on the task.

Micro-NN experts (580 params total, ~58 per domain) are activated on-demand
by keyword matching — no GPU, no large model needed for common queries.

## Key features
- 20 physical domains: forces, collisions, fluids, thermal, chaos, optics...
- Spatial reasoning: inside/outside, near/far, occlusion, pathfinding
- 8-state discrete movement classifier for small LLMs
- Dependency-free (only numpy)
- 8/8 tests passing

## Related
- Closes issue: World model tool should be part of CogniARC
- Extends WorldModelTool with physical simulation capability
