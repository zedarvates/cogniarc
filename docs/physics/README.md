# World Model Tool 🌍

**Approximate physical reasoning engine for small LLMs.**

Simule, classifie et raisonne sur la physique du monde réel — avec des modèles microscopiques (580 paramètres) au lieu de LLMs de 100M+ de paramètres.

## Architecture

```
world-model-tool/
├── simulator/           Moteur physique (forces, matériaux, fluides)
│   ├── physics.py       v2 — formes, collisions SAT, humidité, pression
│   └── physics_v3.py    v3 — causal, composés, énergie, agents, thermique
├── models/              Small world models (MLP ~10K params)
│   └── trainer.py       Entraînement supervisé + normalisation
├── tools/               Moteurs experts pour le raisonnement LLM
│   ├── advanced_physics.py     8 engines: élasticité, résonance, chaos...
│   ├── discrete_classifier.py  Classification 8 états de mouvement
│   ├── kinematic_engine.py     Mobilité, workspace, transmissions
│   ├── mass_gravity.py         Masse/inertie/poids/orbites
│   ├── momentum_inertia.py     Élan/collisions/inertie rotationnelle
│   ├── relation_engine.py      Tensions/contraintes/molécules
│   ├── scene_graph.py          Graphe symbolique ASCII + DOT
│   ├── spatial_reasoning.py    Perception/occlusion/pathfinding
│   ├── spatial_zoning.py       Inside/outside/near/far/between
│   ├── torque_experts.py       Couples + 10 micro-NN experts
│   └── unified_dashboard.py    Dashboard V3 complet
├── tests/               Smoke tests (8/8 passés)
│   └── test_smoke.py
├── visualizer/          Visualisation Three.js 3D
│   ├── index.html          Vue 3D temps réel
│   └── theory-lab.html     Labo de théories interactif
└── constants.py         Constantes physiques partagées
```

## Utilisation rapide

```bash
# Simuler un scénario
PYTHONPATH=. python3 tools/unified_dashboard.py ramp

# Générer une théorie
PYTHONPATH=. python3 tools/llm_tool.py

# Lancer les tests
PYTHONPATH=. python3 tests/test_smoke.py

# Visualisation dans le navigateur
# Ouvrir visualizer/theory-lab.html
```

## Scénarios disponibles

| Scenario | Description | Variables |
|----------|-------------|-----------|
| `ramp` | Balle métal sur planche bois 45° | Hauteur, matériaux |
| `viscous` | Boules dans huile/eau/miel | Viscosité, densité |
| `humidity` | Rampe mouillée (90% humidité) | Humidité 0-1 |
| `depression` | Mini-tornade (dépression) | Pression, vent |
| `magnetic` | Boules métal + aimant | Force magnétique |
| `vehicle` | Crash voiture+passager+bagage | Masses, inertie |
| `jenga` | Tour de blocs + cascade causale | Support, gravité |

## Moteurs experts (20 domaines)

Activation automatique par mots-clés. Seuls les experts pertinents sont chargés.

```
Query: "une bille percute une autre bille"
  → [momentum] Expert Élan/Collisions activé (58 params)

Query: "lâche une balle du 5e étage"
  → [gravity] Expert Gravitation activé

Query: "de l'huile à 200°C touche de l'eau froide"
  → [thermal] + [fluid] activés (116 params)
```

## État du projet

✅ 20 domaines physiques couverts  
✅ 12 modules, ~6,500 lignes, zéro dépendance externe  
✅ 8/8 tests passés  
✅ Architecture micro-NN (580 params totaux, ~58 par expert)  
✅ Licence MIT

## Licence

MIT — voir [LICENSE](LICENSE).
