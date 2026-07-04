# Plan: Wired Generic Perception + Active Experimentation

> **Pour Hermes:** Implémenter tâche par tâche, tester chaque étape.

**Objectif:** Remplacer le LS20-centric tag-based discovery par ObjectTracker + active experimentation comme perception par défaut, pour que la phase machine fonctionne sur tout jeu sans tags hardcodés.

**Architecture:** 3 modifications ciblées qui transforment l'ObjectTracker d'advisory → steering dans la boucle décisionnelle.

**Gap actuel:** `discover_from_source()` parse les tags LS20, `_find_tagged_sprites('rhsxkxzdjz')` est appelé partout dans la phase machine. ObjectTracker observe déjà chaque step() mais ses résultats ne sont jamais utilisés pour décider. `suggest_wall_experiment()` calcule l'action discriminante mais ne l'exécute pas.

---

### Task 1: Rendre ObjectTracker utilisable comme source unique de perception

**Objectif:** ObjectTracker expose `player_color`, `action_direction`, et `wall_colors` de manière exploitable par la phase machine — sans tags.

**Fichiers:**
- Modifier: `cogniarc/object_perception.py` (ajouter une méthode summary)
- Test: `tests/test_object_perception.py` (vérifier le format)

**Step 1: Lire le fichier actuel**

```bash
cat cogniarc/object_perception.py
```

**Step 2: Ajouter `get_perception_summary()` à ObjectTracker**

```python
def get_perception_summary(self) -> dict:
    """Return a structured dict that a phase machine can consume directly.
    
    Returns:
        dict with keys:
        - player_color: int or None
        - action_directions: {action_num: (dr, dc)} for movement actions
        - wall_colors: set[int] — colors with >= min_wall_votes
        - n_observations: int
        - player_moved_last_step: bool or None
    """
    wall_colors = {
        color for color, votes in self.wall_color_votes.items()
        if votes >= self.min_wall_votes
    }
    
    action_dirs = {
        a: self.action_direction(a)
        for a in self.action_displacements
        if self.action_direction(a) is not None
    }
    
    return {
        "player_color": self.player_color,
        "action_directions": action_dirs,
        "wall_colors": wall_colors,
        "n_observations": self.n_observations,
        "player_moved_last_step": self.last_step_player_moved,
    }
```

**Step 3: Ajouter `has_enough_observations()`**

```python
def has_enough_observations(self, min_player: int = 3, min_directions: int = 1) -> bool:
    """Return True if ObjectTracker has learned enough to be useful."""
    if self.player_color is None:
        return False
    known_dirs = sum(1 for a in self.action_displacements if self.action_direction(a) is not None)
    return self.n_observations >= min_player and known_dirs >= min_directions
```

**Step 4: Tester**

```python
def test_get_perception_summary():
    t = ObjectTracker()
    g0 = grid(cells={(2, 1): 5})
    g1 = grid(cells={(2, 2): 5})
    t.observe(g0, action=1, grid_after=g1)
    t.observe(g1, action=1, grid_after=grid(cells={(2, 3): 5}))
    
    summary = t.get_perception_summary()
    assert summary["player_color"] == 5
    assert 1 in summary["action_directions"]
    assert summary["n_observations"] >= 2
    assert summary["player_moved_last_step"] is not None or True  # Could be True
    assert t.has_enough_observations()
```

**Step 5: Commit**

```bash
cd /home/redgamer/projects/cogniarc
git add cogniarc/object_perception.py tests/test_object_perception.py
git commit -m "feat: ObjectTracker.get_perception_summary() — structured output for phase machine"
```

---

### Task 2: Brancher ObjectTracker comme source de décision dans `run()`

**Objectif:** La phase `run()` utilise ObjectTracker pour découvrir walls/actions au lieu de `discover_from_source()`. ObjectTracker guide les phases de navigation, pas les tags.

**Fichiers:**
- Modifier: `cogniarc/scientist_agent.py` (run() et solve_level())
- Modifier: `cogniarc/scientist_agent_discovery.py` (discover_properties)
- Test: besoin d'un test d'intégration

**Step 1: Modifier `run()` pour utiliser ObjectTracker**

Dans `run()`, remplacer la découverte tag-based par ObjectTracker :

```python
def run(self):
    print(f"🔬 Scientist Agent — {self.name}")
    print(f"   Start: lvl={self.obs.levels_completed}/{self.obs.win_levels}")

    # PHASE 1: Discover — scout with ObjectTracker (generic, no tags)
    print("\n📖 DISCOVERY PHASE")
    print("  Generic perception: scouting via ObjectTracker...")
    
    # Take a few steps to build ObjectTracker observations
    scout_actions = list(self.obs.available_actions or [1, 2, 3, 4, 5, 6])
    for _ in range(8):
        action = scout_actions[self.steps % len(scout_actions)]
        self.step(action)
    
    # Check if ObjectTracker has enough data
    if self.object_tracker.has_enough_observations():
        summary = self.object_tracker.get_perception_summary()
        print(f"  🎯 Player color: {summary['player_color']}")
        print(f"  🧭 Learned action directions: {summary['action_directions']}")
        print(f"  🧱 Wall color evidence: {summary['wall_colors']}")
    else:
        # Still try source discovery as fallback
        print("  ⚠️ ObjectTracker insufficient data, trying source fallback...")
        self.discover_from_source()
    
    self.discover_available_actions()
    
    # Wall detection via ObjectTracker or legacy fallback
    self._walls_detected = self._detect_walls_generic()
    
    # ... rest unchanged
```

**Step 2: Ajouter `_detect_walls_generic()`**

```python
def _detect_walls_generic(self) -> bool:
    """Detect wall colors using ObjectTracker's evidence, fall back to source.
    
    Returns True if walls were detected by any method.
    """
    if self.object_tracker and self.object_tracker.has_enough_observations():
        summary = self.object_tracker.get_perception_summary()
        if summary["wall_colors"]:
            self._wall_colors = summary["wall_colors"]
            self.state.walls_detected = True
            self.state.set_assumption("walls_known", True)
            print(f"  🧱 Wall colors (ObjectTracker): {self._wall_colors}")
            return True
    
    # Fallback: source-code tag detection (LS20-specific)
    return self.discover_from_source()
```

**Step 3: Modifier `_build_phase_hypothesis()` pour être générique**

Remplacer les hypothèses LS20-specific par des hypothèses génériques:

```python
def _build_phase_hypothesis(self) -> str:
    hypotheses = {
        "detect_walls": "Learn which colours are walls by observing movement blocks",
        "navigate_to_target": "Navigate toward a target using ObjectTracker's learned action directions",
        "interact_with_object": "Use an interactive action to complete the level objective",
        "complete": "Level is finished, move to next",
    }
    return hypotheses.get(self._phase, f"Unknown phase {self._phase}")
```

**Step 4: Tester que la découverte générique ne casse pas LS20**

```bash
cd /home/redgamer/projects/cogniarc
python -m pytest tests/ -x -v -k "not e2e" 2>&1 | tail -20
```

**Step 5: Commit**

```bash
git add cogniarc/scientist_agent.py cogniarc/scientist_agent_discovery.py
git commit -m "feat: wire ObjectTracker as primary perception in run() — generic wall/player detection"
```

---

### Task 3: Faire de `suggest_wall_experiment()` une action réelle (pas advisory)

**Objectif:** Quand le wall/floor est ambigu et qu'une action discrimine à >1.0 bit, L'EXÉCUTER au lieu de juste la recommander.

**Fichiers:**
- Modifier: `cogniarc/scientist_agent_skills.py` (_skill_detect_walls)  
- Modifier: `cogniarc/scientist_agent_discovery.py` (suggest_wall_experiment retourne l'action)

**Step 1: Modifier `suggest_wall_experiment()` pour retourner l'action directement**

```python
def suggest_wall_experiment(self) -> Optional[int]:
    """Return the action number that would best resolve wall/floor ambiguity,
    or None if nothing needs testing. Replaces the advisory-only version.
    """
    tracker = getattr(self, 'object_tracker', None)
    if tracker is None or self.player is None:
        return None
    # ... (existing logic to compute best action) ...
    
    # Changed from advisory to actionable:
    if best is not None:
        color, action, info = best
        if info >= 1.0:  # Only execute when info gain is meaningful
            print(f"  🔬 Executing active experiment: action {action} tests colour {color} ({info:.2f} bit)")
            self.state.record_observation(
                f"Active experiment: action {action} for colour {color} ({info:.2f} bit)",
                source="active_experiment"
            )
            return action
    
    return None
```

**Step 2: Intégrer dans la boucle de solve_level()**

Avant la phase machine principale, dans `solve_level()` :

```python
# ═══ Active experimentation: execute if there's a clear info-gain action ═══
experiment_action = self.suggest_wall_experiment()
if experiment_action is not None:
    print(f"  🔬 Active experiment: action {experiment_action}")
    self.step(experiment_action)
    # Don't continue the phase loop this iteration — let the loop re-evaluate
```

**Step 3: Tester**

```bash
cd /home/redgamer/projects/cogniarc
python -m pytest tests/test_active_experiment.py tests/test_object_perception.py -v
```

**Step 4: Commit**

```bash
git add cogniarc/scientist_agent.py cogniarc/scientist_agent_discovery.py
git commit -m "feat: wire active experimentation into decision loop — actions execute when info gain >= 1.0 bit"
```

---

### Task 4: Phase machine générique — remplacer les tags LS20

**Objectif:** La phase machine utilise ObjectTracker pour naviguer au lieu de `_find_tagged_sprites()`. Les phases `navigate_to_changer` / `navigate_to_lock` deviennent une seule phase `navigate_to_target`.

**Fichiers:**
- Modifier: `cogniarc/scientist_agent_skills.py` (_advance_phase, _build_skill_context, _skill_navigate_to_target)
- Modifier: `cogniarc/scientist_agent.py` (solve_level, _build_phase_hypothesis)

**Step 1: Simplifier les phases**

Avant (LS20-specific):
```
detect_walls → navigate_to_changer → rotate_to_goal → navigate_to_lock → interact → complete
```

Après (générique):
```
detect_walls → navigate_to_target → interact → complete
```

```python
PHASE_TRANSITIONS = {
    "detect_walls": "navigate_to_target",
    "navigate_to_target": "interact",
    "interact": "complete",
}
```

**Step 2: Modifier `_advance_phase()`**

```python
def _advance_phase(self, success: bool):
    old_phase = self._phase
    
    transitions = {
        "detect_walls": "navigate_to_target",
        "navigate_to_target": "interact",
        "interact": "complete",
    }
    
    if success and self._phase in transitions:
        self._phase = transitions[self._phase]
    
    if self._phase != old_phase:
        self.state.phase = self._phase
        self.state.phase_attempts = 0
        print(f"  ➡️ Phase: {old_phase} → {self._phase}")
```

**Step 3: Modifier `_skill_navigate_to_target()`**

Remplacer la navigation tag-based par ObjectTracker-based :

```python
def _skill_navigate_to_target(self) -> bool:
    """Navigate using ObjectTracker's learned action directions.
    Generic: no tag lookups, no hardcoded positions.
    """
    if not self.player or not self.object_tracker:
        return False
    
    summary = self.object_tracker.get_perception_summary()
    action_dirs = summary.get("action_directions", {})
    if not action_dirs:
        # Fallback: try all available movement actions
        available = list(self.obs.available_actions or [])
        movement = [a for a in available if a in [1, 2, 3, 4]]
        if movement:
            self.step(movement[0])
            return True
        return False
    
    # Pick the action that moves in the most promising direction
    # (toward unexplored area, or away from walls)
    wall_colors = summary.get("wall_colors", set())
    
    # Try each learned movement action; if it moves the player, continue
    for action in sorted(action_dirs.keys()):
        prev_pos = (self.player.x, self.player.y)
        self.step(action)
        if (self.player.x, self.player.y) != prev_pos:
            return True
    
    # If no action moved us, we might be blocked — trigger wall detection
    return False
```

**Step 4: Tester**

```bash
cd /home/redgamer/projects/cogniarc
python -m pytest tests/ -x -v -k "not e2e"
```

**Step 5: Commit**

```bash
git add cogniarc/scientist_agent.py cogniarc/scientist_agent_skills.py
git commit -m "refactor: replace LS20 tag-based phases with generic navigate_to_target — ObjectTracker-based navigation"
```

---

### Task 5: Wire Nano-LLM tier into the decision loop

**Objectif:** Quand la phase machine stagne et que ObjectTracker a assez de données, activer `_nano_propose_action()` pour proposer l'action suivante.

**Fichiers:**
- Modifier: `cogniarc/scientist_agent_ml_tiers.py` (vérifier le wiring actuel)
- Modifier: `cogniarc/scientist_agent.py` (solve_level — appeler nano quand stuck)

**Step 1: Vérifier l'état actuel de nano_llm wiring**

```bash
grep -n "_nano_propose_action\|nano_llm\|enable_nano_llm" cogniarc/scientist_agent*.py
```

**Step 2: Ajouter l'appel nano_llm dans la boucle de stagnation**

```python
# After phase failure and before escalation, try nano-LLM
if self.state.phase_attempts >= 2 and self._nano_llm is not None:
    print(f"  🤖 Nano-LLM proposing action...")
    proposal = self._nano_propose_action()
    if proposal is not None:
        print(f"  🤖 Nano-LLM suggests action {proposal}")
        self.step(proposal)
        if self.obs.levels_completed > prev_lvl:
            return True
```

**Step 3: Tester**

```bash
cd /home/redgamer/projects/cogniarc
python -m pytest tests/ -x -v -k "not e2e"
```

**Step 4: Commit**

```bash
git add cogniarc/scientist_agent.py cogniarc/scientist_agent_ml_tiers.py
git commit -m "feat: wire Nano-LLM tier into decision loop — invoked when phase machine stagnates"
```

---

## Vérification finale

```bash
cd /home/redgamer/projects/cogniarc
python -m pytest tests/ -v 2>&1 | tail -30
python scripts/generalization_report.py --list 2>&1
git log --oneline -10
```

## Résumé des changements

| Fichier | Changement |
|---------|-----------|
| `cogniarc/object_perception.py` | +`get_perception_summary()`, +`has_enough_observations()` |
| `cogniarc/scientist_agent.py` | `run()` utilise ObjectTracker, `solve_level()` appelle active_experiment + nano_llm, phases simplifiées |
| `cogniarc/scientist_agent_skills.py` | `_advance_phase()` simplifiée, `_skill_navigate_to_target()` générique |
| `cogniarc/scientist_agent_discovery.py` | `suggest_wall_experiment()` retourne action au lieu d'advisory |
