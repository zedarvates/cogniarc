# Plan: Game-Type Classifier + Painting Phase Machine

**Goal:** Detect game type from grid-change patterns after scout, route to appropriate phase machine.

**Architecture:** 
- `cogniarc/domain_classifier.py` — pure function that takes scout observations and returns game type
- Modified `run()` in `scientist_agent.py` — uses classifier output to set domain_type
- New `_solve_painting_level()` strategy for sc25-style games

---

### Task 1: Game-Type Classifier

**File:** Create `cogniarc/domain_classifier.py`
**Test:** `tests/test_domain_classifier.py`

Pure function (no arc_agi dependency, testable with synthetic grids):

```python
from typing import Dict, List, Optional, Tuple
import numpy as np

GameType = str  # "navigation" | "painting" | "puzzle" | "unknown"

def classify_game_type(scout_results: dict, actions_tested: List[int],
                       grid_changes: List[Tuple[int, int]]) -> GameType:
    """Classify game type from scout-phase observations.
    
    Args:
        scout_results: {action: {moved, grid_changed, prop_changes}} from PKM
        actions_tested: list of action numbers tested
        grid_changes: [(n_pixels_changed, n_colors_changed)] per action
    
    Returns:
        One of "navigation", "painting", "puzzle", "unknown"
    """
    if not grid_changes:
        return "unknown"
    
    n_movement = sum(1 for r in scout_results.values() if r.get('moved'))
    max_diff = max((c[0] for c in grid_changes), default=0)
    avg_diff = sum(c[0] for c in grid_changes) / len(grid_changes)
    color_diversity = max((c[1] for c in grid_changes), default=0)
    
    # Navigation: a moving region (few pixels change, same color moves)
    if n_movement >= 2 and max_diff <= 20 and color_diversity <= 3:
        return "navigation"
    
    # Painting: many pixels change color in clusters
    if avg_diff >= 8 and color_diversity >= 3:
        return "painting"
    
    # Puzzle: few pixels change, targeted
    if max_diff <= 10 and color_diversity <= 2:
        return "puzzle"
    
    return "unknown"
```

**Tests:**
- Navigation scenario: grid with single moving pixel, 5-10 diff
- Painting scenario: grid with 20+ pixel change area
- Puzzle scenario: tiny 2-3 pixel changes
- Edge case: no changes at all → "unknown"

---

### Task 2: Painting Phase Machine

**File:** Create `cogniarc/painting_strategy.py`

A strategy for sc25-style games where actions paint/change pixel colors:

```python
class PaintingStrategy:
    """Solve strategy for painting/interaction games (sc25, etc.)."""
    
    def __init__(self, agent):
        self.agent = agent
        self._paint_actions_tried = set()
        self._click_actions_tried = set()
    
    def solve_level(self, level_num=None) -> bool:
        """Try painting actions in different patterns until level completes."""
        prev_lvl = self.agent.obs.levels_completed
        
        # Phase 1: Discover which actions affect which colors
        print("  🎨 Paint strategy: discovering action effects...")
        effects = self._discover_action_effects()
        
        # Phase 2: Apply actions systematically
        print(f"  🎨 Effects: {effects}")
        
        for _ in range(50):
            if self.agent.obs.levels_completed > prev_lvl:
                return True
            
            if not self._apply_best_action(effects):
                break
        
        return self.agent.obs.levels_completed > prev_lvl
    
    def _discover_action_effects(self) -> dict:
        """Discover what each action does to grid colors."""
        effects = {}
        for action in [2, 3, 4]:
            if action in self.agent.obs.available_actions:
                grid_before = self.agent.obs.frame[0].copy()
                self.agent.step(action)
                grid_after = self.agent.obs.frame[0]
                changed = np.argwhere(grid_before != grid_after)
                color_pairs = set()
                for r, c in changed:
                    color_pairs.add((int(grid_before[r,c]), int(grid_after[r,c])))
                effects[action] = {
                    'n_changed': len(changed),
                    'color_pairs': color_pairs,
                }
        return effects
    
    def _apply_best_action(self, effects: dict) -> bool:
        """Apply the most impactful painting action."""
        if not effects:
            return False
        best = max(effects, key=lambda a: effects[a]['n_changed'])
        self.agent.step(best)
        return True
```

---

### Task 3: Wire Into run() + solve_level()

**File:** Modify `cogniarc/scientist_agent.py`

In `run()`, after ObjectTracker scout:

```python
from cogniarc.domain_classifier import classify_game_type

# After scout phase
scout_results = self.pkm.get('discovery', 'scout_results', {})
grid_changes = []  # From PKM or computed
game_type = classify_game_type(scout_results, list(scout_results.keys()), grid_changes)
self.state.domain_type = game_type
print(f"  🎮 Game type: {game_type}")
```

In `solve_level()`, route based on game type:

```python
if self.state.domain_type == "painting":
    from cogniarc.painting_strategy import PaintingStrategy
    strategy = PaintingStrategy(self)
    return strategy.solve_level(level_num)
# else: existing navigation phase machine
```
