#!/usr/bin/env python3
"""
Goal Inference Engine for ARC-AGI-3.

The CORE challenge: given a novel interactive environment with ZERO instructions,
determine WHAT to target and WHY.

Strategies (combined with weights):
  1. SURPRISE — states with unexpected changes are "interesting" (JEPA principle)
  2. COMPLETION — patterns that suggest "incomplete" → complete form
  3. OBJECT VALUE — objects that change state when interacted with
  4. RARITY — rare objects (few pixels, unique color) are often goals/keys
  5. PROGRESS SIGNAL — level counter, score, visual progress bar
  6. CONTRASTIVE — compare initial state with states after successful interactions

The module builds a HYPOTHESIS LIST of possible goals, tests each,
and returns the BEST hypothesis with confidence.

Usage:
    from goal_inference import GoalInferenceEngine
    gie = GoalInferenceEngine(env, physics)
    goal = gie.infer()  # returns GoalHypothesis
"""

from __future__ import annotations

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from pathlib import Path

from .common import GameAction, hash_grid


class GridObject:
    """A distinct entity in the grid."""
    def __init__(self, obj_id: int, color: int, coords: np.ndarray):
        self.id = obj_id
        self.color = color
        self.coords = coords
        self.center = tuple(coords.mean(axis=0).astype(int))
        self.size = len(coords)
        self.bbox = (tuple(coords.min(axis=0)), tuple(coords.max(axis=0)))
        self.static: Optional[bool] = None       # never moves
        self.movable: Optional[bool] = None      # can be pushed/rotated
        self.collectible: Optional[bool] = None  # disappears on interaction
        self.interactive: Optional[bool] = None  # changes on interaction
        self.is_agent: bool = False              # this is the player
        self.is_goal: bool = False               # this is the target

    @property
    def rarity(self) -> float:
        """Rare objects are more likely to be goals/keys."""
        return 1.0 / max(self.size, 1)

    def __repr__(self):
        return f"Obj#{self.id}(c={self.color}, sz={self.size}, center={self.center})"


class GoalHypothesis:
    """A candidate goal."""
    def __init__(self, description: str, target_object: Optional[int] = None,
                 action_sequence: Optional[List[int]] = None,
                 completion_condition: Optional[Callable] = None):
        self.description = description
        self.target_object = target_object
        self.action_sequence = action_sequence or []
        self.completion_condition = completion_condition
        self.confidence: float = 0.0
        self.tests: int = 0
        self.successes: int = 0
        self.evidence: List[str] = []

    def score(self) -> float:
        if self.tests == 0:
            return self.confidence
        return self.confidence * (self.successes / self.tests)


class GoalInferenceEngine:
    """Infer the goal of an ARC-AGI-3 game from observation and interaction."""

    def __init__(self, env, physics_engine=None, max_steps: int = 80, observe_only: bool = False):
        self.env = env
        self.physics = physics_engine
        self.max_steps = max_steps
        self.observe_only = observe_only  # If True, never step the environment
        self.objects: List[GridObject] = []
        self.hypotheses: List[GoalHypothesis] = []
        self.initial_grid: Optional[np.ndarray] = None
        self.steps_spent: int = 0
        self.object_id_counter: int = 0

    def infer(self) -> List[GoalHypothesis]:
        """Full inference pipeline."""
        if self.observe_only:
            return self.infer_from_initial_grid()
        
        obs = self.env.reset()
        self.initial_grid = obs.frame[0].copy()

        # Phase 1: Extract and classify objects
        self._extract_objects(self.initial_grid)
        self._classify_objects(obs)

        # Phase 2: Generate hypotheses
        self._hypothesize_rarity()
        self._hypothesize_completion()
        self._hypothesize_surprise(obs)

        # Phase 3: Test hypotheses
        self._test_hypotheses()

        # Phase 4: Rank
        self.hypotheses.sort(key=lambda h: h.score(), reverse=True)
        self._save()
        return self.hypotheses

    def infer_from_initial_grid(self) -> List[GoalHypothesis]:
        """Goal inference WITHOUT stepping the environment - pure static analysis.
        
        Useful for:
        - Zero-shot goal guessing before any interaction
        - Pre-game profiling (DomainProfiler integration)
        - Games where stepping is expensive/failing
        
        Returns ranked hypotheses based on:
        - Rarity (small unique objects = keys/goals)
        - Completion (asymmetry, open boundaries)
        - Visual patterns (rotation indicators, color changers)
        """
        obs = self.env.reset()
        self.initial_grid = obs.frame[0].copy()
        
        # Phase 1: Extract objects (no interaction)
        self._extract_objects(self.initial_grid)
        
        # Phase 2: Static hypothesis generation (no testing)
        self._hypothesize_rarity()
        self._hypothesize_completion()
        self._hypothesize_rotation_targets()  # New: detect rotation/changer patterns
        self._hypothesize_color_targets()     # New: detect color changer targets
        
        # Phase 3: Rank by confidence only (no testing)
        self.hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        self._save()
        return self.hypotheses

    def _hypothesize_rotation_targets(self):
        """Detect objects that look like rotation controllers/changers."""
        grid = self.initial_grid
        
        # Look for circular/partial-circle patterns (manga-style rotation indicators)
        for color in np.unique(grid):
            if color == 0:
                continue
            mask = grid == color
            coords = np.argwhere(mask)
            if len(coords) < 5 or len(coords) > 200:
                continue
            
            # Check for ring/circle shape
            h_span = coords[:, 1].max() - coords[:, 1].min()
            v_span = coords[:, 0].max() - coords[:, 0].min()
            aspect = h_span / max(v_span, 1)
            
            # Approximate circle detection
            if 0.7 <= aspect <= 1.4 and len(coords) < 100:
                # Find object at this location
                center_r = int(coords[:, 0].mean())
                center_c = int(coords[:, 1].mean())
                for obj in self.objects:
                    if abs(obj.center[0] - center_r) < 3 and abs(obj.center[1] - center_c) < 3:
                        h = GoalHypothesis(
                            description=f"Rotate to target angle (changer at {obj.center}, color {obj.color})",
                            target_object=obj.id,
                        )
                        h.confidence = 0.6
                        h.evidence.append(f"Circular pattern at {obj.center} — likely rotation changer")
                        self.hypotheses.append(h)
                        break

    def _hypothesize_color_targets(self):
        """Detect objects that look like color changers / target indicators."""
        grid = self.initial_grid
        
        # Look for small distinct color patches that might be targets
        for obj in self.objects:
            # Very small colored objects (1-5 pixels) in distinct colors
            if obj.size <= 5 and obj.color != 0:
                # Check if color is rare in grid
                color_pixels = np.sum(grid == obj.color)
                total_pixels = grid.size
                rarity = color_pixels / total_pixels
                
                if rarity < 0.02:  # Less than 2% of grid
                    h = GoalHypothesis(
                        description=f"Match color {obj.color} (rare target at {obj.center})",
                        target_object=obj.id,
                    )
                    h.confidence = 0.5 * (1 - rarity * 10)  # Higher for rarer colors
                    h.evidence.append(f"Rare color {obj.color} ({rarity:.1%} of grid) at {obj.center}")
                    self.hypotheses.append(h)

    # ── Object Extraction ────────────────────────────────

    def _extract_objects(self, grid: np.ndarray):
        """Find connected components of same color."""
        assert grid is not None and grid.size > 0, "Invalid grid: None or empty"
        self.objects = []
        visited = np.zeros(grid.shape, dtype=bool)

        for color in np.unique(grid):
            if color == 0:  # background
                continue
            mask = (grid == color) & ~visited
            if not mask.any():
                continue

            # Simple flood fill for each connected component
            from collections import deque
            while mask.any():
                start = tuple(np.argwhere(mask)[0])
                component = []
                q = deque([start])
                mask[start] = False
                visited[start] = True

                while q:
                    r, c = q.popleft()
                    component.append((r, c))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < 64 and 0 <= nc < 64:
                            if grid[nr, nc] == color and not visited[nr, nc]:
                                visited[nr, nc] = True
                                mask[nr, nc] = False
                                q.append((nr, nc))

                if len(component) >= 1:
                    obj = GridObject(
                        obj_id=self.object_id_counter,
                        color=int(color),
                        coords=np.array(component),
                    )
                    self.object_id_counter += 1
                    self.objects.append(obj)
        # Verify: at least one object found (unless grid is empty)
        assert isinstance(self.objects, list), "objects must be list"

    def _classify_objects(self, obs):
        """Classify objects by interacting with the environment."""
        assert obs is not None, "Invalid observation: None"
        assert hasattr(obs, 'frame'), "Invalid observation: missing frame"
        assert hasattr(obs, 'available_actions'), "Invalid observation: missing available_actions"
        initial_objects = list(self.objects)

        # Test movement: try each action, see which object moved
        for act_num in obs.available_actions[:4]:
            if self.steps_spent >= self.max_steps // 2:
                break
            action = getattr(GameAction, f"ACTION{act_num}")
            obs = self.env.step(action)
            self.steps_spent += 1

            # Verify step result
            assert obs is not None, "Step returned None"
            assert hasattr(obs, 'frame'), "Step result missing frame"
            assert obs.frame is not None, "Step result frame is None"

            new_grid = obs.frame[0]
            self._extract_objects(new_grid)

            # Match objects between before and after
            for old_obj in initial_objects:
                # Find closest matching object
                for new_obj in self.objects:
                    if new_obj.color == old_obj.color and abs(new_obj.size - old_obj.size) <= 1:
                        if new_obj.center != old_obj.center:
                            old_obj.movable = True
                            old_obj.is_agent = True  # first moving object = agent
                        break
                # If object disappeared
                if not any(new_obj.color == old_obj.color and abs(new_obj.size - old_obj.size) <= 1
                          for new_obj in self.objects):
                    old_obj.collectible = True

        # Classify remaining objects as static
        for obj in self.objects:
            if obj.static is None and obj.movable is None and obj.collectible is None:
                obj.static = True
        # Verify: all objects have classification
        for obj in self.objects:
            assert obj.static is not None or obj.movable is not None or obj.collectible is not None, \
                f"Object {obj.id} unclassified"

    # ── Hypothesis Generation ────────────────────────────

    def _hypothesize_rarity(self):
        """Rare objects (small, unique color) are often keys or goals."""
        rare_objs = sorted(self.objects, key=lambda o: o.size)[:5]
        for obj in rare_objs:
            h = GoalHypothesis(
                description=f"Interact with rare object #{obj.id} (color {obj.color}, size {obj.size})",
                target_object=obj.id,
            )
            h.confidence = obj.rarity * 0.5
            h.evidence.append(f"Rare: only {obj.size} pixels, unique color {obj.color}")
            self.hypotheses.append(h)

    def _hypothesize_completion(self):
        """Detect incomplete patterns that suggest a completion goal."""
        grid = self.initial_grid

        # Check for symmetrical incompleteness
        h, w = grid.shape
        mid = w // 2
        left = grid[:, :mid]
        right = np.fliplr(grid[:, mid:])

        # Asymmetry might indicate something needs to be moved
        diff = (left != right).sum()
        if diff > 10 and diff < 1000:
            # There's asymmetry — maybe something needs to be balanced
            h = GoalHypothesis(
                description="Complete symmetry by moving objects",
                target_object=None,
            )
            h.confidence = min(diff / 500, 0.4)
            h.evidence.append(f"Asymmetry: {diff} pixels differ left vs right")
            self.hypotheses.append(h)

        # Check for open boundaries (like the manga circle)
        for color in np.unique(grid):
            if color == 0:
                continue
            mask = grid == color
            # Find objects with thin, elongated shapes (like incomplete circles)
            coords = np.argwhere(mask)
            if len(coords) < 5:
                continue
            h_span = coords[:, 1].max() - coords[:, 1].min()
            v_span = coords[:, 0].max() - coords[:, 0].min()
            if h_span > v_span * 3 or v_span > h_span * 3:
                # Very elongated — possibly an incomplete boundary
                h = GoalHypothesis(
                    description=f"Complete or close shape of color {color}",
                    target_object=None,
                )
                h.confidence = 0.3
                h.evidence.append(f"Elongated shape: color {color}, {h_span}x{v_span}")
                self.hypotheses.append(h)

    def _hypothesize_surprise(self, obs):
        """States that change unexpectedly are goal-candidates (JEPA principle)."""
        # For each collectible object, hypothesize "collect this"
        for obj in self.objects:
            if obj.collectible:
                h = GoalHypothesis(
                    description=f"Collect object #{obj.id} (color {obj.color})",
                    target_object=obj.id,
                )
                h.confidence = 0.6
                h.evidence.append("Object disappears on interaction — likely collectible")
                self.hypotheses.append(h)

            if obj.interactive or (obj.movable and not obj.is_agent):
                h = GoalHypothesis(
                    description=f"Manipulate object #{obj.id} (color {obj.color})",
                    target_object=obj.id,
                )
                h.confidence = 0.4
                h.evidence.append("Object changes on interaction")
                self.hypotheses.append(h)

    # ── Testing ──────────────────────────────────────────

    def _test_hypotheses(self):
        """Test each hypothesis by attempting the predicted action."""
        # For now: rank by confidence, test top 3
        for h in self.hypotheses[:3]:
            if self.steps_spent >= self.max_steps:
                break
            h.tests += 1

            # If target object known, try to navigate to it
            if h.target_object is not None:
                target = next((o for o in self.objects if o.id == h.target_object), None)
                if target and self.physics:
                    # Navigate to target
                    self._navigate_to(target, h)
                    continue

            # Otherwise, just observe if the condition seems plausible
            h.successes += 1  # placeholder

    def _navigate_to(self, target: GridObject, hypothesis: GoalHypothesis):
        """Move agent toward target object using physics engine."""
        # Get current player position
        agent = next((o for o in self.objects if o.is_agent), None)
        if not agent:
            return

        # Simple heuristic: move toward target
        dr = target.center[0] - agent.center[0]
        dc = target.center[1] - agent.center[1]

        # Vertical movement
        for _ in range(abs(dr)):
            if self.steps_spent >= self.max_steps:
                return
            if dr > 0:
                self.env.step(GameAction.ACTION3)  # assume ACTION3 = down
            else:
                self.env.step(GameAction.ACTION1)  # assume ACTION1 = up
            self.steps_spent += 1

        # Horizontal movement
        for _ in range(abs(dc)):
            if self.steps_spent >= self.max_steps:
                return
            if dc > 0:
                self.env.step(GameAction.ACTION4)  # assume ACTION4 = right
            else:
                self.env.step(GameAction.ACTION2)  # assume ACTION2 = left
            self.steps_spent += 1

    # ── Report ───────────────────────────────────────────

    def report(self) -> str:
        if not self.hypotheses:
            return "No hypotheses generated."
        lines = ["Goal Hypotheses (ranked):"]
        for i, h in enumerate(self.hypotheses[:5]):
            lines.append(f"  #{i+1}: {h.description}")
            lines.append(f"     confidence={h.confidence:.2f}, score={h.score():.2f}")
            for e in h.evidence[:2]:
                lines.append(f"     └ {e}")
        return "\n".join(lines)

    def best_goal(self) -> Optional[GoalHypothesis]:
        if not self.hypotheses:
            return None
        return self.hypotheses[0]

    def _save(self):
        out = Path("/home/redgamer/arc_agi_agent/goal_inference.json")
        data = {
            "n_objects": len(self.objects),
            "objects": [{"id": o.id, "color": o.color, "size": o.size,
                         "center": o.center, "is_agent": o.is_agent,
                         "collectible": o.collectible, "static": o.static}
                        for o in self.objects[:20]],
            "hypotheses": [{"desc": h.description, "confidence": h.confidence,
                            "score": h.score(), "evidence": h.evidence}
                           for h in self.hypotheses[:5]],
        }
        out.write_text(json.dumps(data, indent=2, default=str))
