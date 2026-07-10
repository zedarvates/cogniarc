"""Skill-execution mixin for ScientistAgent.

Groups the methods that execute individual skills (interact, detect-walls,
navigate-to-target, rotate-to-goal) and advance the phase state machine.
Extracted verbatim from scientist_agent.py (no logic changes) to shrink that
file's God-object footprint.

Calls into `self._init_pathfinder()` (DiscoveryMixin) — note the single
leading underscore: the original `__init_pathfinder()` name was renamed
during the mixin split to avoid Python's per-class name mangling on
double-underscore attributes, which would otherwise break this cross-mixin
call once ScientistAgent composes DiscoveryMixin + SkillsMixin + MLTiersMixin.

Mixed into ScientistAgent; relies on attributes/methods set up in its
__init__ and in the other mixins (self.player, self.obs, self.drives,
self.pathfinder_nn, self.action_predictor, self.world_model, self.state,
self._phase, self._walls_detected, self._detected_wall_colors).
"""
from typing import Any, Dict, Optional
import numpy as np


class SkillsMixin:
    """Skill execution + phase-advance helpers."""

    def _build_skill_context(self) -> Dict[str, Any]:
        """Build context dict for skill selection."""
        available_actions = list(self.obs.available_actions or [])
        is_rotation_game = 6 in available_actions and not any(a in available_actions for a in [1, 2, 3, 4])

        # Use ObjectTracker for generic context when available
        ot = getattr(self, 'object_tracker', None)
        ot_ready = ot is not None and ot.has_enough_observations()
        ot_summary = ot.get_perception_summary() if ot_ready else {}

        return {
            "has_player": self.player is not None,
            "has_pathfinder": self._init_pathfinder() is not None,
            "has_changer": len(self._find_tagged_sprites('rhsxkxzdjz')) > 0,
            "knows_goal_rotation": self._infer_goal_rotation() is not None,
            "adjacent_to_target": self._check_adjacent_to_target(),
            "available_actions": available_actions,
            "source_available": self._check_source_available(),
            "current_obs": True,
            "skill_dag_loaded": True,
            "is_rotation_game": is_rotation_game,
            "ot_ready": ot_ready,
            "ot_known_directions": len(ot_summary.get("action_directions", {})),
            "ot_player_color": ot_summary.get("player_color"),
        }

    def _get_skill_for_phase(self) -> Optional[str]:
        """Get the skill ID for the current phase."""
        phase_skills = {
            # ── GENERIC phases ──
            "observe": "detect-walls-from-source",
            "discovery": "detect-walls-from-source",  # alias
            "hypothesize": "form-goal-hypothesis",
            "plan": "navigate-to-target",       # Reuse: pathfinding via A*
            "execute": "navigate-to-target",    # Reuse: execute path
            "verify": "interact-with-object",   # Reuse: interact/check
            "refine": "form-goal-hypothesis",   # Reuse: re-hypothesize
            # ── LEGACY phases ──
            "detect_walls": "detect-walls-from-source",
            "navigate_to_target": "navigate-to-target",
            "navigate_to_changer": "navigate-to-target",
            "rotate_to_goal": "rotate-to-goal",
            "navigate_to_lock": "navigate-to-target",
            "interact": "interact-with-object",
        }
        return phase_skills.get(self._phase)

    def _execute_skill(self, skill_id: str) -> bool:
        """Execute a single skill by ID. Returns True if skill made progress."""

        # ═══ World Model pre-check: simulate before executing ═══
        if self.world_model and self.world_model.memory_size() > 0:
            # Determine which action this skill will likely take
            predicted_action = self._predict_skill_action(skill_id)
            if predicted_action is not None:
                _, confidence = self._world_model_simulate(predicted_action)
                if confidence > 0.5:
                    print(f"  🌍 WM: skill={skill_id} action={predicted_action} confidence={confidence:.3f}")
                elif confidence > 0.0:
                    print(f"  🌍 WM: skill={skill_id} action={predicted_action} low confidence={confidence:.3f}")

        if skill_id == "detect-walls-from-source":
            return self._skill_detect_walls()

        elif skill_id == "form-goal-hypothesis":
            return self._skill_form_goal_hypothesis()

        elif skill_id == "navigate-to-target":
            return self._skill_navigate_to_target()

        elif skill_id == "rotate-to-goal":
            return self._skill_rotate_to_goal()

        elif skill_id == "interact-with-object":
            return self._skill_interact()

        elif skill_id == "select-skill-for-observation":
            # Not used in phase machine
            return True

        return False

    def _skill_interact(self) -> bool:
        """Execute interact-with-object skill."""
        interact_action = 5
        prev_level = self.obs.levels_completed
        self.step(interact_action)
        return self.obs.levels_completed > prev_level

    def _skill_detect_walls(self) -> bool:
        """Execute detect-walls-from-source skill."""
        if not self._walls_detected:
            self._init_pathfinder()  # Ensure pathfinder is initialized
            self._detect_wall_colors()
            self._walls_detected = True
            self.state.walls_detected = True
            self.state.set_assumption("walls_known", True)
            # Cache wall colors for fast access
            pf = getattr(self, '_pathfinder', None)
            self._detected_wall_colors = pf.wall_colors if pf and hasattr(pf, 'wall_colors') else set()
            return True
        return False  # Already done

    def _skill_form_goal_hypothesis(self) -> bool:
        """Form a goal hypothesis from current observations.
        
        Tufa Labs insight: 'if they find the right hypothesis at the first trial,
        they solve a lot of levels.' This skill analyzes the environment and forms
        a testable goal hypothesis without game-specific knowledge.
        
        Never repeats a previously failed hypothesis (tracks via _failed_hypotheses).
        Now uses three generic strategies in order:
        1. ObjectTracker known positions (generic, any game)
        2. Grid-state changes (try each action and observe effects)
        3. Tagged sprites (LS20-specific fallback)
        """
        is_refine = (self._phase == "refine")
        prev_hypothesis = (
            str(self.state.current_hypothesis.description)
            if self.state.current_hypothesis and hasattr(self.state.current_hypothesis, 'description')
            else str(self.state.current_hypothesis) if self.state.current_hypothesis else ""
        )
        
        # Track failed hypotheses across goal invalidations
        if not hasattr(self, '_failed_hypotheses'):
            self._failed_hypotheses = set()
        if prev_hypothesis and self._phase == "observe" and self.state.phase_attempts == 0:
            self._failed_hypotheses.add(prev_hypothesis)
        
        # ── STRATEGY 1: ObjectTracker generic findings ──
        ot = getattr(self, 'object_tracker', None)
        player_pos = None
        targets = []
        known_colors = {}  # color → position from ObjectTracker
        
        if ot is not None and ot.has_enough_observations():
            # Pass the grid: player_position/known_positions only exist in the
            # summary when it can see the current frame (they were silently
            # absent before — root cause of "no target ever found on holdout
            # games", 2026-07-05 report bug B2).
            _grid = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
            summary = ot.get_perception_summary(grid=_grid)
            player_color = summary.get("player_color")
            player_pos = summary.get("player_position")
            known_positions = summary.get("known_positions", {})
            wall_set = set(summary.get("wall_colors", []))
            
            for color, pos in known_positions.items():
                if color != player_color and color not in wall_set and color != 0:
                    targets.append((f"color_{color}", pos))
                    known_colors[color] = pos
            
            # Also look for the player position itself as a reference
            if player_pos is not None and player_color is not None:
                # Check what colors are adjacent to the player (potential interaction targets)
                grid = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
                if grid is not None:
                    px, py = player_pos
                    for dy in range(-5, 6, 5):
                        for dx in range(-5, 6, 5):
                            ny, nx = py + dy, px + dx
                            if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                                c = int(grid[ny, nx])
                                if c != player_color and c not in wall_set and c != 0 and c not in known_colors:
                                    targets.append((f"adjacent_color_{c}", (nx, ny)))
                                    known_colors[c] = (nx, ny)
        
        # ── STRATEGY 2: Discover actions by observing grid changes ──
        # Try each interaction action (5, 6) and observe if it changes the grid.
        # This discovers game-specific mechanics without any hardcoded tags.
        if not targets and hasattr(self, 'obs') and self.obs is not None:
            available = list(self.obs.available_actions or [])
            interaction_actions = [a for a in available if a >= 5]
            
            if interaction_actions:
                # Try each interaction action and record effects
                for act in interaction_actions:
                    if self.steps > 300:
                        break
                    grid_before = self.obs.frame[0].copy() if self.obs.frame and len(self.obs.frame) > 0 else None
                    prev_level = self.obs.levels_completed
                    
                    self.step(act)
                    
                    grid_after = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
                    level_changed = self.obs.levels_completed > prev_level
                    
                    if grid_before is not None and grid_after is not None:
                        diff = int(np.sum(grid_before != grid_after))
                        if level_changed:
                            hypothesis_text = f"Action {act} completes level (from {prev_level} to {self.obs.levels_completed})"
                            confidence = 0.9
                            self.state.update_hypothesis(hypothesis_text, confidence=confidence)
                            print(f"  💡 Hypothesis: {hypothesis_text} (conf={confidence:.2f})")
                            return True
                        if diff > 0:
                            # Store this action as potentially useful
                            self.pkm.set('mechanics', f'action_{act}_effect', diff)
                            self.pkm.set('mechanics', 'usable_actions', 
                                         self.pkm.get('mechanics', 'usable_actions', []) + [act])
                            # Mark that this action changes the grid — useful knowledge
                            hypothesis_text = f"Action {act} changes {diff} grid cells — explore to understand effect"
                            confidence = 0.5
                            self.state.update_hypothesis(hypothesis_text, confidence=confidence)
                            print(f"  💡 Hypothesis: {hypothesis_text} (conf={confidence:.2f})")
                            # Don't return — continue to find navigation targets
                            # Only add if we have a valid position to navigate to
                            if len(self.obs.frame) > 0 and self.obs.frame[0] is not None:
                                grid = self.obs.frame[0]
                                # Find the grid cell that changed — we don't know which interaction
                                # target this action affects, so skip position-based targeting
                                pass
        
        # ── STRATEGY 3: Tagged sprites (LS20-specific fallback) ──
        if not targets:
            lock_sprites = self._find_tagged_sprites('rjlbuycveu')
            changer_sprites = self._find_tagged_sprites('rhsxkxzdjz')
            for s in lock_sprites:
                targets.append(("lock", (getattr(s, 'x', 0), getattr(s, 'y', 0))))
            for s in changer_sprites:
                targets.append(("changer", (getattr(s, 'x', 0), getattr(s, 'y', 0))))
        
        # ── Select hypothesis (avoid previously failed ones) ──
        if is_refine and targets:
            primary = targets[0]
            primary_text = f"Navigate to {primary[0]} at ({primary[1][0]},{primary[1][1]})"
            
            if primary_text in prev_hypothesis or primary_text in self._failed_hypotheses:
                # Same target failed — try alternative
                if len(targets) > 1:
                    alt = targets[1]
                    hypothesis_text = f"Navigate to {alt[0]} at ({alt[1][0]},{alt[1][1]})"
                    confidence = 0.4
                elif player_pos:
                    tx, ty = primary[1]
                    px, py = player_pos
                    if tx == px and ty < py:
                        mid_x = max(0, px - 15)
                        hypothesis_text = f"Navigate to intermediate waypoint at ({mid_x},{py}) to bypass wall, then to {primary[0]} at ({tx},{ty})"
                        confidence = 0.35
                    else:
                        hypothesis_text = f"Explore alternative route to {primary[0]} at ({tx},{ty})"
                        confidence = 0.3
                else:
                    hypothesis_text = "Explore environment to find alternative path"
                    confidence = 0.2
            else:
                hypothesis_text = primary_text
                confidence = 0.5
        elif targets:
            # Skip already-failed targets
            viable = [(t, p) for t, p in targets 
                      if p is not None and f"Navigate to {t} at ({p[0]},{p[1]})" not in self._failed_hypotheses]
            if viable:
                target_type, (tx, ty) = viable[0]
                hypothesis_text = f"Navigate to {target_type} at ({tx},{ty})"
                confidence = 0.6
            elif targets:
                # All failed — try first one anyway with lower confidence
                target_type, (tx, ty) = targets[0]
                hypothesis_text = f"Navigate to {target_type} at ({tx},{ty})"
                confidence = 0.3
            else:
                hypothesis_text = "Explore environment"
                confidence = 0.2
        elif player_pos:
            # Use ObjectTracker to find potential interaction targets
            if ot is not None:
                _grid = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
                summary = ot.get_perception_summary(grid=_grid)
                known = summary.get("known_positions", {})
                player_color = summary.get("player_color")
                wall_set = getattr(self, '_detected_wall_colors', set())
                interactables = {
                    c: p for c, p in known.items()
                    if c != player_color and c not in wall_set and c != 0
                }
                if interactables:
                    px, py = player_pos
                    best_color, best_pos = min(
                        interactables.items(),
                        key=lambda kv: abs(kv[1][0]-px) + abs(kv[1][1]-py)
                    )
                    hypothesis_text = f"Interact with color {best_color} at ({best_pos[0]},{best_pos[1]})"
                    confidence = 0.45
                    # Store target for navigate-to-target
                    self._ot_target = best_pos
                    self._ot_target_color = best_color
                else:
                    px, py = player_pos
                    hypothesis_text = f"Explore from ({px},{py}) — no known targets"
                    confidence = 0.25
            else:
                hypothesis_text = f"Explore environment from ({player_pos[0]},{player_pos[1]})"
                confidence = 0.3
        else:
            hypothesis_text = "Explore environment to discover game mechanics"
            confidence = 0.1
        
        self.state.update_hypothesis(hypothesis_text, confidence=confidence)
        print(f"  💡 Hypothesis: {hypothesis_text} (conf={confidence:.2f})")
        return True

    def _navigate_one_step(self) -> Optional[int]:
        """Take one navigation step using ObjectTracker's learned action directions.

        Uses ObjectTracker.current_position() instead of self.player.x/.y,
        so it works on ANY game (no hardcoded attribute name lookup).

        Returns the action taken, or None if no movement action is known.
        Falls back to trying all available movement actions sequentially.
        """
        ot = getattr(self, 'object_tracker', None)
        if ot is not None and ot.has_enough_observations():
            summary = ot.get_perception_summary()
            action_dirs = summary.get("action_directions", {})
            if action_dirs:
                # Get current position from ObjectTracker (generic, no attribute name)
                grid = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
                if grid is not None:
                    current_pos = ot.current_position(grid)
                    for action in sorted(action_dirs.keys()):
                        prev_pos = current_pos
                        self.step(action)
                        # Re-check position after step
                        grid_after = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
                        if grid_after is not None:
                            new_pos = ot.current_position(grid_after)
                            if new_pos is not None and new_pos != prev_pos:
                                return action
                    # All known actions failed
                    return None

        # Fallback: try all available movement actions
        available = list(self.obs.available_actions or [])
        movement = [a for a in available if a in [1, 2, 3, 4]]
        if not movement:
            return None
        action = movement[self.steps % len(movement)]
        self.step(action)
        return action

    def _skill_navigate_to_target(self) -> bool:
        """Execute navigate-to-target skill using ObjectTracker-based generic
        navigation. Falls back to tag-based pathfinding for known games.

        Generic path: use GenericNavigator + best_action_toward for
        game-agnostic multi-step navigation.
        Tag-based path: A* with known wall colours (legacy, LS20-specific).
        """
        # Phase-specific tag-based target (legacy LS20 + generic)
        target_pos = None
        if self._phase in ("navigate_to_changer", "plan", "execute"):
            changers = self._find_tagged_sprites('rhsxkxzdjz')
            if changers:
                ch = changers[0]
                target_pos = (getattr(ch, 'x', 0), getattr(ch, 'y', 0))
                if self.player and self.player.x == target_pos[0] and self.player.y == target_pos[1]:
                    return True
        if self._phase in ("navigate_to_lock", "verify", "plan", "execute"):
            locks = self._find_tagged_sprites('rjlbuycveu')
            if locks:
                lk = locks[0]
                target_pos = (getattr(lk, 'x', 0), getattr(lk, 'y', 0))
                if self.player and self.player.x == target_pos[0] and self.player.y == target_pos[1]:
                    return True

        # GENERIC path: use ObjectTracker target if tags don't give us one
        if target_pos is None:
            ot_target = getattr(self, '_ot_target', None)
            if ot_target is not None:
                target_pos = ot_target
                print(f"  🎯 Using ObjectTracker target: {target_pos}")

        ot = getattr(self, 'object_tracker', None)

        # If we have a target and an ObjectTracker, use GenericNavigator
        if target_pos is not None and ot is not None:
            from .generic_navigation import GenericNavigator
            nav = GenericNavigator(ot, self.obs)

            # Ensure ObjectTracker has enough observations before navigating
            if not ot.has_enough_observations():
                print(f"  🔍 Feeding ObjectTracker before navigation...")
                import random
                for _ in range(10):
                    avail = self.obs.available_actions or [1, 2, 3, 4]
                    action = random.choice(avail)
                    grid_before = self.obs.frame[0].copy() if self.obs.frame else None
                    self.step(action)
                    grid_after = self.obs.frame[0].copy() if self.obs.frame else None
                    if grid_before is not None and grid_after is not None:
                        ot.observe(grid_before, action, grid_after)
                print(f"  ✅ ObjectTracker ready: player={ot.player_color}")

            # Verify player position is known
            pos = nav.get_player_position()
            if pos is None:
                print(f"  ⚠️ Cannot find player position — can't navigate")
                # Try single-step fallback
                if ot.has_enough_observations():
                    action = self._navigate_one_step()
                    if action is not None:
                        return True
                return False

            print(f"  🧭 Navigating from {pos} to {target_pos}...")
            success = nav.navigate(
                target_pos,
                self.step,
                max_steps=100,
                obs=self.obs,
            )
            if success:
                print(f"  ✅ Reached target!")
                return True
            else:
                print(f"  ⚠️ Navigation to {target_pos} failed")
                return False

        # Single-step fallback (B2 fix)
        if ot is not None and ot.has_enough_observations() and target_pos is None:
            action = self._navigate_one_step()
            if action is not None:
                return True
            return False

        # No target and no tracker — explore to feed ObjectTracker
        import random
        actions = [a for a in [1,2,3,4,5,6] if a in (self.obs.available_actions or [1,2,3,4])]
        if not actions:
            actions = [1,2,3,4]

        explore_steps = 5
        ot_ready = ot is not None and ot.has_enough_observations()

        if not ot_ready:
            print(f"  🔍 Exploring to feed ObjectTracker ({explore_steps} steps)...")

        for i in range(explore_steps):
            action = random.choice(actions)
            self.step(action)
            prev_lvl = getattr(self, '_explore_start_level', self.obs.levels_completed)
            if i == 0:
                self._explore_start_level = self.obs.levels_completed
            if self.obs.levels_completed > self._explore_start_level:
                print(f"  🎉 Level completed during exploration!")
                return True

        # After exploration, try navigation again if we have a target
        if ot is not None and ot.has_enough_observations():
            summary = ot.get_perception_summary()
            directions = summary.get("action_directions", {})
            if directions:
                # Try to get a target from known_positions
                known = summary.get("known_positions", {})
                if known:
                    # Pick the most interesting target (non-player, non-wall)
                    pc = ot.player_color
                    for color, pos in known.items():
                        if color != pc and color not in ot.wall_colors:
                            target_pos = (int(pos[0]), int(pos[1]))
                            print(f"  🎯 Found potential target at {target_pos} (color {color})")
                            from .generic_navigation import GenericNavigator
                            nav = GenericNavigator(ot, self.obs)
                            success = nav.navigate(
                                target_pos,
                                self.step,
                                max_steps=100,
                                obs=self.obs,
                            )
                            if success:
                                return True
                            break

                # No known positions but actions exist — use best_action_toward
                if target_pos is None and self._phase == "navigate_to_target":
                    # Last resort: explore with a bias toward new areas
                    action = random.choice(list(directions.keys()) if directions else actions)
                    print(f"  🔍 ObjectTracker learned {len(directions)} movement directions")
                    return True
            
            return len(actions) > 0  # Return True if we did any steps

        tx, ty = target_pos

        # ═══ GENERIC TIER: ObjectTracker navigation when self.player is None ═══
        # Every tier below is gated on `if self.player` — on any game whose
        # internal player object isn't found by the attribute-name guesslist
        # (all holdout games), a target could be set but NO tier could act on
        # it: the skill silently failed 5x and force-skipped the level
        # (2026-07-05 report, wa30/dc22/ar25 pattern). Here we navigate with
        # what ObjectTracker actually learned: greedy step along the learned
        # action direction that best reduces distance to the target.
        if self.player is None:
            ot = getattr(self, 'object_tracker', None)
            grid = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
            if ot is not None and grid is not None and ot.has_enough_observations():
                from .object_perception import best_action_toward
                cur_rc = ot.current_position(grid)
                if cur_rc is not None:
                    # Arrived? (x=col, y=row target vs (row, col) position)
                    if abs(cur_rc[1] - tx) + abs(cur_rc[0] - ty) <= 2:
                        print(f"  🎯 OT-nav: arrived at target ({tx},{ty})")
                        return True
                    dirs = {a: d for a in ot.action_displacements
                            if (d := ot.action_direction(a)) is not None}
                    action = best_action_toward(dirs, cur_rc, (tx, ty))
                    if action is not None:
                        self.step(action)
                        grid_after = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
                        new_rc = ot.current_position(grid_after) if grid_after is not None else None
                        moved = new_rc is not None and new_rc != cur_rc
                        print(f"  🎯 OT-nav: action {action} toward ({tx},{ty}) — "
                              f"{'moved to ' + str(new_rc) if moved else 'blocked'}")
                        # blocked => honest failure so the phase machine can
                        # escalate; the blocked-move wall evidence was already
                        # recorded by observe() inside step().
                        return moved
            # No ObjectTracker position/directions yet — fall through to the
            # player-gated tiers (which will no-op) rather than pretending.

        # Ensure pathfinder is initialized (needed for wall colors)
        pf = self._init_pathfinder()

        # ═══ UNSTICK: if player out of grid or on wall, step off first ═══
        if self.player:
            grid = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
            if grid is not None:
                wall_set = getattr(self, '_detected_wall_colors', set())
                px, py = self.player.x, self.player.y
                in_grid = (0 <= py < grid.shape[0] and 0 <= px < grid.shape[1])
                player_color = int(grid[py, px]) if in_grid else -1
                if not in_grid or player_color in wall_set:
                    reason = "out of grid" if not in_grid else f"on wall color {player_color}"
                    print(f"  🔓 Unstick: player {reason} at ({px},{py}) → stepping off")
                    for action, dx, dy in [(1, 0, -5), (3, -5, 0), (4, 5, 0), (2, 0, 5)]:
                        nx, ny = px + dx, py + dy
                        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            if int(grid[ny, nx]) not in (wall_set | {0}):
                                self.step(action)
                                return True
                    for action, dx, dy in [(1, 0, -5), (3, -5, 0), (4, 5, 0), (2, 0, 5)]:
                        nx, ny = px + dx, py + dy
                        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            self.step(action)
                            return True

        # ═══ TIER -1: Random exploration when stuck (Tufa Labs: "just give it a go") ═══
        if self.drives.stagnation_counter >= 5:
            import random
            actions = [a for a in [1, 2, 3, 4] if a in (self.obs.available_actions or [1,2,3,4])]
            if actions:
                action = random.choice(actions)
                print(f"  🎲 Random explore: action {action} (stagnation={self.drives.stagnation_counter})")
                # self.player is None on games where the attribute-name probe
                # fails (all holdout games) — this line crashed with
                # AttributeError before the guard. Fall back to ObjectTracker's
                # movement evidence, which needs no player object at all.
                prev_x, prev_y = (self.player.x, self.player.y) if self.player else (None, None)
                self.step(action)
                # If we moved, set flag so NanoPath is skipped next iteration
                # (lets A*/heuristic try the new position first)
                moved = False
                if self.player and prev_x is not None:
                    moved = abs(self.player.x - prev_x) > 2 or abs(self.player.y - prev_y) > 2
                else:
                    ot = getattr(self, 'object_tracker', None)
                    moved = bool(ot and ot.last_step_player_moved)
                if moved:
                    self.drives.stagnation_counter = 0
                    self._just_moved = True
                    print(f"  🎲 Moved! Flag set → NanoPath skipped next call.")
                return True

        # ═══ TIER 0: Micro-NN Pathfinder (primary) ═══
        nano_used = False
        if (self.pathfinder_nn and self.pathfinder_nn.available
            and self.drives.stagnation_counter < 5
            and not getattr(self, '_just_moved', False)):
            self._just_moved = False  # Consume flag
            grid = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
            if grid is not None and self.player:
                wall_colors = getattr(self, '_pathfinder', None)
                wall_set = wall_colors.wall_colors if wall_colors and hasattr(wall_colors, 'wall_colors') else set()
                # Take MULTIPLE steps in same direction (burst mode)
                # LS20 action mapping: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
                # Player moves 5 cells per step
                ACTION_DXDY = {1:(0,-5), 2:(0,5), 3:(-5,0), 4:(5,0)}

                prev_action = None
                burst_steps = 0
                max_burst = 10  # 5× less because each step moves 5 cells

                while burst_steps < max_burst:
                    action, conf = self.pathfinder_nn.predict_action(
                        grid, self.player.x, self.player.y, tx, ty,
                        wall_set, self.drives.stagnation_counter
                    )
                    if prev_action is not None and action != prev_action: break
                    if conf < 0.5: break

                    px, py = self.player.x, self.player.y
                    dx, dy = ACTION_DXDY.get(action, (0,0))
                    nx, ny = px+dx, py+dy
                    if not (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]): break
                    if int(grid[ny, nx]) in wall_set: break

                    prev_action = action
                    burst_steps += 1
                    self.step(action)
                    nano_used = True

                    if self.player.x == tx and self.player.y == ty:
                        print(f"  🤖 NanoPath: burst {burst_steps}×{['','→','↓','←','↑'][action]} → reached target!")
                        return True

                if burst_steps > 0:
                    print(f"  🤖 NanoPath: burst {burst_steps}×{['','→','↓','←','↑'][prev_action]} (conf={conf:.3f})")
                    return True

        # ═══ TIER 0b: Heuristic wall circumvention (activated early) ═══
        if self.drives.stagnation_counter >= 1:
            grid = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
            if grid is not None and self.player:
                # Get wall colors — force detection if cache empty
                wall_set = getattr(self, '_detected_wall_colors', set())
                if not wall_set:
                    self._detect_wall_colors()
                    pf = getattr(self, '_pathfinder', None)
                    wall_set = pf.wall_colors if pf and hasattr(pf, 'wall_colors') else set()
                    self._detected_wall_colors = wall_set

                # Also include color 5 (lock area barrier) and 11 if player gets blocked
                # These are secondary wall colors not detected by the primary heuristic
                wall_set = wall_set | {5, 11}

                print(f"  [DEBUG] wall_set={wall_set}, player=({self.player.x},{self.player.y}), target=({tx},{ty})")
                from cogniarc.heuristic_path import heuristic_navigate
                path = heuristic_navigate(grid, self.player.x, self.player.y, tx, ty, wall_set, max_steps=30)

                if path:
                    action, reason = path[0]
                    print(f"  🧭 Heuristic: {reason} (stagnation={self.drives.stagnation_counter})")
                    self.step(action)
                    if self.player.x == tx and self.player.y == ty:
                        return True
                    return True

        # ═══ TIER 1: A* pathfinding (fallback) ═══
        pathfinder = self._init_pathfinder()
        pathfinder.walkable_overrides.add((tx, ty))
        pathfinder.update_from_observation(self.obs)
        astar_result = pathfinder.navigate_astar((tx, ty), max_steps=200, obs=self.obs)

        if astar_result:
            return True

        # ═══ A* FAILED → try wall circumvention via intermediate waypoint ═══
        # Tufa Labs insight: "they think no but that's outside the maze so you
        # can't move there. And the human would just give it a go and see what happens."
        if self.player and tx is not None:
            px, py = self.player.x, self.player.y
            
            # ═══ UNSTICK: if player is out of grid or on wall, step off first ═══
            grid = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None
            if grid is not None:
                wall_set = getattr(self, '_detected_wall_colors', set())
                in_grid = (0 <= py < grid.shape[0] and 0 <= px < grid.shape[1])
                player_color = int(grid[py, px]) if in_grid else -1
                if not in_grid or player_color in wall_set:
                    reason = "out of grid" if not in_grid else f"on wall color {player_color}"
                    print(f"  🔓 Player stuck ({reason}) at ({px},{py}), stepping off...")
                    for action, dx, dy in [(1, 0, -5), (3, -5, 0), (4, 5, 0), (2, 0, 5)]:
                        nx, ny = px + dx, py + dy
                        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            if int(grid[ny, nx]) not in (wall_set | {0}):
                                self.step(action)
                                return True
                    for action, dx, dy in [(1, 0, -5), (3, -5, 0), (4, 5, 0), (2, 0, 5)]:
                        nx, ny = px + dx, py + dy
                        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                            self.step(action)
                            return True
            
            # Try heuristic wall circumvention
            if grid is not None:
                from cogniarc.heuristic_path import heuristic_navigate
                wall_set = getattr(self, '_detected_wall_colors', set()) | {5, 11}
                path = heuristic_navigate(grid, px, py, tx, ty, wall_set, max_steps=30)
                if path:
                    action, reason = path[0]
                    print(f"  🧭 Heuristic: {reason}")
                    self.step(action)
                    return True
            
            # If same column blocked, try going DOWN first to exit wall zone
            if tx == px:
                # Player and target on same column but A* blocked.
                # Likely a wall between them. Go down to exit wall zone, then left.
                if ty < py:
                    # Target is above — go down first to get below the wall
                    print(f"  🧭 Same column blocked: descending to bypass wall...")
                    for _ in range(3):  # 3 steps down = 15 cells (LS20 step=5)
                        self.step(2)  # DOWN
                    # Now try going left from the new position
                    mid_x = max(0, self.player.x - 15)
                    print(f"  🧭 After descend: trying waypoint ({mid_x},{self.player.y})")
                    waypoint_result = pathfinder.navigate_astar(
                        (mid_x, self.player.y), max_steps=80, obs=self.obs
                    )
                    if waypoint_result:
                        pathfinder.update_from_observation(self.obs)
                        retry = pathfinder.navigate_astar((tx, ty), max_steps=200, obs=self.obs)
                        if retry:
                            return True
                
                # Fallback: original waypoint left
                mid_x = max(0, px - 15)
                print(f"  🧭 A* blocked: trying waypoint ({mid_x},{py}) before ({tx},{ty})")
                # Try navigating to waypoint
                waypoint_result = pathfinder.navigate_astar((mid_x, py), max_steps=50, obs=self.obs)
                if waypoint_result:
                    print(f"  🧭 Waypoint reached! Now path to target...")
                    # Update observation and retry A* to target
                    pathfinder.update_from_observation(self.obs)
                    retry = pathfinder.navigate_astar((tx, ty), max_steps=200, obs=self.obs)
                    if retry:
                        return True
            # If same row blocked, try going up first  
            elif ty == py:
                mid_y = max(0, py - 15)
                print(f"  🧭 A* blocked: trying waypoint ({px},{mid_y}) before ({tx},{ty})")
                waypoint_result = pathfinder.navigate_astar((px, mid_y), max_steps=50, obs=self.obs)
                if waypoint_result:
                    print(f"  🧭 Waypoint reached! Now path to target...")
                    pathfinder.update_from_observation(self.obs)
                    retry = pathfinder.navigate_astar((tx, ty), max_steps=200, obs=self.obs)
                    if retry:
                        return True

        # ═══ A* FAILED → Micro-NN or World Model fallback ═══
        if self.action_predictor or (self.world_model and self.world_model.memory_size() > 0):
            return self._world_model_navigate_fallback(tx, ty)

        # ═══ Auto-interact: if player is on an interactable, execute action ═══
        if self.player and target_pos:
            tx, ty = target_pos
            if self.player.x == tx and self.player.y == ty:
                # On a changer → rotate
                if self._phase in ("navigate_to_changer", "plan", "execute"):
                    changers = self._find_tagged_sprites('rhsxkxzdjz')
                    if changers and self.player.x == getattr(changers[0], 'x', -1) and self.player.y == getattr(changers[0], 'y', -1):
                        print(f"  🔄 Auto-rotating on changer at ({tx},{ty})...")
                        for _ in range(10):  # Max 10 rotations
                            prev_level = self.obs.levels_completed
                            self.step(6)  # Action 6 = rotate
                            if self.obs.levels_completed > prev_level:
                                return True
                            # Check if rotation changed
                            current_rot = getattr(self.game, 'cklxociuu', 0) if self.game else 0
                            goal_rot = self._infer_goal_rotation()
                            if goal_rot is not None and current_rot == goal_rot:
                                print(f"  🔄 Rotation matched goal: {current_rot}")
                                return True
                        return True  # Rotated enough, move on
                # On a lock → already collected by walking on it
                if self._phase in ("navigate_to_lock", "verify"):
                    locks = self._find_tagged_sprites('rjlbuycveu')
                    if locks and self.player.x == getattr(locks[0], 'x', -1) and self.player.y == getattr(locks[0], 'y', -1):
                        print(f"  🔑 Lock collected at ({tx},{ty})")
                        return True
                # Generic: on any ObjectTracker target → try interact (action 5)
                ot_target = getattr(self, '_ot_target', None)
                if ot_target and self.player.x == ot_target[0] and self.player.y == ot_target[1]:
                    print(f"  🖐️ Interacting with ObjectTracker target at ({tx},{ty})...")
                    prev_level = self.obs.levels_completed
                    if 5 in (self.obs.available_actions or []):
                        self.step(5)
                        if self.obs.levels_completed > prev_level:
                            print(f"  ✅ Level completed via interact!")
                            return True
                    if 6 in (self.obs.available_actions or []):
                        self.step(6)
                        if self.obs.levels_completed > prev_level:
                            return True
                    self._ot_target = None
                    return True

        return False

    def _skill_rotate_to_goal(self) -> bool:
        """Execute rotate-to-goal skill. Loops until rotation matches goal."""
        goal_rot = self._infer_goal_rotation()
        if goal_rot is None:
            return False

        current_rot = getattr(self.game, 'cklxociuu', 0) if self.game else 0
        if current_rot == goal_rot:
            return True  # Already rotated

        # Need to reach changer first
        changers = self._find_tagged_sprites('rhsxkxzdjz')
        if not changers:
            return False

        ch = changers[0]
        cx, cy = getattr(ch, 'x', 0), getattr(ch, 'y', 0)

        # Navigate to changer if not adjacent
        if self.player and abs(self.player.x - cx) + abs(self.player.y - cy) > 1:
            pathfinder = self._init_pathfinder()
            pathfinder.walkable_overrides.add((cx, cy))
            pathfinder.navigate_to((cx, cy), max_steps=100)
            return True  # Made progress toward changer

        # At changer — search for the goal rotation instead of blindly
        # spamming action 4 then 3 in a fixed loop. The transition function
        # (what does each action actually do to the rotation counter, from
        # each state?) is LEARNED from real interaction — never assumed — via
        # next_probe_or_action(): it either takes an untried action to
        # observe its effect ("probe") or replays a path already confirmed by
        # real observation ("advance"). The table persists across calls
        # (self._rotation_transition_table) so later phases/levels reuse
        # what earlier ones learned instead of re-probing from scratch.
        from .program_synthesis import next_probe_or_action
        if not hasattr(self, '_rotation_transition_table'):
            self._rotation_transition_table = {}
        table = self._rotation_transition_table
        rotation_actions = [3, 4]  # the two changer actions this skill uses

        max_actions = 20
        for _ in range(max_actions):
            current_rot = getattr(self.game, 'cklxociuu', 0)
            mode, action = next_probe_or_action(table, current_rot, goal_rot, rotation_actions)
            if mode == "done":
                return True
            if mode == "stuck":
                # Every action tried from this state; the goal is provably
                # unreachable with what's been learned so far — stop instead
                # of spinning, unlike the old fixed-count loop.
                return False
            self.step(action)
            new_rot = getattr(self.game, 'cklxociuu', 0)
            table[(current_rot, action)] = new_rot

        return getattr(self.game, 'cklxociuu', 0) == goal_rot

    def _advance_phase(self, success: bool):
        """Advance phase based on skill result. Syncs with ScientificState.

        GENERIC phase flow (Tufa Labs interview insight):
          observe → hypothesize → plan → execute → verify → refine
          
        Replaces the legacy LS20-specific phases (navigate_to_changer,
        rotate_to_goal, navigate_to_lock) with a game-agnostic cognitive loop.
        The agent discovers mechanics through observation, not hardcoded tags.
        
        GoalSanityChecker integration: after 3+ refine failures, the goal is
        invalidated and the agent returns to observe (fresh exploration).
        """
        old_phase = self._phase
        
        # ── GENERIC: Observe ──
        if self._phase == "observe" and success:
            # Scout done — form a hypothesis
            self._phase = "hypothesize"
        elif self._phase == "observe" and not success:
            # Observation failed — retry (or walls already known, that's OK)
            self._phase = "hypothesize"  # Assume we know enough
        
        # ── GENERIC: Hypothesize ──
        elif self._phase == "hypothesize" and success:
            # Hypothesis formed — plan actions
            self._phase = "plan"
        elif self._phase == "hypothesize" and not success:
            # Can't form hypothesis — re-observe
            self._phase = "observe"
        
        # ── GENERIC: Plan ──
        elif self._phase == "plan" and success:
            # Plan ready — execute it
            self._phase = "execute"
        elif self._phase == "plan" and not success:
            # Plan failed (e.g., no path) — refine hypothesis
            self._phase = "refine"
        
        # ── GENERIC: Execute ──
        elif self._phase == "execute" and success:
            # Execution done — verify result
            self._phase = "verify"
        elif self._phase == "execute" and not success:
            # Execution failed — refine
            self._phase = "refine"
        
        # ── GENERIC: Verify ──
        elif self._phase == "verify" and success:
            # Level completed!
            self._phase = "complete"
        elif self._phase == "verify" and not success:
            # Verification failed — refine hypothesis
            self._phase = "refine"
        
        # ── GENERIC: Refine ──
        elif self._phase == "refine" and success:
            # Refined hypothesis — re-plan
            self._phase = "plan"
        elif self._phase == "refine" and not success:
            # Refine failed — try again (GoalSanityChecker will catch loops)
            self._phase = "hypothesize"
        
        # ── LEGACY FALLBACK: keep old phases working for backward compat ──
        elif self._phase == "detect_walls" and success:
            ot = getattr(self, 'object_tracker', None)
            has_tagged_changer = len(self._find_tagged_sprites('rhsxkxzdjz')) > 0
            if ot is not None and ot.has_enough_observations() and not has_tagged_changer:
                self._phase = "navigate_to_target"
            else:
                self._phase = "navigate_to_changer"
        
        elif self._phase == "navigate_to_target" and success:
            self._phase = "interact"
        
        elif self._phase == "navigate_to_changer" and success:
            self._phase = "rotate_to_goal"
        elif self._phase == "rotate_to_goal" and success:
            self._phase = "navigate_to_lock"
        elif self._phase == "navigate_to_lock" and success:
            if self.player:
                locks = self._find_tagged_sprites('rjlbuycveu')
                if locks:
                    lk = locks[0]
                    lx, ly = getattr(lk, 'x', 0), getattr(lk, 'y', 0)
                    if self.player.x == lx and self.player.y == ly:
                        self._phase = "complete"
                    elif abs(self.player.x - lx) + abs(self.player.y - ly) == 1:
                        if self.player.x < lx: action = 1
                        elif self.player.x > lx: action = 3
                        elif self.player.y < ly: action = 2
                        else: action = 4
                        print(f"  🔑 Stepping onto lock at ({lx},{ly}) — action {action}")
                        self.step(action)
                        self._phase = "complete"
                    else:
                        self._phase = "navigate_to_lock"
                else:
                    self._phase = "interact"
            else:
                self._phase = "interact"
        
        elif self._phase == "interact" and success:
            self._phase = "complete"
        
        # Sync with ScientificState
        if self._phase != old_phase:
            self.state.phase = self._phase
            self.state.phase_attempts = 0
