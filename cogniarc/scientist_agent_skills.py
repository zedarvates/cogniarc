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


class SkillsMixin:
    """Skill execution + phase-advance helpers."""

    def _build_skill_context(self) -> Dict[str, Any]:
        """Build context dict for skill selection."""
        available_actions = list(self.obs.available_actions or [])
        is_rotation_game = 6 in available_actions and not any(a in available_actions for a in [1, 2, 3, 4])

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
        }

    def _get_skill_for_phase(self) -> Optional[str]:
        """Get the skill ID for the current phase."""
        phase_skills = {
            "detect_walls": "detect-walls-from-source",
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

    def _skill_navigate_to_target(self) -> bool:
        """Execute navigate-to-target skill. Falls back to world model if A* fails."""
        target_pos = None

        # Phase 1: Navigate to changer
        if self._phase == "navigate_to_changer":
            changers = self._find_tagged_sprites('rhsxkxzdjz')
            if changers:
                ch = changers[0]
                target_pos = (getattr(ch, 'x', 0), getattr(ch, 'y', 0))

                if self.player and self.player.x == target_pos[0] and self.player.y == target_pos[1]:
                    return True  # At changer

        # Phase 2: Navigate to lock
        elif self._phase == "navigate_to_lock":
            locks = self._find_tagged_sprites('rjlbuycveu')
            if locks:
                lk = locks[0]
                target_pos = (getattr(lk, 'x', 0), getattr(lk, 'y', 0))

                if self.player and self.player.x == target_pos[0] and self.player.y == target_pos[1]:
                    return True  # At lock

        if target_pos is None:
            return False

        tx, ty = target_pos

        # Ensure pathfinder is initialized (needed for wall colors)
        pf = self._init_pathfinder()

        # ═══ TIER 0: Micro-NN Pathfinder (primary) ═══
        nano_used = False
        if self.pathfinder_nn and self.pathfinder_nn.available and self.drives.stagnation_counter < 5:
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

        # ═══ TIER 0b: Heuristic wall circumvention (when stuck) ═══
        if self.drives.stagnation_counter >= 3:
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

        # ═══ A* FAILED → Micro-NN or World Model fallback ═══
        if self.action_predictor or (self.world_model and self.world_model.memory_size() > 0):
            return self._world_model_navigate_fallback(tx, ty)

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

        # At changer — cycle rotation using R+L (actions 4+3)
        max_cycles = 20
        for _ in range(max_cycles):
            current_rot = getattr(self.game, 'cklxociuu', 0)
            if current_rot == goal_rot:
                return True
            self.step(4)  # Right
            self.step(3)  # Left

        return False  # Failed to reach target rotation

    def _advance_phase(self, success: bool):
        """Advance phase based on skill result. Syncs with ScientificState."""
        old_phase = self._phase

        if self._phase == "detect_walls" and success:
            self._phase = "navigate_to_changer"
        elif self._phase == "navigate_to_changer" and success:
            self._phase = "rotate_to_goal"
        elif self._phase == "rotate_to_goal" and success:
            self._phase = "navigate_to_lock"
        elif self._phase == "navigate_to_lock" and success:
            # In LS20, locks are collected by walking ON them, not by interact (action 5).
            # Skip interact phase — step onto the lock directly.
            if self.player:
                locks = self._find_tagged_sprites('rjlbuycveu')
                if locks:
                    lk = locks[0]
                    lx, ly = getattr(lk, 'x', 0), getattr(lk, 'y', 0)
                    if self.player.x == lx and self.player.y == ly:
                        self._phase = "complete"  # Already on lock
                    elif abs(self.player.x - lx) + abs(self.player.y - ly) == 1:
                        # Adjacent: step onto lock
                        if self.player.x < lx: action = 1
                        elif self.player.x > lx: action = 3
                        elif self.player.y < ly: action = 2
                        else: action = 4
                        print(f"  🔑 Stepping onto lock at ({lx},{ly}) — action {action}")
                        self.step(action)
                        self._phase = "complete"
                    else:
                        self._phase = "navigate_to_lock"  # Need more movement
                else:
                    self._phase = "interact"  # Fallback
            else:
                self._phase = "interact"
        elif self._phase == "interact" and success:
            self._phase = "complete"

        # Sync with ScientificState
        if self._phase != old_phase:
            self.state.phase = self._phase
            self.state.phase_attempts = 0
