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
        """
        is_refine = (self._phase == "refine")
        prev_hypothesis = str(self.state.current_hypothesis) if self.state.current_hypothesis else ""
        
        # Track failed hypotheses across goal invalidations
        if not hasattr(self, '_failed_hypotheses'):
            self._failed_hypotheses = set()
        if prev_hypothesis and self._phase == "observe" and self.state.phase_attempts == 0:
            # We're back at observe after a GOAL INVALID — remember the failed hypothesis
            self._failed_hypotheses.add(prev_hypothesis)
        
        # Gather evidence
        ot = getattr(self, 'object_tracker', None)
        player_pos = None
        targets = []
        
        if ot is not None and ot.has_enough_observations():
            summary = ot.get_perception_summary()
            player_color = summary.get("player_color")
            player_pos = summary.get("player_position")
            known_positions = summary.get("known_positions", {})
            for color, pos in known_positions.items():
                if color != player_color:
                    targets.append((color, pos))
        
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
                      if f"Navigate to {t} at ({p[0]},{p[1]})" not in self._failed_hypotheses]
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

        Generic path: use ObjectTracker action directions to move around,
        recording wall/floor evidence with each step.
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

        # GENERIC path: use ObjectTracker if tags don't give us a target
        ot = getattr(self, 'object_tracker', None)
        if ot is not None and ot.has_enough_observations() and target_pos is None:
            action = self._navigate_one_step()
            if action is not None:
                return True
            return False

        # LEGACY path: A* with known wall colours (LS20-specific)
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

        # ═══ A* FAILED → try wall circumvention via intermediate waypoint ═══
        # Tufa Labs insight: "they think no but that's outside the maze so you
        # can't move there. And the human would just give it a go and see what happens."
        if self.player and tx is not None:
            px, py = self.player.x, self.player.y
            # If same column blocked, try going left first
            if tx == px:
                # Waypoint: go 15 cells left, then path to target from there
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
