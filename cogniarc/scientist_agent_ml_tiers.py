"""ML-tier escalation mixin for ScientistAgent.

Groups the methods that query the optional ML tiers (V-JEPA world model,
nano-LLM, micro-NN navigation fallback) used to simulate or propose actions
when deterministic skills/A* need a hint or a fallback. Extracted verbatim
from scientist_agent.py (no logic changes) to shrink that file's God-object
footprint — see README "Logic vs Micro-NN" section for the tier policy.

Mixed into ScientistAgent; relies on attributes set up in its __init__
(self.world_model, self.nano_llm, self.nano_harness, self.obs, self.player,
self.drives, self.action_predictor, self._pathfinder).
"""
from typing import Optional


class MLTiersMixin:
    """World model / nano-LLM / micro-NN escalation helpers."""

    # ═══════ WORLD MODEL TOOL ═══════

    def _world_model_simulate(self, action: int):
        """Simulate the effect of an action using the world model.

        Queries the V-JEPA encoder + k-NN predictor:
        "If I take action X from my current state, what state do I predict?"

        Returns:
            (predicted_latent, confidence) or (None, 0.0) if world model unavailable
        """
        if not self.world_model:
            return None, 0.0

        if not self.obs.frame or len(self.obs.frame) == 0:
            return None, 0.0

        obs = self.obs.frame[0]
        predicted, confidence = self.world_model.predict(obs, action)
        return predicted, confidence

    def _nano_propose_action(self, recent_history: str = "") -> Optional[int]:
        """Nano-LLM tier: ask Qwen2.5-0.5B (via Ollama) for a safe next action.

        Sits between the micro-NN tier and the V-JEPA world model in the
        escalation chain. Proposals pass through NanoLLMHarness, which rejects
        actions that hit known walls or repeat known failures. Returns the
        validated action number, or None if the nano-LLM is unavailable.
        """
        if not self.nano_harness or not self.nano_llm or not self.nano_llm.available:
            return None
        if not self.obs.frame or len(self.obs.frame) == 0:
            return None

        available = list(self.obs.available_actions or [])
        if not available:
            return None

        player_pos = getattr(self, '_player_pos', None)
        wall_cells = getattr(self, '_wall_cells', None)
        game_state = grid_to_text(self.obs.frame[0]) if 'grid_to_text' in globals() else str(self.obs.frame[0])

        action, conf, reasoning, is_safe = self.nano_harness.propose_safe(
            game_state=game_state,
            available_actions=available,
            player_pos=player_pos,
            recent_history=recent_history,
            wall_cells=wall_cells,
        )
        print(f"  🤖 NanoLLM: action={action} conf={conf:.2f} safe={is_safe} ({reasoning})")
        return action if is_safe else None

    def _world_model_report(self) -> str:
        """Human-readable report of world model state."""
        if not self.world_model:
            return "World model: disabled"

        mem = self.world_model.memory_size()
        return f"World model: {mem} transitions memorized"

    def _predict_skill_action(self, skill_id: str) -> Optional[int]:
        """Predict which action a skill will likely take (for world model pre-check).

        Maps skill IDs to their most probable action number.
        Returns None if the skill doesn't involve a predictable action.
        """
        action_map = {
            "navigate-to-target": None,  # Multi-action A* — too complex
            "rotate-to-goal": 6,          # Rotation action
            "interact-with-object": 5,    # Interact
            "detect-walls-from-source": None,  # No step
        }

        # Navigation: determine direction from phase
        if skill_id == "navigate-to-target":
            if self._phase == "navigate_to_changer":
                # Predict direction toward changer
                if self.player:
                    changers = self._find_tagged_sprites('rhsxkxzdjz')
                    if changers:
                        ch = changers[0]
                        cx, cy = getattr(ch, 'x', 0), getattr(ch, 'y', 0)
                        px, py = self.player.x, self.player.y
                        if cx > px: return 1  # right
                        if cx < px: return 3  # left
                        if cy > py: return 2  # down
                        if cy < py: return 4  # up
            elif self._phase == "navigate_to_lock":
                if self.player:
                    locks = self._find_tagged_sprites('rjlbuycveu')
                    if locks:
                        lk = locks[0]
                        lx, ly = getattr(lk, 'x', 0), getattr(lk, 'y', 0)
                        px, py = self.player.x, self.player.y
                        if lx > px: return 1
                        if lx < px: return 3
                        if ly > py: return 2
                        if ly < py: return 4

        return action_map.get(skill_id)

    def _world_model_navigate_fallback(self, tx: int, ty: int) -> bool:
        """When A* fails, try to circumvent walls instead of forcing through.

        Strategy:
        1. If stagnant (<3 fails): micro-NN suggests toward-target direction
        2. If stagnant (≥3 fails): wall detected → try PERPENDICULAR escape
        3. World Model V-JEPA as last resort
        """
        if not self.player:
            return False

        px, py = self.player.x, self.player.y
        stagnation = self.drives.stagnation_counter

        # Determine primary axis toward target
        dx = tx - px
        dy = ty - py
        toward_vertical = abs(dy) > abs(dx)  # Target is more up/down than left/right

        # ═══ TIER 1: Stuck ≥ 3 times → WALL CIRCUMVENTION ═══
        if stagnation >= 3:
            wall_colors = getattr(self, '_pathfinder', None)
            wall_set = wall_colors.wall_colors if wall_colors and hasattr(wall_colors, 'wall_colors') else set()
            grid = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None

            if grid is not None and wall_set:
                # Check if toward-target cell is a wall
                toward_action = None
                if dy < 0 and py > 0: toward_action = 4  # up
                elif dy > 0: toward_action = 2  # down
                elif dx > 0: toward_action = 1  # right
                elif dx < 0: toward_action = 3  # left

                # Try perpendicular escape: left/right if vertical target, up/down if horizontal
                escape_actions = [1, 3] if toward_vertical else [2, 4]  # perpendicular

                for action in escape_actions:
                    # Check if escape cell is walkable (not a wall)
                    nx, ny = px, py
                    if action == 1: nx = px + 1
                    elif action == 3: nx = px - 1
                    elif action == 2: ny = py + 1
                    elif action == 4: ny = py - 1

                    if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                        cell_color = grid[ny, nx]
                        if cell_color not in wall_set:
                            action_names = ['', 'right', 'down', 'left', 'up']
                            print(f"  🧱 Wall blocked → escaping {action_names[action]} (cell {cell_color} is walkable)")
                            self.step(action)
                            return True

                # All perpendicular cells are walls → trapped
                print(f"  🧱 Surrounded by walls at ({px},{py}) — cannot escape")
                return False

        # ═══ TIER 2: Micro-NN (low stagnation) ═══
        if self.action_predictor and self.action_predictor.available:
            wall_colors = getattr(self, '_pathfinder', None)
            wall_set = wall_colors.wall_colors if wall_colors and hasattr(wall_colors, 'wall_colors') else set()
            grid = self.obs.frame[0] if self.obs.frame and len(self.obs.frame) > 0 else None

            if grid is not None:
                best_action = None
                best_conf = 0.0

                for action in [1, 2, 3, 4]:
                    if action == 1 and px >= tx: continue
                    if action == 3 and px <= tx: continue
                    if action == 2 and py >= ty: continue
                    if action == 4 and py <= ty: continue

                    prob = self.action_predictor.predict_action(
                        (px, py), (tx, ty), action, wall_set, grid,
                        stagnation=stagnation, steps=self.steps
                    )
                    if prob > best_conf:
                        best_conf = prob
                        best_action = action

                if best_action is not None and best_conf > 0.5:
                    print(f"  ⚡ MicroNN: A* failed → stepping {['','right','down','left','up'][best_action]} (prob={best_conf:.3f})")
                    self.step(best_action)
                    return True

        # ═══ TIER 3: World Model V-JEPA (last resort) ═══
        if self.world_model and self.world_model.memory_size() > 0:
            return self._world_model_navigate_fallback_vjepa(tx, ty)

        return False

    def _world_model_navigate_fallback_vjepa(self, tx: int, ty: int) -> bool:
        """V-JEPA world model fallback (original implementation)."""
        if not self.player:
            return False

        px, py = self.player.x, self.player.y
        best_action = None
        best_conf = 0.0

        for action in [1, 2, 3, 4]:
            if action == 1 and px >= tx: continue
            if action == 3 and px <= tx: continue
            if action == 2 and py >= ty: continue
            if action == 4 and py <= ty: continue

            _, conf = self._world_model_simulate(action)
            if conf > best_conf:
                best_conf = conf
                best_action = action

        if best_action is not None and best_conf > 0.3:
            print(f"  🌍 WM fallback: A* failed → stepping {['','right','down','left','up'][best_action]} (conf={best_conf:.3f})")
            self.step(best_action)
            return True

        return False
