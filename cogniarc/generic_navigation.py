"""
Generic navigation — move toward any target on any grid.

Uses ObjectTracker for player position + wall colors + action directions,
combined with A* path planning on the observed grid.

Works on ALL games, not just LS20 with tag-based sprites.
"""

from typing import List, Tuple, Optional, Set, Dict
import numpy as np
from collections import deque

from .pathfinding import GridMap, astar, astar_path_to_actions
from .object_perception import ObjectTracker


class GenericNavigator:
    """Navigate toward a target using ObjectTracker + A*.

    No dependency on tag-based sprites or self.player.
    Works on ANY game where ObjectTracker can identify the player.
    """

    def __init__(self, tracker: ObjectTracker, obs=None):
        self.tracker = tracker
        self.grid_map: Optional[GridMap] = None
        self.obs = obs
        self.walkable_overrides: Set[Tuple[int, int]] = set()

    def update_grid(self, obs) -> None:
        """Build/update grid map from observation + ObjectTracker knowledge."""
        self.obs = obs
        if not hasattr(obs, 'frame') or not obs.frame:
            return

        grid = obs.frame[0]
        wall_colors = self.tracker.wall_colors

        if self.grid_map is None:
            # Pass wall_colors explicitly. None = use heuristic.
            # Empty set = no walls known, don't use heuristic.
            wc_for_map = wall_colors if wall_colors is not None else None
            self.grid_map = GridMap.from_observation(obs, wc_for_map)
        else:
            # Update walkable map with current wall knowledge
            new_walkable = np.ones_like(grid, dtype=bool)
            for wc in wall_colors:
                new_walkable[grid == wc] = False
            for x, y in self.walkable_overrides:
                if 0 <= x < self.grid_map.width and 0 <= y < self.grid_map.height:
                    new_walkable[y, x] = True
            self.grid_map.walkable = new_walkable

    def get_player_position(self) -> Optional[Tuple[int, int]]:
        """Find player position via ObjectTracker player_color.

        Scans the latest grid for the player color region and returns
        its center-of-mass position as (col=x, row=y).

        Region center is (row, col) internally — we swap to (x, y).
        """
        if self.obs is None:
            return None
        pc = self.tracker.player_color
        if pc is None:
            return None

        from .object_perception import segment_regions
        try:
            regions = segment_regions(self.obs.frame[0])
            for r in regions:
                if r.color == pc:
                    # Region center is (row, col). Return (col, row) = (x, y).
                    col = int(round(r.center[1]))
                    row = int(round(r.center[0]))
                    return (col, row)
        except Exception:
            return None
        return None

    def find_path(self, target: Tuple[int, int]) -> Optional[List[int]]:
        """Find A* path from current player position to target.

        Returns list of actions, or None if no path exists.
        """
        pos = self.get_player_position()
        if pos is None or self.grid_map is None:
            return None

        path = astar(self.grid_map, pos, target)
        if path is None:
            return None

        return astar_path_to_actions(path)

    def validate_and_adapt(
        self,
        target: Tuple[int, int],
        obs,
        action: int,
    ) -> Tuple[bool, Optional[List[int]]]:
        """After executing an action, check if we're still on track.

        If the action moved us closer to target: continue.
        If it didn't (hit wall, teleported): rebuild and replan.

        Returns (still_valid, new_path_if_needed).
        """
        self.update_grid(obs)
        pos = self.get_player_position()
        if pos is None:
            return False, None

        if pos == target:
            return True, []  # Arrived!

        # Check if we moved as expected
        new_path = self.find_path(target)
        if new_path is None:
            return False, None

        return True, new_path

    def navigate(
        self,
        target: Tuple[int, int],
        step_fn,
        max_steps: int = 100,
        obs=None,
    ) -> bool:
        """Full navigation loop: greedily step toward target using learned action
        directions. Replans after each step in case of walls or teleportation.

        Uses the ObjectTracker's learned action directions (best_action_toward)
        for each step — no hardcoded action mapping.

        Args:
            target: (x, y) position to navigate to
            step_fn: callable(action) that executes one step and returns new obs
            max_steps: maximum steps before giving up
            obs: initial observation

        Returns: True if target reached
        """
        if obs is not None:
            self.obs = obs
            self.update_grid(obs)

        steps_taken = 0
        stuck_count = 0
        prev_pos = None

        while steps_taken < max_steps:
            pos = self.get_player_position()
            if pos is None:
                print(f"  ⚠️ Lost player position at step {steps_taken}")
                return False

            # Check if we arrived
            if abs(pos[0] - target[0]) <= 2 and abs(pos[1] - target[1]) <= 2:
                print(f"  ✅ Reached target area at {pos}")
                return True

            # Detect stagnation
            if prev_pos == pos:
                stuck_count += 1
                if stuck_count >= 5:
                    print(f"  ⚠️ Stuck at {pos} for 5 steps")
                    return False
            else:
                stuck_count = 0
            prev_pos = pos

            # Use the ObjectTracker's best_action_toward for actual step direction
            from .object_perception import best_action_toward
            ot = self.tracker

            # Get action directions from tracker summary
            try:
                summary = ot.get_perception_summary()
            except Exception:
                summary = {}
            action_dirs = summary.get("action_directions", {})
            if not action_dirs:
                # Fallback: build from action_direction() calls
                action_dirs = {}
                for a in [1, 2, 3, 4, 5, 6]:
                    d = ot.action_direction(a)
                    if d is not None:
                        action_dirs[a] = d

            # Convert position (x,y) = (col,row) to (row,col) for best_action_toward
            current_rc = (pos[1], pos[0])

            action = best_action_toward(action_dirs, current_rc, target)
            if action is None:
                # Fallback: try each action and see what moves us toward target
                print(f"  ⚠️ No action toward target at {pos}, trying random")
                import random
                if self.obs and hasattr(self.obs, 'available_actions'):
                    avail = self.obs.available_actions or [1, 2, 3, 4]
                    action = random.choice(avail)
                else:
                    return False

            # Execute the action
            new_obs = step_fn(action)
            steps_taken += 1
            self.obs = new_obs

            # Update grid with latest obs
            self.update_grid(new_obs)

        pos = self.get_player_position()
        return pos is not None and abs(pos[0] - target[0]) <= 2 and abs(pos[1] - target[1]) <= 2


def navigate_to_target_with_tracker(
    agent,
    target: Tuple[int, int],
    max_steps: int = 100,
) -> bool:
    """Convenience: create GenericNavigator from agent and navigate to target.

    Works with any ScientistAgent that has an ObjectTracker.
    """
    ot = getattr(agent, 'object_tracker', None)
    if ot is None:
        print("  ⚠️ No ObjectTracker available")
        return False

    navigator = GenericNavigator(ot, agent.obs)
    return navigator.navigate(target, agent.step, max_steps=max_steps, obs=agent.obs)
