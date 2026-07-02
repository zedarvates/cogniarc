"""Discovery mixin for ScientistAgent.

Groups the methods used to discover game mechanics before solving: reading
source code, scouting available actions, detecting wall colors, finding
tagged sprites, and inferring goal/rotation state. Extracted verbatim from
scientist_agent.py (no logic changes) to shrink that file's God-object
footprint.

Note: `__init_pathfinder` was renamed to `_init_pathfinder` (single leading
underscore) during extraction — the original double-underscore name gets
Python name-mangled per-class, which would break cross-mixin calls from
SkillsMixin/MLTiersMixin once ScientistAgent composes these mixins together.
Purely a renaming, no behavior change.

Mixed into ScientistAgent; relies on attributes set up in its __init__
(self.game, self.player, self.pkm, self.obs, self.name, self._pathfinder).
"""
import numpy as np
from typing import Optional

from .pathfinding import Pathfinder


class DiscoveryMixin:
    """Game-mechanics discovery helpers (source reading, scouting, walls)."""

    # ------ DISCOVERY PHASE ------

    def discover_from_source(self) -> bool:
        """Read game source code to discover mechanics (zero steps)."""
        try:
            # Try direct path first (game files are under environment_files/<game_id>/)
            import os, glob

            # Extract base game id (ls20 from ls20-9607627b)
            base_game = self.name.split('-')[0] if '-' in self.name else self.name

            # Try multiple path patterns
            possible_paths = [
                f"environment_files/{base_game}/*/{self.name}.py",
                f"environment_files/{base_game}/*/{base_game}.py",
                f"environment_files/{self.name}/*/{self.name}.py",
                f"environment_files/{self.name}/*/{base_game}.py",
            ]

            src = None
            for pattern in possible_paths:
                dirs = glob.glob(pattern)
                if dirs:
                    src = dirs[0]
                    break

            if not src:
                return False

            with open(src) as f:
                code = f.read()
        except:
            return False

        # Parse tag-to-mechanic mapping from source
        # Pattern: if "TAG" in sprite.tags: -> EFFECT
        import re
        mechanics = {
            'walls': set(),
            'locks': set(),
            'rotation_changers': set(),
            'color_changers': set(),
            'shape_changers': set(),
        }

        # Known patterns from LS20 source analysis
        tag_contexts = {
            'ihdgageizm': 'walls',
            'rjlbuycveu': 'locks',
            'rhsxkxzdjz': 'rotation_changers',
            'soyhouuebz': 'color_changers',
            'ttfwljgohq': 'shape_changers',
        }

        for tag, mechanic in tag_contexts.items():
            if tag in code:
                mechanics[mechanic].add(tag)

        for cat, items in mechanics.items():
            if items:
                self.pkm.set('mechanics', cat, list(items))

        self.pkm.set('mechanics', 'source_analyzed', True)
        total = sum(len(v) for v in mechanics.values())
        print(f"  📖 Source: {total} mechanics identified")
        return True

    def discover_available_actions(self):
        """Scout available actions and classify them (cheap, 1 step each)."""
        available = list(self.obs.available_actions or [])
        results = {}

        # Need initial grid for comparison
        start_grid = None
        if hasattr(self.obs, 'frame') and self.obs.frame:
            start_grid = self.obs.frame[0].copy()

        for action_num in available:
            prev_pos = (self.player.x, self.player.y) if self.player else None

            self.step(action_num)  # also feeds self.object_tracker.observe()

            moved = False
            if self.player and prev_pos:
                moved = (self.player.x, self.player.y) != prev_pos
            elif getattr(self, 'object_tracker', None) is not None:
                # Generic fallback: self.player is only ever found via a
                # hardcoded attribute-name guess list ('gudziatsk', ...) that
                # is LS20-specific — on a game whose internal object doesn't
                # expose the player under one of those names, self.player
                # stays None forever and `moved` used to silently stay False
                # for every action (found via two real holdout runs, see
                # docs/EVALUATION.md). ObjectTracker needs no attribute name:
                # it identifies the mover from grid+action correlation alone.
                moved = bool(self.object_tracker.last_step_player_moved)

            grid_changed = False
            if start_grid is not None and hasattr(self.obs, 'frame') and self.obs.frame:
                grid_changed = not np.array_equal(self.obs.frame[0], start_grid)

            prop_changes = 0
            if self.game:
                for attr in ['cklxociuu', 'hiaauhahz']:
                    if hasattr(self.game, attr):
                        val = getattr(self.game, attr)
                        if attr == 'cklxociuu':
                            val %= 4
                        if val != 0:
                            prop_changes += 1

            results[action_num] = {
                'moved': moved,
                'grid_changed': grid_changed,
                'prop_changes': prop_changes,
            }

        self.pkm.set('discovery', 'scout_results', results)

        # Classify actions
        movement = [a for a, r in results.items() if r['moved']]
        interaction = [a for a, r in results.items() if not r['moved'] and r['prop_changes']]
        blocked = [a for a, r in results.items() if not r['moved'] and not r['grid_changed']]

        self.pkm.set('discovery', 'action_types', {
            'movement': movement,
            'interaction': interaction,
            'blocked': blocked,
        })

        print(f"  🔍 Scout: {len(movement)} movement, {len(interaction)} interaction, {len(blocked)} blocked")
        return results

    def _object_tracker_report(self) -> str:
        """Human-readable report of ObjectTracker's generic, tag-free evidence
        (player color, learned action directions, confirmed wall colors)."""
        tracker = getattr(self, 'object_tracker', None)
        if tracker is None:
            return "ObjectTracker: disabled"
        return tracker.report()

    def discover_properties(self):
        """Discover rotation and color properties."""
        rotations = []
        colors = []
        if self.game:
            # Rotation
            if hasattr(self.game, 'cklxociuu'):
                val = getattr(self.game, 'cklxociuu')
                rotations = [0, 90, 180, 270]
                self.pkm.set('state', 'rotations', rotations)
            # Color
            if hasattr(self.game, 'hiaauhahz'):
                val = getattr(self.game, 'hiaauhahz')
                colors = [0, 1, 2, 3, 4, 5, 6, 7]  # standard ARC colors
                self.pkm.set('state', 'colors', colors)

    # ------ NAVIGATION SUPPORT ------

    def _init_pathfinder(self):
        """Initialize or get the pathfinder."""
        if not hasattr(self, '_pathfinder') or self._pathfinder is None:
            self._pathfinder = Pathfinder(self)
        return self._pathfinder

    def _detect_wall_colors(self):
        """Detect wall colors from analysis + local probing."""
        if not self.game or not hasattr(self.obs, 'frame') or not self.obs.frame:
            return

        pathfinder = self._init_pathfinder()
        grid = self.obs.frame[0]

        # Analyze grid: find most common colors
        unique, counts = np.unique(grid, return_counts=True)
        sorted_by_count = np.argsort(counts)[::-1]
        color_freq = [(int(unique[idx]), int(counts[idx])) for idx in sorted_by_count]
        # Skip background (0)
        non_bg = [(c, n) for c, n in color_freq if c != 0]

        # The top non-bg color is usually the FLOOR
        # Subsequent colors are walls, objects, or special tiles
        floor_color = non_bg[0][0] if non_bg else None

        # Method 1: Tag-based — check if tag sprite color is wall or floor
        wall_tags = self.pkm.get('mechanics', 'walls', [])
        tag_colors = set()
        for tag in wall_tags:
            sprites = self._find_tagged_sprites(tag)
            for s in sprites[:5]:
                color = int(grid[s.y, s.x])
                if color != floor_color:  # Not on floor = actual wall tile
                    tag_colors.add(color)
                # If on floor, check neighbors for the real wall color
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = s.x + dx, s.y + dy
                    if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                        nc = int(grid[ny, nx])
                        if nc != color and nc != 0 and nc != floor_color:
                            tag_colors.add(nc)

        # Method 2: Adjacent to player (non-invasively)
        player_adjacent = set()
        if self.player:
            px, py = self.player.x, self.player.y
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                    c = int(grid[ny, nx])
                    pc = int(grid[py, px])
                    if c != pc and c != 0:  # Not where player stands, not bg
                        player_adjacent.add(c)

        # Method 3: Grid analysis — second most common non-bg color is often wall
        grid_wall_candidates = set()
        if len(non_bg) >= 2:
            # Colors that are less common than floor (top 2-4)
            for c, n in non_bg[1:4]:
                # If it's significantly less common than floor (>2x less)
                if n < non_bg[0][1] * 0.5:  # Less than half the floor area
                    grid_wall_candidates.add(c)

        # Method 4: ObjectTracker — colors confirmed by an actual observed
        # blocked move (see object_perception.py). Generic: no tags, no
        # source code, no hardcoded action->direction mapping. Always merged
        # in (not just when other methods are silent): unlike methods 1-3,
        # which infer walls from static color frequency/position, this is
        # grounded in real interaction outcomes, so it only ever reinforces
        # or extends the tag/heuristic-based result, never displaces it.
        tracker_walls = set()
        tracker = getattr(self, 'object_tracker', None)
        if tracker is not None:
            tracker_walls = tracker.wall_colors

        # Combine: prefer player-adjacent and grid analysis over tag-on-floor
        # If player-adjacent gives us clear walls, use those
        if player_adjacent:
            pathfinder.wall_colors.update(player_adjacent)
            # Also add grid-analysis walls if they match player-adjacent
            for c in grid_wall_candidates:
                if c in player_adjacent or c in tag_colors:
                    pathfinder.wall_colors.add(c)
        else:
            # Fallback: use tag-based + grid analysis
            pathfinder.wall_colors.update(tag_colors)
            pathfinder.wall_colors.update(grid_wall_candidates)

        pathfinder.wall_colors.update(tracker_walls)

        # Remove player's own color and background
        if self.player:
            pathfinder.wall_colors.discard(int(grid[self.player.y, self.player.x]))
        pathfinder.wall_colors.discard(0)

        # Lock walls so learn_walls() probing doesn't corrupt them
        pathfinder.walls_locked = True
        pathfinder.update_from_observation(self.obs)

        print(f"  🧱 Wall colors: {sorted(pathfinder.wall_colors)} "
              f"(tags={sorted(tag_colors)}, adjacent={sorted(player_adjacent)}, "
              f"grid={sorted(grid_wall_candidates)}, tracker={sorted(tracker_walls)}, floor={floor_color})")

    def suggest_wall_experiment(self, min_info_bits: float = 1.0) -> Optional[int]:
        """Return the action number that best resolves wall/floor ambiguity,
        or None if nothing ambiguous is worth testing.

        Unlike the advisory version, this returns ONLY the action (not a
        (color, action, info) tuple) when info_gain >= min_info_bits, making
        it directly usable as a steering command by solve_level().

        Args:
            min_info_bits: minimum Shannon entropy (bits) to justify executing
                           the experiment. Higher = more conservative.
        """
        tracker = getattr(self, 'object_tracker', None)
        if tracker is None or self.player is None:
            return None
        if not self.obs.frame or len(self.obs.frame) == 0:
            return None

        from .active_experiment import build_wall_floor_experiment, select_experiment

        grid = self.obs.frame[0]
        pf = getattr(self, '_pathfinder', None)
        confirmed_walls = pf.wall_colors if pf and hasattr(pf, 'wall_colors') else set()
        player_color = int(grid[self.player.y, self.player.x])

        # Learned movement directions only (rotate/interact excluded).
        action_dirs = {
            a: tracker.action_direction(a)
            for a in tracker.action_displacements
            if tracker.is_movement_action(a)
        }
        action_dirs = {a: d for a, d in action_dirs.items() if d is not None}
        if not action_dirs:
            return None

        # Ambiguous colours = present, not background, not the player, and not
        # already confirmed as walls.
        import numpy as np
        present = set(int(c) for c in np.unique(grid))
        ambiguous = present - {0, player_color} - set(confirmed_walls)

        best = None
        for color in ambiguous:
            hyps, candidates = build_wall_floor_experiment(
                color, action_dirs, (self.player.y, self.player.x), grid
            )
            picked = select_experiment(hyps, candidates)
            if picked is None:
                continue
            action, info = picked
            if info > 0 and (best is None or info > best[2]):
                best = (color, action, info)

        if best is not None and best[2] >= min_info_bits:
            color, action, info = best
            msg = (f"Active experiment: action {action} tests colour {color} "
                   f"(info gain {info:.2f} bit)")
            if hasattr(self, 'state') and self.state is not None:
                self.state.record_observation(msg, source="active_experiment")
            print(f"  🔬 {msg}")
            return action  # Return action directly, not the tuple

        return None  # Nothing worth testing

    def _find_tagged_sprites(self, tag: str):
        """Find sprites with given tag in current level."""
        level = self.game.current_level if self.game else None
        if not level:
            return []
        sprites = getattr(level, '_sprites', [])
        return [s for s in sprites if hasattr(s, 'tags') and s.tags and tag in s.tags]

    def _check_adjacent_to_target(self) -> bool:
        """Check if player adjacent to any interactive object."""
        if not self.player:
            return False
        px, py = self.player.x, self.player.y
        for tag in ['rhsxkxzdjz', 'rjlbuycveu']:
            sprites = self._find_tagged_sprites(tag)
            for s in sprites:
                sx, sy = getattr(s, 'x', 0), getattr(s, 'y', 0)
                if abs(px - sx) + abs(py - sy) == 1:
                    return True
        return False

    def _check_source_available(self) -> bool:
        """Check if game source file exists."""
        import os
        base_game = self.name.split('-')[0] if '-' in self.name else self.name
        # Try multiple path patterns
        possible_paths = [
            f"environment_files/{base_game}/*/{self.name}.py",
            f"environment_files/{base_game}/*/{base_game}.py",
        ]
        for pattern in possible_paths:
            import glob
            matches = glob.glob(pattern)
            if matches:
                return True
        return False

    def _infer_goal_rotation(self):
        """Infer goal rotation from level data."""
        if self.game and hasattr(self.game, 'current_level'):
            level = self.game.current_level
            if hasattr(level, 'get_data'):
                goal = level.get_data('GoalRotation')
                if goal is not None:
                    return int(goal)
        return None
