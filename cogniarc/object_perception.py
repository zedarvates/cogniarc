"""Generic object-role inference from grid + action observations.

Answers "which region is the player, what does each action do, and which
colors are walls?" purely by correlating grid changes with the actions taken
— no game-specific sprite tags, no source-code reading, and no hardcoded
action->direction mapping (the LS20 mapping documented in the README is
itself "non-standard": action 1 isn't always UP in every game, so even that
assumption would be a game-specific hardcode, not a generalizable rule).

This is the generic counterpart to scientist_agent_discovery.py's tag-based
discovery (_find_tagged_sprites('rhsxkxzdjz'), ...), which only works on a
game whose leaked source happens to be readable and whose internal attribute
names happen to be known in advance (true only for LS20 in this repo today).
ObjectTracker works on any game that renders a grid and reports which action
was taken, which is the actual generalization test ARC-AGI poses.

Method:
  1. Segment consecutive frames into connected-component regions (reuses
     SpatialReasoner's segmentation).
  2. Match each region in the "before" frame to its likely counterpart in the
     "after" frame (same color, nearest center, similar area) — a lightweight
     single-step tracker, not full multi-object tracking.
  3. The color whose region moves most consistently across many actions is
     the best player candidate (`player_color`).
  4. The average displacement vector observed for each action number is that
     action's *learned* direction (`action_direction`) — empirically derived,
     not assumed.
  5. When a movement-classified action produces zero displacement for the
     player-candidate region, the color of the grid cell immediately adjacent
     to it in the (already-learned) expected direction is recorded as wall
     evidence — grounded in an observed failed move, not a static
     color-frequency guess.

Intentionally NOT wired to override scientist_agent_discovery.py's tag-based
path (which stays authoritative for LS20, where source is available and the
behavior is verified). See docs/EVALUATION.md for when/how this should start
steering decisions for a genuinely new game.
"""
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .spatial_inference import Region, SpatialReasoner


def segment_regions(grid: np.ndarray) -> List[Region]:
    """Segment a grid into connected-component regions (thin wrapper).

    Avoids SpatialReasoner's constructor auto-computing relate() (O(n^2) over
    region pairs), which ObjectTracker doesn't need.
    """
    sr = SpatialReasoner()
    sr.grid = grid
    return sr.segment()


def _best_match(region: Region, candidates: List[Region], max_area_ratio: float = 1.5) -> Optional[Region]:
    """Find the most likely counterpart of `region` among `candidates`:
    same color, closest center, similar area. None if no same-color candidate.
    """
    same_color = [c for c in candidates if c.color == region.color]
    if not same_color:
        return None

    def score(c: Region) -> float:
        dr = c.center[0] - region.center[0]
        dc = c.center[1] - region.center[1]
        dist = (dr * dr + dc * dc) ** 0.5
        area_ratio = max(c.area, region.area) / max(min(c.area, region.area), 1)
        area_penalty = 0.0 if area_ratio <= max_area_ratio else (area_ratio - max_area_ratio)
        return dist + area_penalty

    return min(same_color, key=score)


class ObjectTracker:
    """Learns player identity, per-action direction, and wall colors from
    observed (grid_before, action, grid_after) transitions."""

    def __init__(self, move_threshold: float = 0.5, min_wall_votes: int = 2):
        self.move_threshold = move_threshold
        self.min_wall_votes = min_wall_votes

        self.color_move_count: Counter = Counter()
        self.color_static_count: Counter = Counter()
        self.action_displacements: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.wall_color_votes: Counter = Counter()
        self.n_observations = 0
        # Set by the most recent observe() call: True if the player-candidate
        # region moved that step, False if it's known but didn't move, None if
        # no player candidate is established yet. This is what lets callers
        # detect "did the last action move the player?" WITHOUT any attribute
        # name — the generic replacement for `self.player.x/.y` probing, which
        # silently reports no-movement-ever on any game whose internal player
        # object isn't named one of a hardcoded guess list (see
        # scientist_agent.py's ['gudziatsk', 'player', 'agent', ...] probe,
        # found broken on two real holdout games in docs/EVALUATION.md).
        self.last_step_player_moved: Optional[bool] = None

    @property
    def player_color(self) -> Optional[int]:
        """Best player candidate: the color whose region moves most often.

        Ties broken by requiring at least one observed move — a color that
        never moves is never a player candidate, however common it is.
        """
        movers = {c: n for c, n in self.color_move_count.items() if n > 0}
        if not movers:
            return None
        return max(movers, key=movers.get)

    def action_direction(self, action: int) -> Optional[Tuple[float, float]]:
        """Average (d_row, d_col) displacement observed for this action.

        None if the action has never been observed to move anything.
        """
        obs = self.action_displacements.get(action)
        if not obs:
            return None
        dr = sum(o[0] for o in obs) / len(obs)
        dc = sum(o[1] for o in obs) / len(obs)
        return (dr, dc)

    def is_movement_action(self, action: int) -> bool:
        """Whether this action reliably displaces the player-candidate region."""
        direction = self.action_direction(action)
        if direction is None:
            return False
        magnitude = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
        return magnitude >= self.move_threshold

    @property
    def wall_colors(self) -> set:
        """Colors confirmed (by repeated blocked-move observation) to be walls."""
        return {c for c, n in self.wall_color_votes.items() if n >= self.min_wall_votes}

    def observe(self, grid_before: np.ndarray, action: int, grid_after: np.ndarray) -> None:
        """Update player/action/wall evidence from one transition."""
        regions_before = segment_regions(grid_before)
        regions_after = segment_regions(grid_after)

        moved_this_step: Dict[int, Tuple[float, float]] = {}  # color -> displacement

        for rb in regions_before:
            ra = _best_match(rb, regions_after)
            if ra is None:
                continue  # region vanished (destroyed/consumed) — not evidence either way
            dr = ra.center[0] - rb.center[0]
            dc = ra.center[1] - rb.center[1]
            magnitude = (dr * dr + dc * dc) ** 0.5
            if magnitude >= self.move_threshold:
                self.color_move_count[rb.color] += 1
                moved_this_step[rb.color] = (dr, dc)
            else:
                self.color_static_count[rb.color] += 1

        pc = self.player_color
        if pc is not None and pc in moved_this_step:
            self.action_displacements[action].append(moved_this_step[pc])
            self.last_step_player_moved = True
        elif pc is not None:
            # Player-candidate region didn't move this step. If this action is
            # already confidently known to be a movement action, the cell it
            # was attempting to move into is wall evidence.
            self._record_wall_evidence_if_blocked(grid_before, regions_before, pc, action)
            self.last_step_player_moved = False
        else:
            self.last_step_player_moved = None  # no player candidate yet

        self.n_observations += 1

    def current_position(self, grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """Best-guess (row, col) of the player in `grid`, using the learned
        player_color — no attribute name, no tag, works on any game once
        enough moves have been observed to identify a mover.

        Returns None if no player candidate has been established yet, or if
        the player's color isn't present in this particular grid.
        """
        pc = self.player_color
        if pc is None:
            return None
        regions = [r for r in segment_regions(grid) if r.color == pc]
        if not regions:
            return None
        region = max(regions, key=lambda r: r.area)
        return (int(round(region.center[0])), int(round(region.center[1])))

    def _record_wall_evidence_if_blocked(
        self, grid_before: np.ndarray, regions_before: List[Region], player_color: int, action: int
    ) -> None:
        direction = self.action_direction(action)
        if direction is None:
            return  # action's direction not yet established — no bootstrap assumption

        player_regions = [r for r in regions_before if r.color == player_color]
        if not player_regions:
            return
        # Assume the largest same-color region is the player instance (handles
        # stray same-color pixels elsewhere on the grid).
        player_region = max(player_regions, key=lambda r: r.area)

        dr, dc = direction
        # Normalize to a unit step so we probe the immediately adjacent cell.
        step_r = 1 if dr > 0.5 else (-1 if dr < -0.5 else 0)
        step_c = 1 if dc > 0.5 else (-1 if dc < -0.5 else 0)
        if step_r == 0 and step_c == 0:
            return

        r = int(round(player_region.center[0])) + step_r
        c = int(round(player_region.center[1])) + step_c
        if 0 <= r < grid_before.shape[0] and 0 <= c < grid_before.shape[1]:
            blocked_color = int(grid_before[r, c])
            if blocked_color != player_color:
                self.wall_color_votes[blocked_color] += 1

    def get_perception_summary(self, grid: Optional[np.ndarray] = None) -> dict:
        """Return a structured dict consumable by the phase machine.

        Keys:
          - player_color: int or None
          - action_directions: {action_num: (dr, dc)} for movement actions
            (row/col convention: dr = delta row, dc = delta col)
          - wall_colors: set[int] — colours with >= min_wall_votes
          - n_observations: int
          - player_moved_last_step: bool or None
          - player_position: (x, y) or None       [only when `grid` is given]
          - known_positions: {color: (x, y)}      [only when `grid` is given]

        Positional keys use the (x=col, y=row) convention because every
        consumer in scientist_agent_skills.py (hypothesis formation, sprite
        tag targets via .x/.y, _ot_target) already speaks (x, y). Internally
        current_position()/segmentation are (row, col); the conversion
        happens here, at the boundary, so callers never have to flip.

        Bug context: hypothesis formation read `player_position` and
        `known_positions` from this summary since 2026-07-02, but this method
        never returned either key — so target discovery silently yielded
        nothing on every holdout game (2026-07-05 report, bug B2's root:
        agents random-explored forever because no target could ever be found).
        """
        action_dirs = {
            a: self.action_direction(a)
            for a in self.action_displacements
            if self.action_direction(a) is not None
        }
        summary = {
            "player_color": self.player_color,
            "action_directions": action_dirs,
            "wall_colors": set(self.wall_colors),
            "n_observations": self.n_observations,
            "player_moved_last_step": self.last_step_player_moved,
        }

        if grid is not None:
            pos_rc = self.current_position(grid)
            summary["player_position"] = (pos_rc[1], pos_rc[0]) if pos_rc else None

            known: Dict[int, Tuple[int, int]] = {}
            for region in segment_regions(grid):
                color = region.color
                if color == 0:
                    continue
                r, c = int(round(region.center[0])), int(round(region.center[1]))
                # Keep the largest region per colour as its representative.
                if color not in known or region.area > known[color][2]:
                    known[color] = (c, r, region.area)
            summary["known_positions"] = {col: (x, y) for col, (x, y, _a) in known.items()}

        return summary

    def has_enough_observations(self, min_player: int = 3, min_directions: int = 1) -> bool:
        """Return True if enough data has been collected to be useful.

        Args:
            min_player: minimum observations before player_color is trusted.
            min_directions: minimum known action directions.
        """
        if self.player_color is None:
            return False
        known_dirs = sum(1 for a in self.action_displacements if self.action_direction(a) is not None)
        return (
            self.n_observations >= min_player
            and known_dirs >= min_directions
            and self.color_move_count[self.player_color] >= 1
        )

    def report(self) -> str:
        pc = self.player_color
        return (
            f"ObjectTracker: {self.n_observations} observations, "
            f"player_color={pc}, wall_colors={sorted(self.wall_colors)}, "
            f"actions_seen={sorted(self.action_displacements.keys())}"
        )


def best_action_toward(
    action_directions: Dict[int, Tuple[float, float]],
    current_rc: Tuple[int, int],
    target_xy: Tuple[int, int],
) -> Optional[int]:
    """Pick the learned action whose direction best reduces the distance to
    the target. Pure function (unit-testable without a live game) — the
    planning half of generic target navigation; the executing half lives in
    scientist_agent_skills._skill_navigate_to_target.

    Args:
        action_directions: {action: (dr, dc)} as learned by ObjectTracker
            (row/col deltas, from get_perception_summary()["action_directions"]).
        current_rc: player position as (row, col) — ObjectTracker convention
            (current_position()).
        target_xy: target as (x, y) = (col, row) — the skills-file convention
            used by sprite tags and known_positions.

    Returns the action with the highest positive dot-product between its
    learned direction and the needed displacement, or None if no action makes
    progress (all dot products <= 0, or nothing learned yet).
    """
    if not action_directions:
        return None
    cur_r, cur_c = current_rc
    need_dr = target_xy[1] - cur_r   # rows to travel
    need_dc = target_xy[0] - cur_c   # cols to travel
    if need_dr == 0 and need_dc == 0:
        return None  # already there — nothing to do

    best, best_score = None, 0.0
    for action, (dr, dc) in sorted(action_directions.items()):
        score = dr * need_dr + dc * need_dc
        if score > best_score:
            best, best_score = action, score
    return best
