# Skill: navigate-to-target
**Type:** navigation | **Preconditions:** has_player, has_pathfinder, walls_detected, target_known | **Effects:** player_at_target

## Description
Navigates the player to target coordinates using A* pathfinding. Handles both exact target reach and adjacent walkable fallback (for lock interaction).

## Algorithm
1. Ensure pathfinder has wall colors and grid built (`update_from_observation`)
2. Add target to `walkable_overrides` if it's on a wall color (locks, changers)
3. Call `pathfinder.navigate_to(target, max_steps, require_exact)`
4. If exact goal unreachable but adjacent walkable reached → success for locks
5. Return `True` if player position == target (or adjacent for locks)

## Parameters
- `target`: (x, y) tuple of target coordinates
- `level_id`: Current level index
- `require_exact`: If True, must reach exact coordinates; if False, adjacent walkable OK
- `max_steps`: Maximum pathfinding steps (default: 100)

## Returns
`True` if target reached (exact or adjacent walkable), `False` if path blocked or max_steps exceeded

## Integration Points
```python
def _skill_navigate_to_target(self) -> bool:
    if self._phase == "navigate_to_changer":
        target = self._get_changer_position()
        require_exact = True
    elif self._phase == "navigate_to_lock":
        target = self._get_lock_position()
        require_exact = False  # Lock on wall color, adjacent is enough
    else:
        return False

    pathfinder = self.__init_pathfinder()
    pathfinder.update_from_observation(self.obs)

    # Override walkability for target on wall
    if not require_exact:
        pathfinder.walkable_overrides.add(target)

    self.navigate_to(target[0], target[1], self.current_level_idx, require_exact)

    # Check success
    if require_exact:
        return self.player.x == target[0] and self.player.y == target[1]
    else:
        # Lock collected when adjacent (stepping on it)
        return abs(self.player.x - target[0]) + abs(self.player.y - target[1]) <= 1
```

## Validation
- Level 1: Player reaches exact changer coordinates
- Level 2: Player reaches adjacent position to lock
- Level 3: Path goes through walkable tiles only (no wall crossing)