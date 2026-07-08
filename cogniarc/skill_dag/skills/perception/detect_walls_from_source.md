# Skill: detect-walls-from-source
**Type:** perception | **Preconditions:** source_available, current_obs | **Effects:** walls_detected, wall_colors_known

## Description
Analyzes game source code to identify wall-colored sprites. For LS20-type games, walls are typically sprites with `collidable=True` and specific color(s) that block movement. This skill runs once per level.

## Algorithm
1. Access game object via `env.unwrapped` or internal attribute
2. Find sprite class with `collidable=True` (typically walls)
3. Extract color(s) from wall sprites
4. Populate `pathfinder.wall_colors` with detected colors
5. Call `pathfinder.update_from_observation(obs)` to rebuild grid with new wall colors

## Parameters
- `obs`: Current observation from environment
- `game`: Internal game object reference

## Returns
`True` if wall colors detected and pathfinder updated, `False` if source unavailable or no walls found

## Integration Points
```python
def _skill_detect_walls(self) -> bool:
    if self._walls_detected:
        return True  # Already done this level

    game = self._get_game_object()
    if not game:
        return False

    pathfinder = self.__init_pathfinder()

    # Detect wall colors from sprites
    self._detect_wall_colors(game, pathfinder)

    # Rebuild grid with detected wall colors
    pathfinder.update_from_observation(self.obs)
    self._walls_detected = True
    return True
```

## Validation
- Level 1: Wall colors list non-empty after execution
- Level 2: `pathfinder.grid_map` is not None (grid built)
- Level 3: Pathfinder can find path through non-wall areas