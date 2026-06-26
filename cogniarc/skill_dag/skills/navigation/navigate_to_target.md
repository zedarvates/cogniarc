# Skill: navigate-to-target
**Type:** navigation | **Preconditions:** has_player, has_pathfinder | **Effects:** player_at_target

## Description
Navigate the player to a target (x,y) position using A* pathfinding with wall avoidance. Handles dynamic wall learning and lock overrides.

## Algorithm
1. Ensure Pathfinder initialized with current observation
2. If target position is a lock (collidable=False), add to walkable_overrides
3. Call Pathfinder.find_path(start, goal, level_id)
4. Execute returned action sequence step by step
5. After each step, update Pathfinder with new observation
6. If blocked, re-plan (max 3 retries)

## Parameters
- `target_x`, `target_y`: Destination coordinates
- `level_id`: Current level for path caching
- `require_exact`: If True, fail if exact position not reached

## Returns
- `True` if target reached (or adjacent if not require_exact)
- `False` if no path found or max retries exceeded

## Integration Points
- Uses `cogniarc.pathfinding.Pathfinder` (A* + wall learning)
- Updates `Pathfinder.walkable_overrides` for locks/changers
- Caches paths per (start, goal, level_id) tuple

## Validation
- Tested on LS20 Level 1 (movement to changer)
- Tested on LS20 Level 2 (movement to lock after changer)
- Fails gracefully on rotation-only games (VC33) — returns False