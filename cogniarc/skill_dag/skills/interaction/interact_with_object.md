# Skill: interact-with-object
**Type:** interaction | **Preconditions:** adjacent_to_target, target_is_interactive | **Effects:** level_completed

## Description
Executes interaction action (ACTION5) on an adjacent interactive object (lock, button, etc.). For LS20, walking onto the lock (require_exact=False) collects it automatically, but some games need explicit ACTION5.

## Algorithm
1. Verify player is adjacent to interactive target
2. Execute ACTION5 (interaction action)
3. Check if `levels_completed` increased
4. If not, try additional interaction attempts

## Parameters
- `target_pos`: (x, y) of interactive object
- `action_id`: Interaction action (default: 5 for ACTION5)

## Returns
`True` if `levels_completed` increased, `False` if no progress

## Integration Points
```python
def _skill_interact(self) -> bool:
    prev_levels = self.obs.levels_completed

    # Try interaction action
    self.obs = self.env.step(getattr(GameAction, "ACTION5"))
    self.steps += 1

    # Check for level completion
    if self.obs.levels_completed > prev_levels:
        return True

    # For LS20: lock is collected by walking onto it (handled in navigate_to_lock)
    # If we were already adjacent, try moving onto it
    if abs(self.player.x - self.last_target[0]) + abs(self.player.y - self.last_target[1]) == 1:
        # Move onto target
        dx = self.last_target[0] - self.player.x
        dy = self.last_target[1] - self.player.y

        if dx == 1: act = 4  # right
        elif dx == -1: act = 3  # left
        elif dy == 1: act = 2  # down
        elif dy == -1: act = 1  # up
        else: return False

        self.obs = self.env.step(getattr(GameAction, f"ACTION{act}"))
        self.steps += 1
        return self.obs.levels_completed > prev_levels

    return False
```

## Validation
- Level 1: Level completed after interaction
- Level 2: No crash if interaction action unavailable
- Level 3: Handles both ACTION5 and walk-onto patterns