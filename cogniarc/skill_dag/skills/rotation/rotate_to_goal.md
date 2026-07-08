# Skill: rotate-to-goal
**Type:** rotation | **Preconditions:** has_changer, knows_goal_rotation, adjacent_to_changer | **Effects:** rotation_goal_reached

## Description
Rotates the player to the goal rotation by navigating to the rotation changer and cycling through rotation states. For LS20, ACTION6 enters changer, then ACTION3 cycles rotation.

## Algorithm
1. Verify current rotation != goal rotation
2. Ensure adjacent to changer (handled by navigate-to-target phase)
3. Execute ACTION6 to enter changer
4. Cycle ACTION3 + ACTION6 until rotation matches goal
5. Verify rotation achieved

## Parameters
- `goal_rotation`: Target rotation angle (0, 90, 180, 270)
- `changer_pos`: (x, y) of rotation changer sprite

## Returns
`True` if current rotation == goal rotation, `False` if changer not accessible or cycle failed

## Integration Points
```python
def _skill_rotate_to_goal(self) -> bool:
    game = self._get_game_object()
    if not game:
        return False

    goal_rot = self._infer_goal_rotation()
    if goal_rot is None or game.cklxociuu == goal_rot:
        return True  # Already at goal rotation

    # Must be at changer position
    changer = self._find_tagged_sprites('rhsxkxzdjz')
    if not changer:
        return False

    ch = changer[0]
    if abs(self.player.x - ch.x) + abs(self.player.y - ch.y) > 1:
        return False  # Not at changer

    # Enter changer (ACTION6)
    self.obs = self.env.step(getattr(GameAction, "ACTION6"))
    self.steps += 1

    # Cycle rotation (ACTION3+6) until goal
    while game.cklxociuu != goal_rot:
        self.obs = self.env.step(getattr(GameAction, "ACTION3"))
        self.steps += 1
        self.obs = self.env.step(getattr(GameAction, "ACTION6"))
        self.steps += 1

        if self.obs.levels_completed > self.current_level_idx:
            return True  # Level completed during rotation!

    return True
```

## Validation
- Level 1: Rotation reaches goal rotation (e.g., 0° → 270°)
- Level 2: No infinite loop if rotation already correct
- Level 3: Handles level completion during rotation cycle