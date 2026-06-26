# Skill: interact-with-object
**Type:** interaction | **Preconditions:** adjacent_to_target | **Effects:** object_state_changed

## Description
Interact with an adjacent object (changer, lock, etc.) by executing the interact action.

## Algorithm
1. Verify player is adjacent to target (Manhattan distance = 1)
2. Execute interact action (typically action 5)
3. Observe result: level completed, rotation changed, lock collected
4. Update PKM with interaction result

## Parameters
- `target_x`, `target_y`: Position of object to interact with
- `expected_effect`: Optional hint for validation

## Returns
- `True` if interaction caused state change
- `False` if not adjacent or no effect

## Integration Points
- Called after navigate-to-target reaches changer/lock
- Uses: PKM to record discovered mechanics
- Updates: SkillTree with composed skills (e.g., NavigateToChanger+Interact)

## Validation
- Tested on LS20 (interact with changer, interact with lock)
- Tested on VC33 (interact with changer)