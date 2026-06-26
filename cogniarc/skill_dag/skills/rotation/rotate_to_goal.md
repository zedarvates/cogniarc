# Skill: rotate-to-goal
**Type:** rotation | **Preconditions:** has_changer, knows_goal_rotation | **Effects:** rotation_matches_goal

## Description
Rotate player to match goal rotation by navigating to a changer and cycling rotation action.

## Algorithm
1. Use navigate-to-target to reach changer position
2. While current_rotation != goal_rotation:
   - Execute rotation action (typically action 6)
   - Update current_rotation from observation
3. Verify rotation matches goal

## Parameters
- `goal_rotation`: Target rotation value (0-3)
- `changer_position`: (x,y) of rotation changer (from perception)

## Returns
- `True` if rotation matches goal
- `False` if changer unreachable or rotation action unavailable

## Integration Points
- Depends on: navigate-to-target (to reach changer)
- Uses: GoalInferenceEngine to read goal_rotation from level data
- Uses: PKM to store discovered changer positions

## Validation
- Tested on LS20 Level 1 (rot 0→3)
- Tested on LS20 Level 2 (rot 0→3)
- Works on VC33 (rotation-only game)