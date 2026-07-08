# Skill: select-skill-for-observation
**Type:** meta | **Preconditions:** current_obs | **Effects:** skill_selected

## Description
Meta-skill that observes current game state and selects the appropriate next skill to execute. Enables dynamic skill selection based on runtime conditions rather than fixed phase machine.

## Algorithm
1. Observe available actions, level completion, player position
2. Check for rotation changer presence and goal rotation
3. Check for lock/interactable objects
4. Check if walls detected
5. Return skill_id that should execute next

## Parameters
- `obs`: Current observation
- `game_state`: Internal game object reference

## Returns
Skill ID string from: ["detect-walls-from-source", "navigate-to-target", "rotate-to-goal", "interact-with-object"]

## Integration Points
```python
def _skill_select(self, context: SkillContext) -> str:
    # Phase 1: Detect walls if not done
    if not context.has_precondition("walls_detected"):
        return "detect-walls-from-source"

    # Phase 2: Check for changer + goal rotation
    if context.has_precondition("has_changer") and context.has_precondition("knows_goal_rotation"):
        if not context.has_precondition("rotation_goal_reached"):
            if not context.has_precondition("adjacent_to_changer"):
                return "navigate-to-target"  # navigate_to_changer
            else:
                return "rotate-to-goal"

    # Phase 3: Check for lock/interaction
    if context.has_precondition("adjacent_to_target") and context.has_precondition("target_is_interactive"):
        return "interact-with-object"

    # Phase 4: Navigate to lock
    if context.has_precondition("has_changer") and context.has_precondition("rotation_goal_reached"):
        if not context.has_precondition("level_completed"):
            return "navigate-to-target"  # navigate_to_lock

    # Default: explore
    return "navigate-to-target"
```

## Validation
- Level 1: Returns detect-walls from cold start
- Level 2: Returns navigate-to-target when walls known, changer exists
- Level 3: Returns rotate-to-goal when at changer with known goal
- Level 4: Returns interact when adjacent to interactive target