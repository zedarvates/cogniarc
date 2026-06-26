# Skill: select-skill-for-observation
**Type:** meta | **Preconditions:** current_obs, skill_dag_loaded | **Effects:** skill_selected

## Description
Meta-skill that builds context from current observation and selects the next skill to execute via SkillNavigator.

## Algorithm
1. Build context dict from observation:
   - has_player: player object exists
   - has_pathfinder: Pathfinder initialized
   - has_changer: changer sprites detected
   - knows_goal_rotation: goal_rotation inferred
   - adjacent_to_target: player next to interacted object
   - available_actions: from obs.available_actions
   - source_available: game source file exists
2. Call SkillNavigator.select_skills(context)
3. Return highest-priority skill (by type order: perception → navigation → rotation → interaction → meta)

## Returns
- Selected skill_id, or None if no skill applicable

## Integration Points
- Called each decision cycle in ScientistAgent.solve_level()
- Uses: SkillNavigator, SkillRegistry
- Updates: PKM with selected skill for traceability

## Validation
- On LS20 Level 1 start: selects detect-walls-from-source → navigate-to-target → rotate-to-goal → interact-with-object
- On VC33 start: selects detect-walls-from-source → rotate-to-goal