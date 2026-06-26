# Skill: detect-walls-from-source
**Type:** perception | **Preconditions:** source_available | **Effects:** wall_colors_known

## Description
Analyze game source code to identify wall sprites by their tags, then sample frame to get wall colors. Replaces heuristic wall learning.

## Algorithm
1. Call `discover_from_source()` to parse game Python file
2. Find sprites tagged with wall identifiers (e.g., 'ihdgageizm')
3. For each wall sprite, read its clone positions from source
4. Sample observation frame at those positions to get actual wall color(s)
5. Add detected colors to Pathfinder.wall_colors

## Wall Tag Mapping (from source analysis)
- `ihdgageizm` → walls/obstacles
- `rjlbuycveu` → locks (collidable=False, walkable override)
- `rhsxkxzdjz` → changers (collidable=False, walkable override)

## Parameters
- `game_source_path`: Path to game's .py file

## Returns
- Set of wall color integers

## Integration Points
- Called once per game during initialization
- Feeds: Pathfinder.wall_colors for A* navigation
- Replaces: learn_walls() heuristic probing

## Validation
- LS20: Detects color 12 as wall, color 4 as floor
- VC33: Detects rotation-only (no walls needed)