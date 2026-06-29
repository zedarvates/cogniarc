"""Simple heuristic pathfinder for LS20 — solves the level immediately.
When nano-LLM oscillates, this takes over with deterministic wall-circumvention.

Rule: go perpendicular to wall until toward-target cell is clear, then go toward target.
"""

def heuristic_navigate(grid, px, py, tx, ty, wall_colors, max_steps=50):
    """Simple heuristic: circumvent walls by going perpendicular.
    
    Returns list of (action, reason) tuples.
    """
    path = []
    sim_x, sim_y = px, py
    h, w = grid.shape
    
    for _ in range(max_steps):
        if (sim_x, sim_y) == (tx, ty):
            break
        
        # Determine toward-target direction
        dx = tx - sim_x
        dy = ty - sim_y
        
        # Try toward-target first
        toward_actions = []
        if dx > 0: toward_actions.append((1, '→'))
        elif dx < 0: toward_actions.append((3, '←'))
        if dy > 0: toward_actions.append((2, '↓'))
        elif dy < 0: toward_actions.append((4, '↑'))
        
        # Check if toward-target cell is walkable
        can_go_toward = False
        for action, _ in toward_actions:
            move = {1:(1,0), 2:(0,1), 3:(-1,0), 4:(0,-1)}[action]
            nx, ny = sim_x + move[0], sim_y + move[1]
            if 0 <= ny < h and 0 <= nx < w and int(grid[ny, nx]) not in wall_colors:
                path.append((action, f'{_}: toward target'))
                sim_x, sim_y = nx, ny
                can_go_toward = True
                break
        
        if can_go_toward:
            continue
        
        # Toward-target blocked → go PERPENDICULAR to escape wall
        perpendicular = [1, 3] if abs(dy) > abs(dx) else [2, 4]  # left/right if vertical
        
        found = False
        for action in perpendicular:
            move = {1:(1,0), 2:(0,1), 3:(-1,0), 4:(0,-1)}[action]
            nx, ny = sim_x + move[0], sim_y + move[1]
            if 0 <= ny < h and 0 <= nx < w and int(grid[ny, nx]) not in wall_colors:
                path.append((action, f'{["","→","↓","←","↑"][action]}: wall escape'))
                sim_x, sim_y = nx, ny
                found = True
                break
        
        if not found:
            break  # Trapped
    
    return path
