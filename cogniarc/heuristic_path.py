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
        if dx > 0: toward_actions.append((4, '→'))  # RIGHT = action 4
        elif dx < 0: toward_actions.append((3, '←'))  # LEFT = action 3
        if dy > 0: toward_actions.append((2, '↓'))  # DOWN = action 2
        elif dy < 0: toward_actions.append((1, '↑'))  # UP = action 1
        
        # Check if toward-target cell (and all cells along the 5-cell jump) is walkable
        can_go_toward = False
        for action, _ in toward_actions:
            move = {1:(0,-5), 2:(0,5), 3:(-5,0), 4:(5,0)}[action]  # LS20: 5 cells per step
            # Check ALL cells along the jump, not just destination
            blocked = False
            dx, dy = move
            steps = 5  # Check 5 intermediate cells
            for i in range(1, steps + 1):
                nx = sim_x + (dx * i // steps)
                ny = sim_y + (dy * i // steps)
                if not (0 <= ny < h and 0 <= nx < w) or int(grid[ny, nx]) in wall_colors:
                    blocked = True
                    break
            if not blocked:
                nx, ny = sim_x + dx, sim_y + dy
                path.append((action, f'{_}: toward target'))
                sim_x, sim_y = nx, ny
                can_go_toward = True
                break
        
        if can_go_toward:
            continue
        
        # Toward-target blocked → go PERPENDICULAR to escape wall
        perpendicular = [3, 4] if abs(dy) > abs(dx) else [1, 2]  # left/right if vertical, up/down if horizontal
        
        found = False
        for action in perpendicular:
            move = {1:(0,-5), 2:(0,5), 3:(-5,0), 4:(5,0)}[action]
            # Check ALL cells along the jump
            blocked = False
            dx, dy = move
            for i in range(1, 6):
                nx = sim_x + (dx * i // 5)
                ny = sim_y + (dy * i // 5)
                if not (0 <= ny < h and 0 <= nx < w) or int(grid[ny, nx]) in wall_colors:
                    blocked = True
                    break
            if not blocked:
                nx, ny = sim_x + dx, sim_y + dy
                path.append((action, f'{["","→","↓","←","↑"][action]}: wall escape'))
                sim_x, sim_y = nx, ny
                found = True
                break
        
        if not found:
            break  # Trapped
    
    return path
