"""Grid visualizer for ARC-AGI-3 debugging.
Shows the game grid as ASCII art with player, targets, and walls highlighted.
Usage: instant diagnosis before building tools.
"""

import numpy as np
from typing import Optional, List, Tuple


def visualize_grid(
    grid: np.ndarray,
    player_pos: Optional[Tuple[int, int]] = None,
    targets: Optional[List[Tuple[int, int, str]]] = None,
    wall_colors: Optional[set] = None,
    max_size: int = 40,
) -> str:
    """Render an ARC-AGI-3 grid as ASCII art with annotations.
    
    Args:
        grid: 2D numpy array (H, W) with integer color values 0-9
        player_pos: (x, y) of the player sprite
        targets: list of (x, y, label) for important positions
        wall_colors: set of color values that are walls
        max_size: maximum dimension to render (crops larger grids)
    
    Returns:
        Multi-line string with color-coded grid + legend
    """
    h, w = grid.shape
    
    # Crop if too large
    if h > max_size:
        grid = grid[:max_size, :]
        h = max_size
    if w > max_size:
        grid = grid[:, :max_size]
        w = max_size
    
    # Build target lookup
    target_map = {}
    if targets:
        for tx, ty, label in targets:
            if 0 <= ty < h and 0 <= tx < w:
                target_map[(tx, ty)] = label
    
    # Color palette (extended ASCII)
    # 0=empty, 1-9=colors mapped to characters
    color_chars = {
        0: '·',   # empty
        1: '░', 2: '▒', 3: '▓', 4: '█',
        5: '#', 6: '%', 7: '@', 8: '&', 9: '$',
    }
    # Wall indicator (overlays on color)
    WALL_INDICATOR = '▦'
    
    lines = []
    lines.append(f"Grid: {w}×{h}, {len(np.unique(grid))} colors" + 
                 (f", walls={sorted(wall_colors)}" if wall_colors else ""))
    lines.append("─" * min(w + 2, 80))
    
    # Render rows
    for y in range(h):
        row = ""
        for x in range(w):
            cell = int(grid[y, x])
            
            # Check if this is a target position
            if (x, y) in target_map:
                row += target_map[(x, y)][0]  # First char of label
            elif player_pos and x == player_pos[0] and y == player_pos[1]:
                row += '☺'  # Player
            elif wall_colors and cell in wall_colors:
                row += WALL_INDICATOR
            else:
                row += color_chars.get(cell, '?')
        
        # Add row label
        y_label = f" y={y:2d}"
        lines.append(row[:60] + y_label)
    
    lines.append("─" * min(w + 2, 80))
    
    # Legend
    lines.append("Legend: ·=empty  ░▒▓█=colors(1-4)  #%@&$=colors(5-9)  ▦=wall  ☺=player")
    if targets:
        for tx, ty, label in targets:
            if 0 <= ty < h and 0 <= tx < w:
                lines.append(f"  {label[0]} = {label} at ({tx},{ty})")
    if player_pos:
        lines.append(f"  ☺ = player at ({player_pos[0]},{player_pos[1]})")
    
    return "\n".join(lines)


def quick_diagnose(agent) -> str:
    """Quick diagnostic: show grid with all known sprites.
    
    Call this FIRST when debugging a game — before building any tools.
    
    Usage:
        from cogniarc.grid_viz import quick_diagnose
        agent = ScientistAgent('ls20-9607627b')
        print(quick_diagnose(agent))
    """
    if not agent.obs.frame or len(agent.obs.frame) == 0:
        return "No observation frame available"
    
    grid = agent.obs.frame[0]
    
    # Find player
    px = py = None
    if agent.player:
        px, py = agent.player.x, agent.player.y
    
    # Find important sprites
    targets = []
    
    # Lock
    for tag, label in [('rjlbuycveu', 'LOCK'), ('rhsxkxzdjz', 'CHANGER')]:
        sprites = agent._find_tagged_sprites(tag)
        for s in sprites[:3]:
            targets.append((getattr(s, 'x', 0), getattr(s, 'y', 0), label))
    
    # Wall colors
    wall_colors = None
    if hasattr(agent, '_pathfinder') and agent._pathfinder:
        wall_colors = getattr(agent._pathfinder, 'wall_colors', None)
    
    # Available actions
    actions = list(agent.obs.available_actions or [])
    
    viz = visualize_grid(grid, (px, py) if px is not None else None, 
                        targets, wall_colors)
    
    info = [
        viz,
        f"",
        f"Actions: {actions}",
        f"Phase: {getattr(agent, '_phase', '?')}",
        f"Steps: {agent.steps}",
        f"Stagnation: {agent.drives.stagnation_counter}",
    ]
    return "\n".join(info)
