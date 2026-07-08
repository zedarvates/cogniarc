"""Simple BFS Pathfinder for ARC-AGI-3 games.

Uses grid observation colors for walkability (wall color = 3).
Works reliably for LS20 and similar grid-based games.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class GridPathfinder:
    """Grid-based BFS pathfinder for 2D grid worlds using frame colors."""
    
    grid_size: Tuple[int, int] = (64, 64)
    wall_colors: Set[int] = field(default_factory=lambda: {3})
    walkable_overrides: Set[Tuple[int, int]] = field(default_factory=set)
    grid_map: Optional[np.ndarray] = field(default_factory=lambda: None)
    _current_start: Optional[Tuple[int, int]] = None
    
    def update_from_observation(self, obs) -> None:
        """Build walkability grid from observation frame[0]."""
        h, w = self.grid_size
        self.grid_map = np.ones((h, w), dtype=bool)
        
        # Parse frame - assuming frame[0] is the grid (2D: height, width)
        frame = obs.frame[0]
        if frame.ndim == 2:
            # Single channel: colors directly at frame[y, x]
            for color in self.wall_colors:
                self.grid_map &= (frame != color)
        elif frame.ndim == 3:
            # Multi-channel: check if any channel has wall color
            for color in self.wall_colors:
                self.grid_map &= ~np.any(frame == color, axis=0)
        
        # Apply walkable overrides (e.g., target on wall)
        for (x, y) in self.walkable_overrides:
            if 0 <= x < w and 0 <= y < h:
                self.grid_map[y, x] = True
    
    def is_walkable(self, x: int, y: int) -> bool:
        """Check if cell is walkable."""
        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
            return False
        if (x, y) in self.walkable_overrides:
            return True
        if self.grid_map is None:
            return True  # No map built yet, assume walkable
        return self.grid_map[y, x]
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find shortest path using BFS. Returns list of (x, y) including start and goal."""
        sx, sy = start
        gx, gy = goal
        
        if start == goal:
            return [start]
        
        # Check if goal is walkable (or override)
        if not self.is_walkable(gx, gy) and (gx, gy) not in self.walkable_overrides:
            # Try adjacent cells
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = gx + dx, gy + dy
                if self.is_walkable(nx, ny):
                    gx, gy = nx, ny
                    break
        
        # BFS
        queue = deque([(sx, sy)])
        visited: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) == (gx, gy):
                path = []
                curr = (x, y)
                while curr is not None:
                    path.append(curr)
                    curr = visited[curr]
                return path[::-1]
            
            # 4-connected neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                nxt = (nx, ny)
                
                if nxt not in visited and self.is_walkable(nx, ny):
                    visited[nxt] = (x, y)
                    queue.append(nxt)
        
        return []
    
    def navigate_to(self, goal: Tuple[int, int], max_steps: int = 100) -> List[int]:
        """Execute navigation to goal, returning action sequence.
        
        Actions: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        """
        if self._current_start is None:
            return []
        
        path = self.find_path(self._current_start, goal)
        
        if not path:
            return []
        
        actions = []
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            if x2 > x1: actions.append(4)  # RIGHT
            elif x2 < x1: actions.append(3)  # LEFT
            elif y2 > y1: actions.append(2)  # DOWN
            elif y2 < y1: actions.append(1)  # UP
        
        return actions[:max_steps]
    
    def set_start(self, start: Tuple[int, int]) -> None:
        """Set current start position."""
        self._current_start = start
    
    def learn_walls(self, obs) -> None:
        """Learn wall colors from observation (alias for update_from_observation)."""
        self.update_from_observation(obs)


def create_pathfinder(grid_size: Tuple[int, int] = (64, 64)) -> GridPathfinder:
    return GridPathfinder(grid_size=grid_size)