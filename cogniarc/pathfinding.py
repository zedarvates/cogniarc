#!/usr/bin/env python3
"""
A* Pathfinding for ARC-AGI-3 grid navigation.

Provides optimal path finding on grid with walls/obstacles.
Can be used by any agent needing grid navigation.
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
import heapq


@dataclass
class GridMap:
    """Walkable grid with wall detection."""
    walkable: np.ndarray  # True = can walk, False = wall/obstacle
    width: int
    height: int
    
    @classmethod
    def from_observation(cls, obs, wall_colors: Optional[Set[int]] = None, player_pos: Optional[Tuple[int, int]] = None) -> "GridMap":
        """Create grid map from observation frame."""
        if not hasattr(obs, 'frame') or obs.frame is None:
            return cls(np.ones((64, 64), dtype=bool), 64, 64)
        
        grid = obs.frame[0]
        h, w = grid.shape
        
        if wall_colors is None:
            # Heuristic: most common non-background color is likely walls
            # Background is usually 0, walls are the dominant structural color
            unique, counts = np.unique(grid, return_counts=True)
            # Sort by count descending, skip background (0)
            sorted_idx = np.argsort(counts)[::-1]
            wall_colors = set()
            for idx in sorted_idx:
                color = int(unique[idx])
                if color != 0:
                    wall_colors.add(color)
                    # Take top 2 non-background colors as potential walls
                    if len(wall_colors) >= 2:
                        break
        
        walkable = np.ones_like(grid, dtype=bool)
        for wc in wall_colors:
            walkable[grid == wc] = False
        
        # Ensure player position is always walkable
        if player_pos is not None:
            px, py = player_pos
            if 0 <= px < w and 0 <= py < h:
                walkable[py, px] = True
        
        return cls(walkable, w, h)
    
    def is_walkable(self, x: int, y: int) -> bool:
        """Check if position is walkable."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.walkable[y, x]
        return False
    
    def neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """Get walkable neighbors (4-connected)."""
        result = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if self.is_walkable(nx, ny):
                result.append((nx, ny))
        return result


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid_map: GridMap, start: Tuple[int, int], goal: Tuple[int, int], max_steps: int = 5000) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding on grid.
    
    Returns list of (x, y) positions from start to goal (inclusive), or None if no path.
    """
    if not grid_map.is_walkable(*start) or not grid_map.is_walkable(*goal):
        return None
    
    if start == goal:
        return [start]
    
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    steps = 0
    while open_set and steps < max_steps:
        steps += 1
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))
        
        for neighbor in grid_map.neighbors(*current):
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found


def astar_path_to_actions(path: Optional[List[Tuple[int, int]]]) -> List[int]:
    """Convert a path (list of positions) to action sequence (1=up, 2=down, 3=left, 4=right)."""
    if not path or len(path) < 2:
        return []
    
    actions = []
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 1:
            actions.append(4)  # right
        elif dx == -1:
            actions.append(3)  # left
        elif dy == 1:
            actions.append(2)  # down
        elif dy == -1:
            actions.append(1)  # up
        else:
            # Not adjacent - shouldn't happen
            pass
    
    return actions


class Pathfinder:
    """High-level pathfinder with caching and wall learning."""
    def __init__(self, agent=None):
        self.agent = agent
        self.grid_map: Optional[GridMap] = None
        self.wall_colors: Set[int] = set()
        self.walkable_overrides: Set[Tuple[int, int]] = set()  # positions forced walkable
        self.path_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int], int], List[int]] = {}
        self.walls_locked: bool = False  # Once true, learn_walls is no-op

    def update_from_observation(self, obs):
        """Update grid map from latest observation."""
        if not hasattr(obs, 'frame') or obs.frame is None or len(obs.frame) == 0:
            return
        if self.grid_map is None:
            self.grid_map = GridMap.from_observation(obs, self.wall_colors or None)
        else:
            # Update walkable map
            grid = obs.frame[0]
            new_walkable = np.ones_like(grid, dtype=bool)
            for wc in self.wall_colors:
                new_walkable[grid == wc] = False
            # Apply walkable overrides (e.g., locks on walls)
            for x, y in self.walkable_overrides:
                if 0 <= x < self.grid_map.width and 0 <= y < self.grid_map.height:
                    new_walkable[y, x] = True
            self.grid_map.walkable = new_walkable
    
    def learn_walls(self, obs):
        """Learn wall colors by observing failed movements.
        No-op if walls_locked=True (source-based detection already done)."""
        if self.walls_locked:
            return
        if not self.agent or not self.agent.player:
            return
        if not hasattr(obs, 'frame') or obs.frame is None or len(obs.frame) == 0:
            return
        
        # Test each direction from current position
        px, py = self.agent.player.x, self.agent.player.y
        grid = obs.frame[0]
        
        for action, (dx, dy) in [(1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0))]:
            nx, ny = px + dx, py + dy
            if 0 <= nx < 64 and 0 <= ny < 64:
                color = int(grid[ny, nx])
                # Try moving
                prev_x, prev_y = px, py
                self.agent.step(action)
                if self.agent.player.x == prev_x and self.agent.player.y == prev_y:
                    # Blocked - this color is likely a wall
                    self.wall_colors.add(color)
                # Move back if we moved
                if self.agent.player.x != prev_x or self.agent.player.y != prev_y:
                    # Find opposite action
                    opposite = {1: 2, 2: 1, 3: 4, 4: 3}
                    self.agent.step(opposite[action])
        
        # Rebuild grid map with learned walls
        self.grid_map = GridMap.from_observation(obs, self.wall_colors)
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int], level_id: int = 0) -> Optional[List[int]]:
        """Find path and return as action sequence."""
        cache_key = (start, goal, level_id)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        if self.grid_map is None:
            return None
        
        path = astar(self.grid_map, start, goal)
        if path is None:
            return None
        
        actions = astar_path_to_actions(path)
        self.path_cache[cache_key] = actions
        return actions
    
    def navigate_greedy(self, goal: Tuple[int, int], max_steps: int = 300) -> bool:
        """Navigate greedily toward goal with exploration and backtracking."""
        if not self.agent or not self.agent.player:
            return False
        
        visited = set()
        stuck_count = 0
        
        for step in range(max_steps):
            if not self.agent.player:
                return False
            px, py = self.agent.player.x, self.agent.player.y
            if (px, py) == goal:
                return True
            
            pos = (px, py)
            visited.add(pos)
            
            # Determine best direction toward goal
            dx = goal[0] - px
            dy = goal[1] - py
            
            # Build candidate directions (prioritize toward goal)
            candidates = []
            if dx > 0:
                candidates.append((4, abs(dx)))  # right
            elif dx < 0:
                candidates.append((3, abs(dx)))  # left
            if dy > 0:
                candidates.append((2, abs(dy)))  # down
            elif dy < 0:
                candidates.append((1, abs(dy)))  # up
            
            # Sort by distance (larger delta first)
            candidates.sort(key=lambda x: -x[1])
            
            # Add perpendicular directions for exploration
            if dx != 0:
                candidates.append((1, 0))  # up
                candidates.append((2, 0))  # down
            if dy != 0:
                candidates.append((3, 0))  # left
                candidates.append((4, 0))  # right
            
            moved = False
            for action, _ in candidates:
                # Skip if this would go to already visited (unless stuck)
                next_pos = self._predict_pos(px, py, action)
                if next_pos in visited and stuck_count < 3:
                    continue
                
                prev = (px, py)
                self.agent.step(action)
                if (self.agent.player.x, self.agent.player.y) != prev:
                    moved = True
                    stuck_count = 0
                    break
            
            if not moved:
                stuck_count += 1
                if stuck_count > 5:
                    # Try random exploration to escape
                    import random
                    for action in random.sample([1, 2, 3, 4], 4):
                        prev = (px, py)
                        self.agent.step(action)
                        if (self.agent.player.x, self.agent.player.y) != prev:
                            moved = True
                            stuck_count = 0
                            break
            
            if not moved:
                return False
        
        return (self.agent.player.x, self.agent.player.y) == goal

    def _predict_pos(self, x: int, y: int, action: int) -> Tuple[int, int]:
        """Predict next position from action."""
        if action == 1:
            return (x, y - 1)
        elif action == 2:
            return (x, y + 1)
        elif action == 3:
            return (x - 1, y)
        elif action == 4:
            return (x + 1, y)
        return (x, y)

    def navigate_astar(self, goal: Tuple[int, int], max_steps: int = 200, obs=None) -> bool:
        """Navigate using A* with learned wall map."""
        if not self.agent or not self.agent.player:
            return False
        
        # Learn walls from current observation
        if obs and hasattr(obs, 'frame') and obs.frame:
            self.learn_walls(obs)
            # Rebuild grid map ensuring player position is walkable
            player_pos = (self.agent.player.x, self.agent.player.y)
            self.grid_map = GridMap.from_observation(obs, self.wall_colors, player_pos)
        
        start = (self.agent.player.x, self.agent.player.y)
        actions = self.find_path(start, goal)
        
        if actions is None:
            # Fallback to greedy
            return self.navigate_greedy(goal, max_steps)
        
        for action in actions[:max_steps]:
            self.agent.step(action)
            if (self.agent.player.x, self.agent.player.y) == goal:
                return True
        
        return False

    def navigate_to(self, goal: Tuple[int, int], max_steps: int = 100) -> bool:
        """Execute path to goal."""
        if not self.agent or not self.agent.player:
            return False
        
        start = (self.agent.player.x, self.agent.player.y)
        actions = self.find_path(start, goal)
        
        if actions is None:
            return False
        
        for action in actions[:max_steps]:
            self.agent.step(action)
            if (self.agent.player.x, self.agent.player.y) == goal:
                return True
        
        return False


if __name__ == "__main__":
    # Test with simple grid
    walkable = np.ones((10, 10), dtype=bool)
    walkable[5, :] = False  # horizontal wall
    walkable[5, 5] = True   # gap
    
    grid = GridMap(walkable, 10, 10)
    path = astar(grid, (0, 0), (9, 9))
    print("Path:", path)
    actions = astar_path_to_actions(path)
    print("Actions:", actions)