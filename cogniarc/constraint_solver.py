#!/usr/bin/env python3
"""
Constraint Solver pour CogniArc — OR-Tools CP-SAT pour grilles ARC-AGI.

Résout des problèmes de contrainte spatiale:
  - Coloring: assigner des couleurs aux cellules sous contraintes
  - Routing: trouver un chemin entre deux points dans une grille
  - Transformation: encoder les règles de transformation

Usage:
    from cogniarc.constraint_solver import ARCConstraintSolver
    solver = ARCConstraintSolver()
    result = solver.solve("coloring", {"colors": 3}, input_grid)
"""

from __future__ import annotations

import numpy as np

try:
    from ortools.sat.python import cp_model
    _ORTOOLS_AVAILABLE = True
except ImportError:
    _ORTOOLS_AVAILABLE = False

from typing import Optional


class ARCConstraintSolver:
    """Solveur de contraintes spatiales pour grilles ARC-AGI.

    Types supportés:
      - coloring: colorier des régions sous contraintes d'adjacence
      - routing: trouver un chemin entre deux points
      - transformation: appliquer des règles de transformation logique
    """

    def __init__(self):
        self.last_solution = None

    def solve(
        self,
        problem_type: str,
        constraints: dict,
        input_grid: Optional[np.ndarray] = None,
    ) -> dict:
        """Résout un problème de contrainte.

        Args:
            problem_type: 'coloring', 'routing', 'transformation'
            constraints: dict de contraintes
                coloring: {colors: int, adjacency: bool, regions: list}
                routing: {start: (r,c), end: (r,c), obstacles: list[(r,c)]}
                transformation: {rule: str, params: dict}
            input_grid: grille d'entrée (numpy 2D, valeurs 0-9)

        Returns:
            dict avec solution ou message d'erreur
        """
        if not _ORTOOLS_AVAILABLE:
            return self._fallback(problem_type, constraints, input_grid)

        solvers = {
            "coloring": self._solve_coloring,
            "routing": self._solve_routing,
            "transformation": self._solve_transformation,
        }

        solver_fn = solvers.get(problem_type)
        if not solver_fn:
            return {"error": f"Type inconnu: {problem_type}", "solved": False}

        try:
            return solver_fn(constraints, input_grid)
        except Exception as e:
            return {"error": str(e), "solved": False}

    def _solve_coloring(self, constraints: dict, grid: Optional[np.ndarray]) -> dict:
        """Colorie une grille sous contraintes d'adjacence."""
        if grid is None:
            h, w = constraints.get("shape", (5, 5))
            grid = np.zeros((h, w), dtype=int)

        h, w = grid.shape
        num_colors = constraints.get("colors", 4)
        adjacency = constraints.get("adjacency", True)
        regions = constraints.get("regions", None)

        model = cp_model.CpModel()

        # Variables: cell[r][c] = couleur (0..num_colors-1)
        cell = {}
        for r in range(h):
            for c in range(w):
                cell[(r, c)] = model.NewIntVar(0, num_colors - 1, f"cell_{r}_{c}")

        # Contrainte: cellules adjacentes ≠ même couleur (si adjacency=True)
        if adjacency:
            for r in range(h):
                for c in range(w):
                    for dr, dc in [(0, 1), (1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            model.Add(cell[(r, c)] != cell[(nr, nc)])

        # Contrainte: régions pré-colorées
        if regions:
            for region in regions:
                for (r, c), color in region.items():
                    model.Add(cell[(r, c)] == color)

        # Objectif: minimiser les conflits d'adjacence (0 = solution parfaite)
        # Pas d'objective complexe — juste trouver une solution valide
        model.Maximize(1)  # Trouver n'importe quelle solution valide

        # Résoudre
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 2
        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return {"error": "Pas de solution", "solved": False}

        result_grid = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                result_grid[r, c] = solver.Value(cell[(r, c)])

        self.last_solution = result_grid
        return {
            "solved": True,
            "grid": result_grid,
            "colors_used": len(set(result_grid.flatten())),
        }

    def _solve_routing(self, constraints: dict, grid: Optional[np.ndarray]) -> dict:
        """Trouve un chemin entre deux points."""
        start = tuple(constraints.get("start", (0, 0)))
        end = tuple(constraints.get("end", (4, 4)))
        obstacles = set(tuple(o) for o in constraints.get("obstacles", []))

        h = constraints.get("height", grid.shape[0] if grid is not None else 5)
        w = constraints.get("width", grid.shape[1] if grid is not None else 5)
        max_steps = constraints.get("max_steps", h * w)

        # BFS simple (OR-Tools overkill pour du routing simple)
        # On utilise BFS pour la vitesse, CP-SAT pour les contraintes complexes
        from collections import deque

        q = deque([(start[0], start[1], 0, [])])
        visited = {start}

        while q:
            r, c, dist, path = q.popleft()
            new_path = path + [(r, c)]

            if (r, c) == end:
                return {
                    "solved": True,
                    "path": new_path,
                    "length": dist,
                    "steps": len(new_path),
                }

            if dist >= max_steps:
                continue

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if (nr, nc) not in visited and (nr, nc) not in obstacles:
                        visited.add((nr, nc))
                        q.append((nr, nc, dist + 1, new_path))

        return {"error": "Aucun chemin trouvé", "solved": False}

    def _solve_transformation(self, constraints: dict, grid: Optional[np.ndarray]) -> dict:
        """Applique une transformation logique à la grille."""
        rule = constraints.get("rule", "identity")
        params = constraints.get("params", {})

        if grid is None:
            return {"error": "Grille d'entrée requise", "solved": False}

        if rule == "rotate":
            k = params.get("k", 1)
            result = np.rot90(grid, k=k)
        elif rule == "flip":
            axis = params.get("axis", 0)
            result = np.flip(grid, axis=axis)
        elif rule == "translate":
            dr, dc = params.get("dr", 0), params.get("dc", 0)
            result = np.roll(grid, (dr, dc), axis=(0, 1))
        elif rule == "invert":
            max_color = constraints.get("max_color", 9)
            result = max_color - grid
        elif rule == "replace":
            mapping = params.get("mapping", {})
            result = grid.copy()
            for old, new in mapping.items():
                result[grid == int(old)] = int(new)
        else:
            return {"error": f"Règle inconnue: {rule}", "solved": False}

        return {
            "solved": True,
            "grid": result,
            "rule": rule,
        }

    def detect_problem_type(self, grid: np.ndarray, target: Optional[np.ndarray] = None) -> str:
        """Détecte automatiquement le type de problème.

        Returns: 'coloring', 'routing', 'transformation'
        """
        if target is not None:
            # Si les deux grilles ont la même forme → transformation
            if grid.shape == target.shape:
                return "transformation"
            return "routing"

        # Grille seule → coloring (pattern d'adjacence)
        unique = np.unique(grid)
        if len(unique) <= 2:
            return "routing"  # Binaire → probablement un chemin/labyrinthe
        return "coloring"

    def _fallback(self, problem_type: str, constraints: dict, grid: Optional[np.ndarray]) -> dict:
        """Fallback textuel quand OR-Tools pas dispo."""
        return {
            "solved": False,
            "fallback": True,
            "message": f"OR-Tools non disponible. Problème: {problem_type}",
            "constraints": constraints,
            "grid_shape": grid.shape if grid is not None else None,
        }


if __name__ == "__main__":
    solver = ARCConstraintSolver()
    print(f"OR-Tools available: {_ORTOOLS_AVAILABLE}")

    # Test coloring
    print("\n=== Coloring (5x5, 4 couleurs, adjacency) ===")
    result = solver.solve("coloring", {"colors": 4, "shape": (5, 5)})
    if result.get("solved"):
        g = result["grid"]
        print(f"   Couleurs utilisées: {result['colors_used']}")
        for row in g:
            print(f"   {row}")
    else:
        print(f"   ❌ {result.get('error', 'échec')}")

    # Test routing
    print("\n=== Routing (5x5, start→end) ===")
    result = solver.solve("routing", {
        "start": (0, 0), "end": (4, 4),
        "obstacles": [(1, 1), (2, 2), (3, 3)],
    })
    if result.get("solved"):
        print(f"   Chemin trouvé: {result['length']} étapes, {result['steps']} pas")
        # Afficher la grille
        grid = np.full((5, 5), 0)
        for r, c in result["path"]:
            grid[r, c] = 1
        for row in grid:
            print(f"   {row}")
    else:
        print(f"   ❌ {result.get('error', 'échec')}")

    # Test transformation
    print("\n=== Transformation (rotate) ===")
    grid = np.array([[1, 2], [3, 4]], dtype=int)
    result = solver.solve("transformation", {"rule": "rotate", "params": {"k": 1}}, grid)
    if result.get("solved"):
        print(f"   Règle: {result['rule']}")
        print(f"   Input:  {grid}")
        print(f"   Output: {result['grid']}")

    # Test fallback
    print("\n=== Fallback (sans OR-Tools) ===")
    _ORTOOLS_AVAILABLE_backup = _ORTOOLS_AVAILABLE
    import cogniarc.constraint_solver as cs
    cs._ORTOOLS_AVAILABLE = False
    fallback = solver.solve("coloring", {"colors": 4}, np.zeros((3, 3)))
    print(f"   Fallback: {fallback.get('fallback', False)}")
    print(f"   Message: {fallback.get('message', '')}")
    cs._ORTOOLS_AVAILABLE = _ORTOOLS_AVAILABLE_backup
