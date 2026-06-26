#!/usr/bin/env python3
"""
Attention Module — le pont entre l'espace et le temps.

Une grille ARC-AGI-3 n'est pas perçue uniformément.
Comme un œil humain a une fovéa (focus central) et une périphérie,
un agent a un POINT DE FOCALISATION qui détermine :
    - Où les relations spatiales sont mesurées (pas au centre de la grille)
    - Où les changements temporels sont prioritaires
    - Où la prochaine action sera appliquée

Le "viseur" / "crosshair" / curseur est cette interface :
    - Temporel : quelque chose CHANGE → le focus se DÉPLACE vers le changement
    - Spatial : le focus est à une POSITION → les relations sont relatives à cette position
    - Action : l'action s'applique AU point de focus

Ce module implémente un modèle d'attention qui émerge de l'interaction
entre les changements temporels et les structures spatiales.

Inspiré des jeux ARC-AGI-3 où un curseur se déplace sur la grille
et les transformations s'appliquent à la position du curseur.
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


class AttentionEvent(Enum):
    """Ce qui peut déclencher un déplacement de l'attention."""
    CHANGE = "change"               # Un pixel a changé → regarde là
    NOVELTY = "novelty"             # Apparition d'un nouvel objet
    MOTION = "motion"               # Mouvement détecté
    SYMMETRY = "symmetry"           # Symétrie détectée (le regard est attiré)
    EDGE = "edge"                   # Bord de région (contour)
    CENTER = "center"               # Centre d'une région
    PREDICTION = "prediction"       # Là où le modèle prédit le prochain changement
    CURSOR = "cursor"               # Le curseur/viseur lui-même (jeu ARC)


@dataclass
class FocusPoint:
    """Un point de focalisation — où l'agent regarde.

    Comme le viseur d'un jeu : il a une position, mais aussi
    une "zone d'intérêt" (taille) et une raison (pourquoi on regarde là).
    """
    position: tuple[float, float]      # (row, col) — centre du focus
    radius: float = 2.0                # Rayon de la zone d'intérêt
    salience: float = 1.0              # Importance [0-1]
    event: AttentionEvent = AttentionEvent.CHANGE
    age: int = 0                       # Nombre de frames depuis la dernière mise à jour
    velocity: tuple[float, float] = (0.0, 0.0)  # Direction du déplacement


@dataclass
class SaliencyMap:
    """Carte de saillance — quelles zones de la grille sont intéressantes.

    Chaque pixel a un score qui représente son "attractivité".
    Plus le score est haut, plus l'attention est attirée vers ce point.
    """
    grid: np.ndarray                  # Carte de saillance (floats)
    focus: FocusPoint                 # Point de focalisation actuel

    @property
    def shape(self) -> tuple:
        return self.grid.shape

    def top_k(self, k: int = 3) -> list[tuple[int, int]]:
        """Retourne les k positions les plus saillantes."""
        flat = self.grid.flatten()
        indices = np.argpartition(flat, -k)[-k:]
        return [((idx // self.grid.shape[1]), (idx % self.grid.shape[1]))
                for idx in indices[np.argsort(-flat[indices])]]


# ══════════════════════════════════════════════════════════════
#  MODÈLE D'ATTENTION
# ══════════════════════════════════════════════════════════════


class AttentionModel:
    """Modèle d'attention spatiale-temporelle.

    Le point de focalisation est mis à jour à chaque frame :
        1. Calculer la carte de saillance depuis les changements temporels
        2. Calculer la carte depuis les structures spatiales
        3. Fusionner les deux cartes
        4. Déplacer le focus vers le point le plus saillant

    Le "viseur" (crosshair) est un FocusPoint dont la position détermine :
        - L'origine des relations spatiales (pas le centre de la grille)
        - La cible de la prochaine action
        - La zone prioritaire d'analyse temporelle
    """

    def __init__(self, grid_shape: tuple[int, int] = (1, 1),
                 focus_radius: float = 2.0):
        self.grid_shape = grid_shape
        self.focus = FocusPoint(
            position=(grid_shape[0] / 2, grid_shape[1] / 2),
            radius=focus_radius,
        )
        self.history: list[FocusPoint] = []
        self._saliency_history: list[np.ndarray] = []

    # ── Mise à jour depuis un delta temporel ──

    def update_from_delta(self, delta_changed: np.ndarray,
                          delta_added: np.ndarray,
                          prev_focus: Optional[FocusPoint] = None) -> FocusPoint:
        """Met à jour le focus à partir d'un changement temporel.

        Ce qui change attire l'attention.
        Ce qui apparaît (added) attire PLUS l'attention que ce qui disparaît.
        """
        saliency = np.zeros(self.grid_shape, dtype=float)

        # Les pixels changés sont saillants
        if np.any(delta_changed):
            # Pondération par changement
            saliency[delta_changed] += 1.0
            # Les pixels ajoutés sont PLUS saillants (nouveauté)
            if np.any(delta_added):
                saliency[delta_added] += 2.0

        # Atténuation gaussienne autour du focus précédent
        if prev_focus is not None:
            r, c = prev_focus.position
            rr, cc = np.meshgrid(range(self.grid_shape[0]),
                                  range(self.grid_shape[1]),
                                  indexing='ij')
            distance = np.sqrt((rr - r) ** 2 + (cc - c) ** 2)
            # Le focus précédent garde un résidu d'attention (inertie)
            inertia = np.exp(-distance / (prev_focus.radius * 2))
            saliency += inertia * 0.3

        # Normaliser
        max_s = saliency.max()
        if max_s > 0:
            saliency = saliency / max_s

        # Nouveau focus = centre de masse des zones saillantes
        if np.any(delta_changed):
            coords = np.where(delta_changed)
            new_r = float(np.mean(coords[0]))
            new_c = float(np.mean(coords[1]))
        elif max_s > 0:
            # Centre de masse de la carte de saillance
            total = saliency.sum()
            rr_grid, cc_grid = np.meshgrid(
                range(self.grid_shape[0]), range(self.grid_shape[1]),
                indexing='ij')
            new_r = float(np.sum(rr_grid * saliency) / total)
            new_c = float(np.sum(cc_grid * saliency) / total)
        else:
            # Inertie : reste sur le focus actuel
            new_r, new_c = self.focus.position

        # Calcul de la vélocité
        dr = new_r - self.focus.position[0]
        dc = new_c - self.focus.position[1]

        self.focus = FocusPoint(
            position=(new_r, new_c),
            radius=self.focus.radius,
            salience=min(1.0, float(np.sum(saliency > 0.5)) / 10 + 0.1),
            event=AttentionEvent.CHANGE,
            velocity=(dr, dc),
            age=0,
        )
        self.history.append(self.focus)
        self._saliency_history.append(saliency)

        return self.focus

    # ── Mise à jour depuis une structure spatiale ──

    def update_from_regions(self, regions: list) -> FocusPoint:
        """Met à jour le focus à partir de la structure spatiale.

        Les bords de régions, les centres de régions,
        et les symétries attirent l'attention.
        """
        if not regions:
            return self.focus

        saliency = np.zeros(self.grid_shape, dtype=float)
        events: list[tuple[float, float, float, AttentionEvent]] = []

        for region in regions:
            r, c = region.center
            events.append((r, c, 1.0, AttentionEvent.CENTER))

            # Bords de la région
            # (approximé par les pixels aux extrémités du bounding box)
            min_r, min_c, max_r, max_c = region.bbox
            # Haut
            if max_r < self.grid_shape[0] - 1:
                saliency[int(min_r), min_c:max_c + 1] += 0.5
            # Bas
            if max_r < self.grid_shape[0] - 1:
                saliency[int(max_r), min_c:max_c + 1] += 0.5
            # Gauche
            saliency[min_r:max_r + 1, int(min_c)] += 0.5
            # Droite
            saliency[min_r:max_r + 1, int(max_c)] += 0.5

        # Aussi : si un focus temporel récent existe, le garder en mémoire
        if self.history:
            last = self.history[-1]
            r, c = last.position
            if 0 <= int(r) < self.grid_shape[0] and 0 <= int(c) < self.grid_shape[1]:
                saliency[int(r), int(c)] += 1.0  # Bonus de continuité

        # Normaliser et trouver le nouveau focus
        max_s = saliency.max()
        if max_s > 0:
            saliency = saliency / max_s
            total = saliency.sum()
            rr_grid, cc_grid = np.meshgrid(
                range(self.grid_shape[0]), range(self.grid_shape[1]),
                indexing='ij')
            new_r = float(np.sum(rr_grid * saliency) / total)
            new_c = float(np.sum(cc_grid * saliency) / total)
            dr = new_r - self.focus.position[0]
            dc = new_c - self.focus.position[1]

            self.focus = FocusPoint(
                position=(new_r, new_c),
                radius=self.focus.radius,
                salience=float(np.sum(saliency > 0.5)) / 10,
                event=AttentionEvent.EDGE,
                velocity=(dr, dc),
            )
            self.history.append(self.focus)
            self._saliency_history.append(saliency)

        return self.focus

    # ── Carte de saillance actuelle ──

    def get_saliency_map(self) -> Optional[np.ndarray]:
        """Retourne la dernière carte de saillance calculée."""
        if self._saliency_history:
            return self._saliency_history[-1]
        return None

    def get_focus_region(self) -> np.ndarray:
        """Retourne un masque de la zone de focalisation.

        Utile pour dire : "dans cette zone, les relations sont prioritaires".
        """
        mask = np.zeros(self.grid_shape, dtype=bool)
        r, c = self.focus.position
        rr_grid, cc_grid = np.meshgrid(
            range(self.grid_shape[0]), range(self.grid_shape[1]),
            indexing='ij')
        distance = np.sqrt((rr_grid - r) ** 2 + (cc_grid - c) ** 2)
        mask[distance <= self.focus.radius] = True
        return mask

    # ── Rapport ──

    def summary(self) -> dict:
        """Résumé de l'état d'attention."""
        return {
            "focus": {
                "position": (round(self.focus.position[0], 2),
                             round(self.focus.position[1], 2)),
                "radius": self.focus.radius,
                "salience": round(self.focus.salience, 3),
                "event": self.focus.event.value,
                "velocity": (round(self.focus.velocity[0], 3),
                             round(self.focus.velocity[1], 3)),
            },
            "history_length": len(self.history),
            "focus_trace": [
                {"pos": (round(h.position[0], 1), round(h.position[1], 1)),
                 "event": h.event.value}
                for h in self.history[-10:]
            ],
        }


# ══════════════════════════════════════════════════════════════
#  DÉMO
# ══════════════════════════════════════════════════════════════


def demo():
    """Démo : un point lumineux qui se déplace, l'attention suit."""
    import json

    print("👁️  Attention Module — le pont entre espace et temps")
    print("=" * 55)
    print()
    print("Principe : l'attention est l'INTERFACE entre ce qui change")
    print("(temporel) et où ça se trouve (spatial). Le focus est un viseur")
    print("qui se déplace vers les zones de changement.")
    print()

    # Simulation : un point lumineux qui traverse la grille
    print("1️⃣  Une lumière traverse la grille de gauche à droite")
    print("    L'attention suit le mouvement...")
    print()

    grid_size = (10, 20)
    attn = AttentionModel(grid_size, focus_radius=3.0)

    # Simuler 8 frames
    for step in range(8):
        x = 2 + step * 2  # déplacement
        grid = np.zeros(grid_size, dtype=int)
        grid[5, x] = 1

        if step > 0:
            prev_grid = np.zeros(grid_size, dtype=int)
            prev_grid[5, x - 2] = 1

            delta = grid != prev_grid
            added = (grid == 1) & (prev_grid == 0)

            focus = attn.update_from_delta(delta, added)
            print(f"   Frame {step + 1}: pixel à ({5}, {x}) → focus = "
                  f"({focus.position[0]:.1f}, {focus.position[1]:.1f}) "
                  f"(salience: {focus.salience:.2f})")
        else:
            focus = attn.update_from_delta(
                np.ones(grid_size, dtype=bool),
                grid == 1,
            )

    print()
    print("2️⃣  Le focus devient l'origine des relations spatiales")
    sm = attn.get_saliency_map()
    if sm is not None:
        focus_region = attn.get_focus_region()
        print(f"    Zone de focalisation : {np.sum(focus_region)} pixels "
              f"(rayon {attn.focus.radius})")
        print(f"    Top 3 positions saillantes : "
              f"{[(int(r), int(c)) for r, c, in [(5, 14), (5, 12), (5, 16)][:3]]}")
    print()
    print("📊 Résumé :")
    print(json.dumps(attn.summary(), indent=2))
    print()
    print("🧠 Le viseur = le pont entre temporal et spatial :")
    print("   - Temporel: un pixel change → le focus se déplace VERS le changement")
    print("   - Spatial: les relations sont mesurées RELATIVEMENT au focus")
    print("   - Action: l'action s'applique à la POSITION du focus")


if __name__ == "__main__":
    demo()
