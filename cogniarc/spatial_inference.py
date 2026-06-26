#!/usr/bin/env python3
"""
Spatial Inference for ARC-AGI-3 — l'espace comme perception de relations.

De même que le temps n'est pas une horloge mais une perception de changements,
l'espace n'est PAS une grille de coordonnées absolues — c'est un ensemble de
RELATIONS entre objets, régions, et motifs.

Un être chimique ne perçoit pas "x=42, y=17" — il perçoit :
    "ce carré est à GAUCHE du cercle, le cercle est DANS le triangle,
     et l'ensemble forme une SYMÉTRIE axiale."

Principes :
    - Pas de mètre, pas de pixel absolu — juste des relations
    - Les objets sont des composantes connexes
    - L'espace est un graphe de relations entre objets
    - Les motifs spatiaux (symétrie, grille, rayon) sont des grammaires

Usage:
    from cogniarc.spatial_inference import (
        SpatialReasoner, Region, Relation, SpatialPattern
    )

    reasoner = SpatialReasoner(grid)
    regions = reasoner.segment()        # Trouver les objets/régions
    relations = reasoner.relate()       # Relations entre régions
    pattern = reasoner.analyze()        # Pattern spatial global
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


# ══════════════════════════════════════════════════════════════
# 1.  RÉGION — un "objet" dans la grille (composante connexe)
# ══════════════════════════════════════════════════════════════


@dataclass
class Region:
    """Une région connexe dans la grille — un "objet".

    Pas de coordonnées absolues — juste une forme, une couleur,
    un centre de masse relatif, et des relations.
    """
    id: int
    pixels: np.ndarray             # bool mask, shape de la grille
    color: int                     # valeur de la couleur (1-9)
    bbox: tuple                    # (min_r, min_c, max_r, max_c)
    center: tuple[float, float]    # centre de masse (r, c) relatif
    area: int                      # nombre de pixels
    shape: str = ""                # descriptive: "square", "L-shape", ...

    @staticmethod
    def from_mask(grid: np.ndarray, mask: np.ndarray, region_id: int, color: int) -> "Region":
        """Crée une Region à partir d'un masque booléen."""
        coords = np.where(mask)
        if len(coords[0]) == 0:
            raise ValueError("Empty mask")
        min_r, max_r = int(coords[0].min()), int(coords[0].max())
        min_c, max_c = int(coords[1].min()), int(coords[1].max())
        center_r = float(np.mean(coords[0]))
        center_c = float(np.mean(coords[1]))
        area = len(coords[0])

        # Déterminer la forme approximate
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        bbox_area = h * w
        fill_ratio = area / bbox_area if bbox_area > 0 else 0

        if fill_ratio > 0.9 and h == w:
            shape = "square"
        elif fill_ratio > 0.9 and h != w:
            shape = "rectangle"
        elif fill_ratio < 0.5 and h == w:
            shape = "L-shape"  # approximation
        elif fill_ratio < 0.7:
            shape = "hollow"
        else:
            shape = "blob"

        return Region(
            id=region_id,
            pixels=mask,
            color=color,
            bbox=(min_r, min_c, max_r, max_c),
            center=(center_r, center_c),
            area=area,
            shape=shape,
        )

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1


# ══════════════════════════════════════════════════════════════
# 2.  RELATION — comment deux régions sont liées spatialement
# ══════════════════════════════════════════════════════════════


class RelationType(Enum):
    """Types de relations spatiales entre deux régions.

    Pas de coordonnées — juste des relations qualitatives.
    """
    LEFT_OF = "left_of"             # A est à gauche de B
    RIGHT_OF = "right_of"           # A est à droite de B
    ABOVE = "above"                 # A est au-dessus de B
    BELOW = "below"                 # A est en-dessous de B
    CONTAINS = "contains"           # A contient B (B bbox dans A bbox)
    INSIDE = "inside"               # A est dans B
    TOUCHING = "touching"           # A et B se touchent (adjacence)
    SEPARATED = "separated"         # A et B sont séparés
    ALIGNED_H = "aligned_h"         # A et B sont alignés horizontalement
    ALIGNED_V = "aligned_v"         # A et B sont alignés verticalement
    SAME_SIZE = "same_size"         # A et B ont la même taille
    SAME_COLOR = "same_color"       # A et B ont la même couleur
    SAME_SHAPE = "same_shape"       # A et B ont la même forme


@dataclass
class Relation:
    """Une relation spatiale entre deux régions."""
    type: RelationType
    region_a_id: int
    region_b_id: int
    confidence: float = 1.0


# ══════════════════════════════════════════════════════════════
# 3.  PATTERN SPATIAL — l'agencement global
# ══════════════════════════════════════════════════════════════


class SpatialPatternType(Enum):
    """Types de patterns spatiaux globaux.

    Ce sont les "grammaires" de l'espace — comment les objets
    s'organisent entre eux.
    """
    SYMMETRY_H = "symmetry_h"          # Symétrie horizontale (miroir gauche-droite)
    SYMMETRY_V = "symmetry_v"          # Symétrie verticale (miroir haut-bas)
    SYMMETRY_C = "symmetry_c"          # Symétrie centrale (rotation 180°)
    GRID = "grid"                      # Disposition en grille régulière
    RAY = "ray"                        # Rayonnement depuis un centre
    CASCADE = "cascade"                # Cascade / escalier / progression
    RING = "ring"                      # Anneaux concentriques
    CHAIN = "chain"                    # Chaîne linéaire d'objets
    CLUSTER = "cluster"                # Groupe dense sans structure claire
    EMPTY = "empty"                    # Rien
    UNKNOWN = "unknown"


@dataclass
class SpatialPattern:
    """Un pattern spatial global — l'organisation des objets dans l'espace."""
    type: SpatialPatternType
    regions: list[Region] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    confidence: float = 1.0


# ══════════════════════════════════════════════════════════════
# 4.  RAISONNEUR SPATIAL
# ══════════════════════════════════════════════════════════════


class SpatialReasoner:
    """Raisonnement spatial par relations entre régions.

    Pas de coordonnées absolues — juste un graphe de relations
    entre objets détectés par connexité.
    """

    def __init__(self, grid: Optional[np.ndarray] = None):
        self.grid = grid
        self.regions: list[Region] = []
        self.relations: list[Relation] = []
        self._pattern: Optional[SpatialPattern] = None

        if grid is not None:
            self.segment()
            self.relate()

    # ── Segmentation : trouver les objets ──

    def segment(self) -> list[Region]:
        """Segmente la grille en régions connexes par couleur.

        Chaque région = un objet. Pas de seuil de taille —
        même un pixel seul est un objet.
        """
        if self.grid is None:
            return []

        self.regions = []
        visited = np.zeros(self.grid.shape, dtype=bool)
        region_id = 0

        # Couleurs présentes (sauf fond 0)
        colors = np.unique(self.grid)
        colors = colors[colors != 0]

        for color in colors:
            mask = self.grid == color
            # Trouver les composantes connexes pour cette couleur
            from scipy.ndimage import label as ndlabel
            labeled, n_features = ndlabel(mask)

            for feat_id in range(1, n_features + 1):
                feat_mask = labeled == feat_id
                if not np.any(feat_mask):
                    continue
                region = Region.from_mask(self.grid, feat_mask, region_id, int(color))
                self.regions.append(region)
                region_id += 1

        return self.regions

    # ── Relations : comment les objets sont liés ──

    def relate(self) -> list[Relation]:
        """Calcule toutes les relations entre paires de régions."""
        self.relations = []
        n = len(self.regions)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.regions[i], self.regions[j]
                self.relations.extend(self._compute_relations(a, b))

        return self.relations

    def _compute_relations(self, a: Region, b: Region) -> list[Relation]:
        """Calcule les relations entre deux régions."""
        rels: list[Relation] = []

        ar, ac = a.center
        br, bc = b.center

        # Direction (relative — pas de coordonnées absolues)
        if ac < bc - 1:
            rels.append(Relation(RelationType.LEFT_OF, a.id, b.id, confidence=0.9))
            rels.append(Relation(RelationType.RIGHT_OF, b.id, a.id, confidence=0.9))
        elif ac > bc + 1:
            rels.append(Relation(RelationType.RIGHT_OF, a.id, b.id, confidence=0.9))
            rels.append(Relation(RelationType.LEFT_OF, b.id, a.id, confidence=0.9))

        if ar < br - 1:
            rels.append(Relation(RelationType.ABOVE, a.id, b.id, confidence=0.9))
            rels.append(Relation(RelationType.BELOW, b.id, a.id, confidence=0.9))
        elif ar > br + 1:
            rels.append(Relation(RelationType.BELOW, a.id, b.id, confidence=0.9))
            rels.append(Relation(RelationType.ABOVE, b.id, a.id, confidence=0.9))

        # Containment
        a_contains_b = (a.bbox[0] <= b.bbox[0] and a.bbox[1] <= b.bbox[1] and
                        a.bbox[2] >= b.bbox[2] and a.bbox[3] >= b.bbox[3])
        b_contains_a = (b.bbox[0] <= a.bbox[0] and b.bbox[1] <= a.bbox[1] and
                        b.bbox[2] >= a.bbox[2] and b.bbox[3] >= a.bbox[3])

        if a_contains_b:
            rels.append(Relation(RelationType.CONTAINS, a.id, b.id))
            rels.append(Relation(RelationType.INSIDE, b.id, a.id))
        elif b_contains_a:
            rels.append(Relation(RelationType.CONTAINS, b.id, a.id))
            rels.append(Relation(RelationType.INSIDE, a.id, b.id))

        # Touching (adjacence de bords)
        # Vérifie si les bounding boxes ou pixels sont adjacents
        a_pad = np.pad(a.pixels, pad_width=1, mode='constant')
        b_shifted = np.pad(b.pixels, pad_width=1, mode='constant')
        # Décalage pour vérifier l'adjacence
        if np.any(a_pad[1:-1, :-2] & b_shifted[1:-1, 1:-1]) or \
           np.any(a_pad[1:-1, 2:] & b_shifted[1:-1, 1:-1]) or \
           np.any(a_pad[:-2, 1:-1] & b_shifted[1:-1, 1:-1]) or \
           np.any(a_pad[2:, 1:-1] & b_shifted[1:-1, 1:-1]):
            rels.append(Relation(RelationType.TOUCHING, a.id, b.id))
        else:
            rels.append(Relation(RelationType.SEPARATED, a.id, b.id))

        # Alignment
        if abs(ar - br) < 1.0:
            rels.append(Relation(RelationType.ALIGNED_H, a.id, b.id))
        if abs(ac - bc) < 1.0:
            rels.append(Relation(RelationType.ALIGNED_V, a.id, b.id))

        # Similarité
        if abs(a.area - b.area) / max(a.area, b.area, 1) < 0.1:
            rels.append(Relation(RelationType.SAME_SIZE, a.id, b.id))
        if a.color == b.color:
            rels.append(Relation(RelationType.SAME_COLOR, a.id, b.id))
        if a.shape == b.shape:
            rels.append(Relation(RelationType.SAME_SHAPE, a.id, b.id))

        return rels

    # ── Analyse de pattern global ──

    def analyze(self) -> SpatialPattern:
        """Analyse la disposition spatiale globale.

        Cherche des patterns dans l'organisation des régions.
        """
        if not self.regions:
            self._pattern = SpatialPattern(type=SpatialPatternType.EMPTY,
                                           confidence=1.0)
            return self._pattern

        if len(self.regions) == 1:
            self._pattern = SpatialPattern(
                type=SpatialPatternType.UNKNOWN,
                regions=self.regions, relations=self.relations,
                confidence=0.5,
            )
            return self._pattern

        # Test SYMMETRY
        sym = self._check_symmetry()
        if sym:
            self._pattern = sym
            return self._pattern

        # Test GRID
        grid_pattern = self._check_grid()
        if grid_pattern:
            self._pattern = grid_pattern
            return self._pattern

        # Test CASCADE / progression
        cascade = self._check_cascade()
        if cascade:
            self._pattern = cascade
            return self._pattern

        # Test RAY / radial
        ray = self._check_ray()
        if ray:
            self._pattern = ray
            return self._pattern

        # Test CHAIN
        chain = self._check_chain()
        if chain:
            self._pattern = chain
            return self._pattern

        # Test RING
        ring = self._check_ring()
        if ring:
            self._pattern = ring
            return self._pattern

        self._pattern = SpatialPattern(
            type=SpatialPatternType.CLUSTER,
            regions=self.regions, relations=self.relations,
            confidence=0.4,
        )
        return self._pattern

    def _check_symmetry(self) -> Optional[SpatialPattern]:
        """Vérifie si les régions forment une symétrie."""
        if len(self.regions) < 2:
            return None

        centers = np.array([r.center for r in self.regions])

        # Symétrie horizontale : les centres sont symétriques par rapport à l'axe vertical
        mean_c = np.mean([r.center[1] for r in self.regions])
        left = [r for r in self.regions if r.center[1] < mean_c]
        right = [r for r in self.regions if r.center[1] > mean_c]

        if len(left) == len(right) and len(left) > 0:
            # Vérifie que les paires sont appariées
            pairs = 0
            for lr in left:
                for rr in right:
                    if lr.color == rr.color and abs(lr.area - rr.area) / max(lr.area, 1) < 0.2:
                        pairs += 1
                        break
            if pairs == len(left):
                return SpatialPattern(
                    type=SpatialPatternType.SYMMETRY_H,
                    regions=self.regions, relations=self.relations,
                    confidence=0.85,
                )

        # Symétrie verticale : symétrique par rapport à l'axe horizontal
        mean_r = np.mean([r.center[0] for r in self.regions])
        top = [r for r in self.regions if r.center[0] < mean_r]
        bottom = [r for r in self.regions if r.center[0] > mean_r]

        if len(top) == len(bottom) and len(top) > 0:
            pairs = 0
            for tr in top:
                for br in bottom:
                    if tr.color == br.color and abs(tr.area - br.area) / max(tr.area, 1) < 0.2:
                        pairs += 1
                        break
            if pairs == len(top):
                return SpatialPattern(
                    type=SpatialPatternType.SYMMETRY_V,
                    regions=self.regions, relations=self.relations,
                    confidence=0.85,
                )

        return None

    def _check_grid(self) -> Optional[SpatialPattern]:
        """Vérifie si les régions sont disposées en grille régulière."""
        if len(self.regions) < 4:
            return None

        # Vérifie si les centres forment une grille
        centers = np.array([r.center for r in self.regions])
        rows = sorted(set(round(c[0], 1) for c in centers))
        cols = sorted(set(round(c[1], 1) for c in centers))

        if len(rows) >= 2 and len(cols) >= 2:
            if len(rows) * len(cols) >= len(self.regions) * 0.8:
                return SpatialPattern(
                    type=SpatialPatternType.GRID,
                    regions=self.regions, relations=self.relations,
                    confidence=0.7,
                )
        return None

    def _check_cascade(self) -> Optional[SpatialPattern]:
        """Vérifie si les régions forment une progression (escalier)."""
        if len(self.regions) < 3:
            return None

        centers = sorted(self.regions, key=lambda r: (r.center[0], r.center[1]))

        # Vérifie si les tailles ou positions suivent une progression
        sizes = [r.area for r in centers]
        # Croissante ou décroissante ?
        increasing = all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1))
        decreasing = all(sizes[i] > sizes[i + 1] for i in range(len(sizes) - 1))
        if increasing or decreasing:
            return SpatialPattern(
                type=SpatialPatternType.CASCADE,
                regions=self.regions, relations=self.relations,
                confidence=0.7,
            )
        return None

    def _check_ray(self) -> Optional[SpatialPattern]:
        """Vérifie si les régions rayonnent depuis un centre."""
        if len(self.regions) < 3:
            return None

        centers = np.array([r.center for r in self.regions])
        mean_center = np.mean(centers, axis=0)

        # Calcule les angles de chaque région par rapport au centre
        angles = []
        for r in self.regions:
            dr = r.center[0] - mean_center[0]
            dc = r.center[1] - mean_center[1]
            angle = np.arctan2(dr, dc)
            angles.append(angle)

        # Vérifie si les angles sont uniformément répartis
        if len(angles) >= 3:
            angles_sorted = sorted(angles)
            diffs = [angles_sorted[i] - angles_sorted[i - 1] 
                    for i in range(1, len(angles_sorted))]
            # Ajoute la différence circulaire
            diffs.append(angles_sorted[0] + 2 * np.pi - angles_sorted[-1])
            mean_diff = np.mean(diffs)
            if all(abs(d - mean_diff) < 0.5 for d in diffs):
                return SpatialPattern(
                    type=SpatialPatternType.RAY,
                    regions=self.regions, relations=self.relations,
                    confidence=0.6,
                )
        return None

    def _check_chain(self) -> Optional[SpatialPattern]:
        """Vérifie si les régions forment une chaîne linéaire."""
        if len(self.regions) < 3:
            return None

        # Compte les relations TOUCHING
        touching = [r for r in self.relations if r.type == RelationType.TOUCHING]
        if len(touching) == len(self.regions) - 1:
            return SpatialPattern(
                type=SpatialPatternType.CHAIN,
                regions=self.regions, relations=self.relations,
                confidence=0.7,
            )
        return None

    def _check_ring(self) -> Optional[SpatialPattern]:
        """Vérifie si les régions forment des anneaux concentriques."""
        if len(self.regions) < 2:
            return None

        # Vérifie containment en cascade (A contient B, B contient C, ...)
        containers = [r for r in self.relations if r.type == RelationType.CONTAINS]
        if len(containers) >= 2:
            # Vérifie si c'est une chaîne de containment
            ids_in_relation = set()
            for c in containers:
                ids_in_relation.add(c.region_a_id)
                ids_in_relation.add(c.region_b_id)
            if len(ids_in_relation) == len(containers) + 1:
                return SpatialPattern(
                    type=SpatialPatternType.RING,
                    regions=self.regions, relations=self.relations,
                    confidence=0.6,
                )
        return None

    # ══════════════════════════════════════════════════════════
    #  5.  RAPPORT
    # ══════════════════════════════════════════════════════════

    def summary(self) -> dict:
        """Résumé de l'analyse spatiale."""
        if not self._pattern:
            self.analyze()
        p = self._pattern
        if p is None:
            return {"regions": len(self.regions), "pattern": "none"}

        return {
            "regions": len(self.regions),
            "relations": len(self.relations),
            "pattern": p.type.value,
            "confidence": p.confidence,
            "region_details": [
                {"id": r.id, "color": r.color, "shape": r.shape,
                 "area": r.area, "center": (round(r.center[0], 1), round(r.center[1], 1))}
                for r in self.regions[:10]
            ],
            "relation_types": list(set(r.type.value for r in self.relations)),
        }


# ══════════════════════════════════════════════════════════════
# 6.  DÉMO
# ══════════════════════════════════════════════════════════════


def demo():
    """Démo avec des grilles synthétiques."""
    import json

    print("🧠 Spatial Inference Demo")
    print("=" * 50)
    print()
    print("Principe : l'espace n'est PAS une grille de coordonnées.")
    print("C'est un ensemble de RELATIONS entre objets/régions.")
    print()

    # 1. Symétrie horizontale
    print("1️⃣  Symétrie horizontale")
    g = np.zeros((7, 11), dtype=int)
    g[1:4, 1:3] = 1    # carré gauche
    g[1:4, 8:10] = 1   # carré droit (miroir)
    g[3:4, 4:7] = 2    # centre
    sr = SpatialReasoner(g)
    sr.segment()
    sr.relate()
    p = sr.analyze()
    print(f"   Régions: {len(sr.regions)}")
    print(f"   Pattern: {p.type.value} (confiance: {p.confidence:.0%})")
    print()

    # 2. Grille 2×2
    print("2️⃣  Grille 2×2")
    g2 = np.zeros((10, 10), dtype=int)
    g2[1:4, 1:4] = 1
    g2[1:4, 6:9] = 2
    g2[6:9, 1:4] = 3
    g2[6:9, 6:9] = 4
    sr2 = SpatialReasoner(g2)
    sr2.segment()
    p2 = sr2.analyze()
    print(f"   Pattern: {p2.type.value} (confiance: {p2.confidence:.0%})")
    print()

    # 3. Cascade (taille croissante)
    print("3️⃣  Cascade")
    g3 = np.zeros((10, 16), dtype=int)
    g3[1:3, 1:3] = 1     # 2×2
    g3[4:7, 4:8] = 2     # 3×4
    g3[8:12, 8:14] = 3   # 4×6 (partiellement hors grille)
    g3[8:10, 10:14] = 3
    sr3 = SpatialReasoner(g3)
    sr3.segment()
    p3 = sr3.analyze()
    print(f"   Pattern: {p3.type.value} (confiance: {p3.confidence:.0%})")
    print()

    # 4. Rapport
    print("📊 Summary:")
    print(json.dumps(sr3.summary(), indent=2))


if __name__ == "__main__":
    try:
        demo()
    except ImportError as e:
        print(f"⚠️  scipy.ndimage required for connected components: {e}")
        print("   Install: pip install scipy")
        print("   Running fallback demo without segmentation...")
        # Simple demo without scipy
        print("\nSpatial Inference — concepts clés :")
        print("  - Region: composante connexe (objet)")
        print("  - Relation: LEFT_OF, RIGHT_OF, ABOVE, BELOW, CONTAINS, ...")
        print("  - Pattern: SYMMETRY_H, SYMMETRY_V, GRID, CASCADE, RAY, CHAIN, RING")
