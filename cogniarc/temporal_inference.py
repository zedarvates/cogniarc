#!/usr/bin/env python3
"""
Temporal Inference for ARC-AGI-3 — le temps comme perception de changements.

Le temps n'est PAS une mesure absolue. C'est une perception de changements.
Un être chimique (humain) ne sent pas le "temps qui passe" — il sent des
changements d'état : faim → repas → satiété. Pas d'horloge interne.

Cette approche traite le temps comme des DELTAS entre états observés et des
MÉTACHANGEMENTS (patterns de changements de changements).

Principe physique :
    - La relativité générale dit que le temps est local, lié à la masse/gravité
    - Un LLM / agent n'a PAS d'horloge — il a des séquences d'observations
    - Le "temps" n'est qu'une abstraction qui émerge de la comparaison d'états

Pour ARC-AGI-3 :
    Les grilles sont des snapshots. Entre deux snapshots, il y a un DELTA.
    La séquence des deltas (Δ1, Δ2, Δ3, ...) constitue le "temps" de la grille.
    Prédire la frame N+1 = appliquer le pattern de deltas à la frame N.

Usage:
    from cogniarc.temporal_inference import (
        Delta, DeltaPattern, TemporalReasoner,
        PatternType
    )

    reasoner = TemporalReasoner(frames=[grid1, grid2, grid3])
    pattern = reasoner.analyze()          # Quel pattern de changement ?
    next_grid = reasoner.predict()        # Prédire frame 4
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import defaultdict


# ══════════════════════════════════════════════════════════════
# 1.  DELTA — la différence entre deux états
# ══════════════════════════════════════════════════════════════


@dataclass
class Delta:
    """La différence entre deux grilles consécutives.

    Pas de notion de "temps écoulé" — juste ce qui a CHANGÉ.

    Attributs :
        added:    positions de pixels qui apparaissent (n'étaient pas là avant)
        removed:  positions de pixels qui disparaissent
        changed:  positions de pixels qui changent de couleur
        colors_added:    set des nouvelles couleurs
        colors_removed:  set des couleurs perdues
        count_changed:   nombre total de pixels modifiés
        magnitude:       proportion de la grille qui a changé [0, 1]
    """
    added: np.ndarray          # bool mask
    removed: np.ndarray        # bool mask
    changed: np.ndarray        # bool mask (added XOR removed XOR color changes)
    colors_added: set          # int set
    colors_removed: set        # int set
    count_changed: int
    magnitude: float

    @staticmethod
    def compute(before: np.ndarray, after: np.ndarray) -> "Delta":
        """Compute the delta between two grid states.

        Compare directement les grilles — pas de temps, juste des états.
        """
        if before.shape != after.shape:
            raise ValueError(f"Grid shapes differ: {before.shape} vs {after.shape}")

        changed = before != after
        count = int(np.sum(changed))
        total = before.size
        magnitude = count / total if total > 0 else 0.0

        # Pixels qui apparaissent (étaient bg/0, deviennent quelque chose)
        added = changed & (after != 0) & (before == 0)

        # Pixels qui disparaissent (étaient quelque chose, deviennent bg/0)
        removed = changed & (before != 0) & (after == 0)

        # Changements de couleur (les deux non-nuls mais différents)
        color_changed = changed & (before != 0) & (after != 0)

        colors_before = set(np.unique(before[changed]))
        colors_after = set(np.unique(after[changed]))
        colors_added = colors_after - colors_before
        colors_removed = colors_before - colors_after

        return Delta(
            added=added, removed=removed,
            changed=changed,
            colors_added=colors_added, colors_removed=colors_removed,
            count_changed=count, magnitude=magnitude,
        )

    def to_dict(self) -> dict:
        return {
            "count_changed": self.count_changed,
            "magnitude": round(self.magnitude, 4),
            "colors_added": list(self.colors_added),
            "colors_removed": list(self.colors_removed),
        }


# ══════════════════════════════════════════════════════════════
# 2.  PATTERN DE DELTAS — le "métachangement"
# ══════════════════════════════════════════════════════════════


class PatternType(Enum):
    """Types de patterns temporels — comment les changements changent eux-mêmes.

    Chaque pattern est une relation entre deltas consécutifs.
    Pas d'horloge — juste des relations.
    """
    # Le même changement se répète à l'identique
    CONSTANT = "constant"           # Δ1 ≈ Δ2 ≈ Δ3
    # Le changement s'accélère (de plus en plus de pixels changent)
    ACCELERATING = "accelerating"   # |Δ1| < |Δ2| < |Δ3|
    # Le changement décélère (de moins en moins de pixels changent)
    DECELERATING = "decelerating"   # |Δ1| > |Δ2| > |Δ3|
    # Le changement oscille (revient vers un état précédent)
    OSCILLATING = "oscillating"     # Δ1 ≈ -Δ2 ≈ Δ3
    # Le changement se déplace spatialement (une vague qui se propage)
    WAVE = "wave"                   # pattern de positions qui se déplace
    # Le changement s'inverse (rotation, symétrie du delta)
    INVERSION = "inversion"         # Δ2 = transform(Δ1)
    # État stable — plus rien ne change
    STASIS = "stasis"               # |Δ| → 0
    # Pattern non reconnu (trop peu de frames, ou bruit)
    UNKNOWN = "unknown"


@dataclass
class DeltaPattern:
    """Un pattern de changements — la structure du temps perçu.

    C'est ça le "temps" : pas une durée, mais la STRUCTURE des changements.
    """
    type: PatternType
    deltas: list[Delta]
    confidence: float = 1.0          # [0, 1] — à quel point on est sûr
    period: Optional[int] = None     # Pour les patterns oscillants : période
    direction: Optional[np.ndarray] = None  # Pour les vagues : vecteur direction


# ══════════════════════════════════════════════════════════════
# 3.  RAISONNEUR TEMPOREL
# ══════════════════════════════════════════════════════════════


class TemporalReasoner:
    """Raisonnement temporel par analyse de deltas entre états.

    Ne stocke PAS le temps. Stocke des séquences d'états et les relations
    entre les différences de ces états.

    Le "temps" n'est qu'une abstraction émergente : la différence entre
    deux observations consécutives crée un delta. La différence entre
    deltas crée un métachangement. C'est ça, la perception du temps.
    """

    def __init__(self, frames: Optional[list[np.ndarray]] = None):
        self.frames: list[np.ndarray] = frames or []
        self.deltas: list[Delta] = []
        self._pattern: Optional[DeltaPattern] = None

        if len(self.frames) >= 2:
            self._compute_deltas()

    def add_frame(self, frame: np.ndarray) -> None:
        """Ajoute une frame et recalcule les deltas."""
        if self.frames:
            delta = Delta.compute(self.frames[-1], frame)
            self.deltas.append(delta)
        self.frames.append(frame)

    def _compute_deltas(self) -> None:
        """Calcule les deltas entre toutes les frames consécutives."""
        self.deltas = []
        for i in range(1, len(self.frames)):
            delta = Delta.compute(self.frames[i - 1], self.frames[i])
            self.deltas.append(delta)

    def analyze(self) -> DeltaPattern:
        """Analyse les deltas pour trouver le pattern temporel.

        C'est ici qu'on découvre comment le changement change.
        """
        if len(self.deltas) < 1:
            return DeltaPattern(type=PatternType.UNKNOWN, deltas=[],
                                confidence=0.0)

        if len(self.deltas) == 1:
            # Juste un changement — on ne peut pas encore détecter un pattern
            return DeltaPattern(type=PatternType.UNKNOWN, deltas=self.deltas,
                                confidence=0.3)

        magnitudes = [d.magnitude for d in self.deltas]

        # Test STASIS : plus rien ne change
        if len(self.deltas) >= 2 and all(m < 0.01 for m in magnitudes[-2:]):
            self._pattern = DeltaPattern(
                type=PatternType.STASIS, deltas=self.deltas, confidence=0.95)
            return self._pattern

        # Test OSCILLATING : les changements s'inversent (CHECK FIRST — important)
        if len(self.deltas) >= 3:
            osc = self._check_oscillation()
            if osc:
                self._pattern = osc
                return self._pattern

        # Test CONSTANT : les changements ont la même ampleur
        if len(self.deltas) >= 2:
            diffs = [abs(m - magnitudes[0]) for m in magnitudes]
            if all(d < 0.05 for d in diffs):
                self._pattern = DeltaPattern(
                    type=PatternType.CONSTANT, deltas=self.deltas, confidence=0.9)
                return self._pattern

        # Test ACCELERATING / DECELERATING
        if len(magnitudes) >= 2:
            d1, d2 = magnitudes[0], magnitudes[-1]
            if d2 > d1 * 1.3:
                self._pattern = DeltaPattern(
                    type=PatternType.ACCELERATING, deltas=self.deltas, confidence=0.7)
                return self._pattern
            if d1 > d2 * 1.3:
                self._pattern = DeltaPattern(
                    type=PatternType.DECELERATING, deltas=self.deltas, confidence=0.7)
                return self._pattern

        # Test WAVE : le changement se déplace spatialement
        if len(self.deltas) >= 2:
            wave = self._check_wave()
            if wave:
                return wave

        # Test INVERSION : le delta subit une transformation
        if len(self.deltas) >= 2:
            inv = self._check_inversion()
            if inv:
                return inv

        self._pattern = DeltaPattern(
            type=PatternType.UNKNOWN, deltas=self.deltas,
            confidence=0.3,
        )
        return self._pattern

    def _check_constant_position(self) -> bool:
        """Vérifie si les pixels qui changent sont aux mêmes positions."""
        if len(self.deltas) < 2:
            return False
        ref = self.deltas[0].changed
        for d in self.deltas[1:]:
            overlap = np.sum(ref & d.changed)
            total = np.sum(ref | d.changed)
            if total > 0 and overlap / total < 0.8:
                return False
        return True

    def _check_oscillation(self) -> Optional[DeltaPattern]:
        """Vérifie si les changements oscillent entre deux états.

        Un vrai pattern oscillant a des deltas qui s'INVERSENT :
        ce qui est ajouté à T1 est retiré à T2, et vice-versa.
        Simple magnitude matching ne suffit PAS.
        """
        if len(self.deltas) < 3:
            return None
        d1, d2, d3 = self.deltas[-3:]

        # Vérifie que Δ1 et Δ3 sont dans la même direction
        # et que Δ2 est dans la direction opposée
        # En regardant les pixels ajoutés vs retirés
        added_overlap_1_2 = np.sum(d1.added & d2.removed)
        removed_overlap_1_2 = np.sum(d1.removed & d2.added)
        total_change_1 = np.sum(d1.changed)

        if total_change_1 == 0:
            return None

        # Si les pixels ajoutés en Δ1 sont retirés en Δ2 (et vice versa) → oscillation
        inversion_ratio = (added_overlap_1_2 + removed_overlap_1_2) / max(total_change_1, 1)
        if inversion_ratio > 0.5:
            return DeltaPattern(
                type=PatternType.OSCILLATING, deltas=self.deltas,
                confidence=min(0.95, inversion_ratio),
                period=2,
            )

        # Fallback sur la magnitude si on a 4+ deltas
        if len(self.deltas) >= 4:
            m1, m2, m3, m4 = (self.deltas[-4].magnitude, d1.magnitude,
                              d2.magnitude, d3.magnitude)
            if abs(m1 - m3) < 0.05 and abs(m2 - m4) < 0.05 and abs(m1 - m2) > 0.2:
                return DeltaPattern(
                    type=PatternType.OSCILLATING, deltas=self.deltas,
                    confidence=0.7, period=2,
                )

        return None

    def _check_wave(self) -> Optional[DeltaPattern]:
        """Vérifie si le changement se déplace comme une vague.

        Compare le centre de masse des pixels changés entre deltas.
        """
        centers = []
        for d in self.deltas[-5:]:
            if np.sum(d.changed) > 0:
                ys, xs = np.where(d.changed)
                centers.append((np.mean(xs), np.mean(ys)))
            else:
                centers.append(None)

        # Si on a au moins 2 centres, regarde s'ils bougent linéairement
        valid = [c for c in centers if c is not None]
        if len(valid) >= 3:
            xs = np.array([c[0] for c in valid])
            ys = np.array([c[1] for c in valid])
            # Test si les centres sont alignés (vague directionnelle)
            if np.std(xs) > 2 or np.std(ys) > 2:
                dx = xs[-1] - xs[0]
                dy = ys[-1] - ys[0]
                direction = np.array([dx, dy])
                norm = np.linalg.norm(direction)
                if norm > 0:
                    return DeltaPattern(
                        type=PatternType.WAVE, deltas=self.deltas,
                        confidence=0.7,
                        direction=direction / norm,
                        period=int(np.ceil(norm)) if norm > 0 else None,
                    )
        return None

    def _check_inversion(self) -> Optional[DeltaPattern]:
        """Vérifie si un delta est la transformation d'un delta précédent."""
        # Compare les magnitudes et la distribution spatiale
        d1_mag = self.deltas[-2].magnitude
        d2_mag = self.deltas[-1].magnitude
        if abs(d1_mag - d2_mag) < 0.05:
            # Même magnitude — vérifie la complémentarité spatiale
            added = self.deltas[-2].added
            removed = self.deltas[-1].removed
            overlap = np.sum(added & removed)
            if overlap > 0:
                return DeltaPattern(
                    type=PatternType.INVERSION, deltas=self.deltas,
                    confidence=0.65,
                )
        return None

    # ══════════════════════════════════════════════════════════
    #  4.  PRÉDICTION — le futur comme extension du pattern
    # ══════════════════════════════════════════════════════════

    def predict(self) -> Optional[np.ndarray]:
        """Prédit la prochaine frame à partir du pattern temporel."""
        if len(self.frames) < 1:
            return None

        if self._pattern is None:
            self.analyze()
        pattern = self._pattern  # guaranteed non-None after analyze()
        if pattern is None:
            return self.frames[-1].copy()

        last_frame = self.frames[-1].copy()

        if pattern.type == PatternType.CONSTANT:
            return self._predict_constant(last_frame)
        elif pattern.type == PatternType.ACCELERATING:
            return self._predict_accelerating(last_frame)
        elif pattern.type == PatternType.DECELERATING:
            return self._predict_decelerating(last_frame)
        elif pattern.type == PatternType.OSCILLATING:
            return self._predict_oscillating(last_frame)
        elif pattern.type == PatternType.WAVE:
            return self._predict_wave(last_frame)
        elif pattern.type == PatternType.STASIS:
            return last_frame.copy()
        else:
            return self._predict_from_last_delta(last_frame)

    def _predict_constant(self, frame: np.ndarray) -> np.ndarray:
        """Répète le dernier delta — le même changement continue."""
        return self._apply_delta(frame, self.deltas[-1])

    def _predict_accelerating(self, frame: np.ndarray) -> np.ndarray:
        """Amplifie le delta — le changement s'accélère."""
        if len(self.deltas) < 2:
            return self._predict_constant(frame)
        # Extrapole la tendance des magnitudes
        mags = [d.magnitude for d in self.deltas[-3:]]
        if len(mags) >= 2:
            ratio = mags[-1] / max(mags[-2], 0.001)
            scaled = self._scale_delta(self.deltas[-1], ratio)
            return self._apply_delta(frame, scaled)
        return self._predict_constant(frame)

    def _predict_decelerating(self, frame: np.ndarray) -> np.ndarray:
        """Atténue le delta — le changement ralentit."""
        if len(self.deltas) < 2:
            return self._predict_constant(frame)
        mags = [d.magnitude for d in self.deltas[-3:]]
        if len(mags) >= 2:
            ratio = mags[-1] / max(mags[-2], 0.001)
            scaled = self._scale_delta(self.deltas[-1], max(ratio, 0.5))
            return self._apply_delta(frame, scaled)
        # Si ça décélère vers zéro, prédire stasis
        if self.deltas[-1].magnitude < 0.05:
            return frame.copy()
        return self._predict_constant(frame)

    def _predict_oscillating(self, frame: np.ndarray) -> np.ndarray:
        """Inverse le dernier delta — retour vers l'état précédent.

        Comme un pendule : avance, recule, avance, recule...
        """
        if len(self.deltas) >= 2:
            # Revient vers l'état d'il y a 2 deltas
            delta_back = Delta.compute(self.frames[-1], self.frames[-2])
            return self._apply_delta(frame, delta_back)
        return self._predict_constant(frame)

    def _predict_wave(self, frame: np.ndarray) -> np.ndarray:
        """Déplace le delta dans la direction de la vague."""
        if self._pattern.direction is not None:
            shift_y = int(round(self._pattern.direction[1]))
            shift_x = int(round(self._pattern.direction[0]))
            shifted = np.roll(self.deltas[-1].changed,
                              shift=shift_y, axis=0)
            shifted = np.roll(shifted, shift=shift_x, axis=1)
            # Applique le changement décalé
            result = frame.copy()
            result[shifted] = (result[shifted] + 1) % 10
            return result
        return self._predict_constant(frame)

    def _predict_from_last_delta(self, frame: np.ndarray) -> np.ndarray:
        """Fallback : répète simplement le dernier changement."""
        if self.deltas:
            return self._apply_delta(frame, self.deltas[-1])
        return frame.copy()

    # ══════════════════════════════════════════════════════════
    #  5.  FONCTIONS OUTILS
    # ══════════════════════════════════════════════════════════

    @staticmethod
    def _apply_delta(frame: np.ndarray, delta: Delta) -> np.ndarray:
        """Applique un delta à une frame pour produire la frame suivante.

        Les pixels 'added' deviennent actifs, 'removed' s'effacent.
        """
        result = frame.copy()
        result[delta.added] = 1    # Nouveaux pixels
        result[delta.removed] = 0  # Pixels supprimés
        return result

    @staticmethod
    def _scale_delta(delta: Delta, ratio: float) -> Delta:
        """Crée un delta artificiel avec une magnitude différente.

        Pour simuler l'accélération/décélération.
        """
        if ratio <= 0:
            return Delta(
                added=np.zeros_like(delta.added, dtype=bool),
                removed=np.zeros_like(delta.removed, dtype=bool),
                changed=np.zeros_like(delta.changed, dtype=bool),
                colors_added=set(), colors_removed=set(),
                count_changed=0, magnitude=0.0,
            )

        # Prend un sous-échantillon des pixels changés
        indices = np.where(delta.changed)
        n_to_take = max(1, int(len(indices[0]) * min(ratio, 3.0)))
        n_to_take = min(n_to_take, len(indices[0]))
        take = np.random.choice(len(indices[0]), n_to_take, replace=False)

        added = np.zeros_like(delta.added, dtype=bool)
        for idx in take:
            added[indices[0][idx], indices[1][idx]] = delta.added[indices[0][idx], indices[1][idx]]

        return Delta.compute(
            np.zeros_like(delta.changed, dtype=int),
            added.astype(int),
        )

    # ══════════════════════════════════════════════════════════
    #  6.  ÉTAT
    # ══════════════════════════════════════════════════════════

    def summary(self) -> dict:
        """Résumé de l'analyse temporelle."""
        if not self._pattern:
            self.analyze()
        p = self._pattern
        if p is None:
            return {"frames": len(self.frames), "deltas": len(self.deltas),
                    "pattern": "none", "confidence": 0.0}
        return {
            "frames": len(self.frames),
            "deltas": len(self.deltas),
            "pattern": p.type.value,
            "confidence": p.confidence,
            "period": p.period,
            "magnitudes": [round(d.magnitude, 4) for d in self.deltas],
            "colors_added": [list(d.colors_added) for d in self.deltas],
            "colors_removed": [list(d.colors_removed) for d in self.deltas],
        }


# ══════════════════════════════════════════════════════════════
# 7.  TEST / DÉMO
# ══════════════════════════════════════════════════════════════


def demo():
    """Petite démo avec des grilles synthétiques."""
    import json

    print("🧠 Temporal Inference Demo")
    print("=" * 50)
    print()
    print("Principe : le temps n'est PAS une mesure absolue.")
    print("C'est la perception de CHANGEMENTS entre états.")
    print("Le delta est la différence entre deux observations.")
    print("Le métachangement est la différence entre deltas.")
    print()

    # Crée 4 frames d'un objet qui se déplace vers la droite
    print("1️⃣  Pattern CONSTANT — objet qui se déplace pas à pas")
    frames = []
    for x in range(4):
        g = np.zeros((5, 5), dtype=int)
        g[2, 1 + x] = 1
        frames.append(g)

    r = TemporalReasoner(frames)
    p = r.analyze()
    print(f"   Pattern détecté : {p.type.value} (confiance: {p.confidence:.0%})")
    next_grid = r.predict()
    print(f"   Prédiction frame 5 : centre en x={np.where(next_grid[2]==1)[0][0]}")
    print()

    # Crée des frames accélérantes
    print("2️⃣  Pattern ACCELERATING — le mouvement s'accélère")
    frames2 = []
    for i, speed in enumerate([1, 2, 3]):
        g = np.zeros((5, 8), dtype=int)
        pos = sum([1, 2, 3][:i]) if i > 0 else 0
        g[2, pos % 8] = 1
        frames2.append(g)

    # Mieux : la taille du changement augmente
    frames3 = []
    for size in [1, 3, 5]:
        g = np.zeros((7, 7), dtype=int)
        g[3 - size//2:3 + size//2 + 1, 3 - size//2:3 + size//2 + 1] = 1
        frames3.append(g)

    r3 = TemporalReasoner(frames3)
    p3 = r3.analyze()
    print(f"   Pattern détecté : {p3.type.value} (confiance: {p3.confidence:.0%})")
    print(f"   Magnitudes : {[round(d.magnitude, 4) for d in r3.deltas]}")
    next_g = r3.predict()
    print(f"   Prédiction frame 4 : {np.sum(next_g > 0)} pixels actifs")
    print()

    # Pattern oscillant
    print("3️⃣  Pattern OSCILLATING — le changement s'inverse")
    frames4 = [
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, 0]]),
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 1], [0, 0]]),
    ]
    r4 = TemporalReasoner(frames4)
    p4 = r4.analyze()
    print(f"   Pattern détecté : {p4.type.value} (confiance: {p4.confidence:.0%})")
    next_g4 = r4.predict()
    print(f"   Prédiction frame 5 : coin supérieur gauche = {next_g4[0,0]}")
    print()

    # Rapport complet
    print("📊 Résumé du dernier test :")
    print(json.dumps(r4.summary(), indent=2))


if __name__ == "__main__":
    demo()
