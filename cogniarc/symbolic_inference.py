#!/usr/bin/env python3
"""
Symbolic Inference — le pont entre la perception (patterns) et l'action (skills).

Les patterns temporels et spatiaux sont des PERCEPTIONS.
Les symboles sont ce qui transforme ces perceptions en INSTRUCTIONS.

Une symétrie horizontale + un pattern constant = symbole "mirror_h"
Une rotation + un pattern accélérant = symbole "rotate_ccw"
Un containment + oscillation = symbole "toggle_inside_outside"

Le module symbolique :
    1. Reçoit les sorties de TemporalReasoner, SpatialReasoner, AttentionModel
    2. Les combine en DESCRIPTEURS SYMBOLIQUES
    3. Mappe chaque descripteur vers un SKILL du SkillDAG
    4. Retourne une liste de skills candidats, ordonnés par pertinence

Cette couche est ce qui manque entre "je perçois" et "j'agis".
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

# Ces imports fonctionneront quand les modules seront dans le même package
try:
    from .temporal_inference import PatternType, TemporalReasoner
    from .spatial_inference import SpatialPatternType, SpatialReasoner
    from .attention import AttentionModel, AttentionEvent
except ImportError:
    # Fallback pour développement/test standalone
    from temporal_inference import PatternType, TemporalReasoner
    from spatial_inference import SpatialPatternType, SpatialReasoner
    from attention import AttentionModel, AttentionEvent


# ══════════════════════════════════════════════════════════════
#  1.  DESCRIPTEURS SYMBOLIQUES
# ══════════════════════════════════════════════════════════════


class SymbolType(Enum):
    """Types de symboles — ce que l'agent RECONNAÎT dans la scène.

    Chaque symbole est une combinaison d'un pattern temporel ET spatial.
    """
    # ── Mouvements ──
    TRANSLATE_LEFT = "translate_left"
    TRANSLATE_RIGHT = "translate_right"
    TRANSLATE_UP = "translate_up"
    TRANSLATE_DOWN = "translate_down"
    # ── Rotations ──
    ROTATE_90_CW = "rotate_90_cw"
    ROTATE_90_CCW = "rotate_90_ccw"
    ROTATE_180 = "rotate_180"
    # ── Symétries ──
    MIRROR_H = "mirror_horizontal"
    MIRROR_V = "mirror_vertical"
    # ── Changements de couleur ──
    COLOR_SWAP = "color_swap"
    COLOR_SHIFT = "color_shift"          # +1 sur toutes les couleurs
    COLOR_INVERT = "color_invert"        # 9 - couleur
    # ── Taille ──
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    # ── Logique ──
    TOGGLE = "toggle"                    # Apparaît/disparaît cycliquement
    FILL = "fill"                        # Remplir une région
    EMPTY = "empty"                      # Vider une région
    # ── Relations ──
    INSIDE_OUT = "inside_out"            # Inversion containment
    ALIGN = "align"                      # Aligner deux régions
    # ── Méta ──
    COPY = "copy"                        # Copier d'une région à l'autre
    NO_OP = "no_op"                      # Rien (stable)


@dataclass
class Symbol:
    """Un symbole = une interprétation de ce qui se passe.

    Attributes:
        type: Le type de symbole (ex: TRANSLATE_RIGHT)
        confidence: [0-1] à quel point on est sûr
        source_pattern: Pattern temporel qui a déclenché ça
        source_spatial: Pattern spatial associé
        focus_position: Où le focus était quand ce symbole a été inféré
        params: Paramètres supplémentaires (ex: couleur, distance, angle)
    """
    type: SymbolType
    confidence: float = 0.5
    source_pattern: Optional[PatternType] = None
    source_spatial: Optional[SpatialPatternType] = None
    source_attention: Optional[AttentionEvent] = None
    focus_position: tuple[float, float] = (0.0, 0.0)
    params: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
#  2.  INFERENCE SYMBOLIQUE
# ══════════════════════════════════════════════════════════════


class SymbolicInference:
    """Infère des symboles à partir des perceptions temporelles et spatiales.

    Pipeline :
        1. Recevoir les sorties de Temporal, Spatial, Attention
        2. Croiser les patterns pour générer des symboles
        3. Classer les symboles par confiance
        4. Mapper vers les skills du SkillDAG
    """

    def __init__(self):
        self.symbols: list[Symbol] = []
        self._skill_map = self._build_skill_map()

    def _build_skill_map(self) -> dict[SymbolType, str]:
        """Map les symboles vers les IDs de skills du SkillDAG manifest."""
        return {
            SymbolType.TRANSLATE_LEFT: "navigate-to-target",
            SymbolType.TRANSLATE_RIGHT: "navigate-to-target",
            SymbolType.TRANSLATE_UP: "navigate-to-target",
            SymbolType.TRANSLATE_DOWN: "navigate-to-target",
            SymbolType.ROTATE_90_CW: "rotate-to-goal",
            SymbolType.ROTATE_90_CCW: "rotate-to-goal",
            SymbolType.ROTATE_180: "rotate-to-goal",
            SymbolType.MIRROR_H: "rotate-to-goal",
            SymbolType.MIRROR_V: "rotate-to-goal",
            SymbolType.COLOR_SWAP: "interact-with-object",
            SymbolType.COLOR_SHIFT: "interact-with-object",
            SymbolType.COLOR_INVERT: "interact-with-object",
            SymbolType.SCALE_UP: "interact-with-object",
            SymbolType.SCALE_DOWN: "interact-with-object",
            SymbolType.TOGGLE: "interact-with-object",
            SymbolType.FILL: "interact-with-object",
            SymbolType.EMPTY: "interact-with-object",
            SymbolType.INSIDE_OUT: "interact-with-object",
            SymbolType.ALIGN: "navigate-to-target",
            SymbolType.COPY: "interact-with-object",
            SymbolType.NO_OP: "select-skill-for-observation",
        }

    def infer(self,
              temporal_pattern: Optional[tuple[PatternType, float]] = None,
              spatial_pattern: Optional[tuple[SpatialPatternType, float]] = None,
              attention_event: Optional[AttentionEvent] = None,
              focus_position: tuple[float, float] = (0.0, 0.0),
              deltas: Optional[list] = None,
              regions: Optional[list] = None,
              ) -> list[Symbol]:
        """Infère les symboles à partir des patterns perçus.

        Args:
            temporal_pattern: (PatternType, confidence) du module temporel
            spatial_pattern: (SpatialPatternType, confidence) du module spatial
            attention_event: Dernier événement d'attention
            focus_position: Position actuelle du focus
            deltas: Liste des deltas temporels (pour analyser les couleurs)
            regions: Liste des régions spatiales (pour analyser les relations)

        Returns:
            Liste de symboles triés par confiance décroissante
        """
        self.symbols = []

        t_type, t_conf = temporal_pattern or (None, 0.0)
        s_type, s_conf = spatial_pattern or (None, 0.0)

        # ── Combinaisons temporel + spatial → symboles ──

        # CONSTANT + SYMMETRY → MIRROR ou TRANSLATE
        if t_type == PatternType.CONSTANT:
            if s_type == SpatialPatternType.SYMMETRY_H:
                self._add(SymbolType.MIRROR_H, min(t_conf, s_conf, 0.85),
                          t_type, s_type, focus_position)
            elif s_type == SpatialPatternType.SYMMETRY_V:
                self._add(SymbolType.MIRROR_V, min(t_conf, s_conf, 0.85),
                          t_type, s_type, focus_position)
            elif s_type == SpatialPatternType.GRID:
                self._add(SymbolType.TRANSLATE_RIGHT, t_conf * 0.7,
                          t_type, s_type, focus_position)
                self._add(SymbolType.TRANSLATE_DOWN, t_conf * 0.7,
                          t_type, s_type, focus_position)

        # ACCELERATING → rotation ou scale
        if t_type == PatternType.ACCELERATING:
            self._add(SymbolType.ROTATE_90_CW, t_conf * 0.7,
                      t_type, s_type, focus_position)
            self._add(SymbolType.SCALE_UP, t_conf * 0.6,
                      t_type, s_type, focus_position)

        # DECELERATING → ralentissement, stabilisation
        if t_type == PatternType.DECELERATING:
            self._add(SymbolType.SCALE_DOWN, t_conf * 0.6,
                      t_type, s_type, focus_position)
            self._add(SymbolType.NO_OP, t_conf * 0.3,
                      t_type, s_type, focus_position)

        # OSCILLATING → toggle, inside-out
        if t_type == PatternType.OSCILLATING:
            self._add(SymbolType.TOGGLE, t_conf * 0.8,
                      t_type, s_type, focus_position)
            if s_type == SpatialPatternType.RING:
                self._add(SymbolType.INSIDE_OUT, t_conf * 0.7,
                          t_type, s_type, focus_position)

        # WAVE → translation directionnelle
        if t_type == PatternType.WAVE:
            self._add(SymbolType.TRANSLATE_RIGHT, t_conf * 0.7,
                      t_type, s_type, focus_position)
            self._add(SymbolType.TRANSLATE_DOWN, t_conf * 0.7,
                      t_type, s_type, focus_position)

        # STASIS → no-op ou fill
        if t_type == PatternType.STASIS:
            self._add(SymbolType.NO_OP, t_conf * 0.9,
                      t_type, s_type, focus_position)
            if s_type == SpatialPatternType.CLUSTER:
                self._add(SymbolType.FILL, t_conf * 0.5,
                          t_type, s_type, focus_position)

        # ── Attention event → direction du focus ──
        if attention_event == AttentionEvent.CHANGE:
            # Le focus a bougé → le symbole prioritaire est dans cette direction
            pass  # déjà encodé dans focus_position

        # ── Deltas : détection des changements de couleur ──
        if deltas and len(deltas) >= 2:
            colors_added = deltas[-1].get("colors_added", set())
            colors_removed = deltas[-1].get("colors_removed", set())
            if colors_added and colors_removed:
                if colors_added == colors_removed:
                    self._add(SymbolType.COLOR_SWAP, 0.7,
                              t_type, s_type, focus_position,
                              {"from": list(colors_removed), "to": list(colors_added)})
                else:
                    self._add(SymbolType.COLOR_SHIFT, 0.6,
                              t_type, s_type, focus_position,
                              {"removed": list(colors_removed), "added": list(colors_added)})

        # ── Fallback ──
        if not self.symbols:
            self._add(SymbolType.NO_OP, 0.2, t_type, s_type, focus_position)
            if s_type and s_conf > 0.5:
                self._add(SymbolType.COPY, 0.3, t_type, s_type, focus_position)

        # Trier par confiance
        self.symbols.sort(key=lambda s: s.confidence, reverse=True)
        return self.symbols

    def _add(self, sym_type: SymbolType, confidence: float,
             t_type: Optional[PatternType],
             s_type: Optional[SpatialPatternType],
             focus: tuple[float, float],
             params: Optional[dict] = None):
        """Ajoute un symbole à la liste."""
        if confidence < 0.05:
            return
        self.symbols.append(Symbol(
            type=sym_type,
            confidence=round(confidence, 3),
            source_pattern=t_type,
            source_spatial=s_type,
            focus_position=focus,
            params=params or {},
        ))

    def skill_recommendations(self) -> list[tuple[str, float, str]]:
        """Retourne les skills recommandés avec confiance et raison.

        Returns:
            list of (skill_id, confidence, symbol_name)
        """
        recs = []
        seen = set()
        for sym in self.symbols:
            skill_id = self._skill_map.get(sym.type, "select-skill-for-observation")
            if skill_id not in seen:
                recs.append((skill_id, sym.confidence, sym.type.value))
                seen.add(skill_id)
        return recs

    def summary(self) -> dict:
        """Résumé structuré."""
        return {
            "symbols": [
                {
                    "type": s.type.value,
                    "confidence": s.confidence,
                    "from_temporal": s.source_pattern.value if s.source_pattern else None,
                    "from_spatial": s.source_spatial.value if s.source_spatial else None,
                }
                for s in self.symbols
            ],
            "skill_recommendations": [
                {"skill_id": s, "confidence": c, "reason": r}
                for s, c, r in self.skill_recommendations()
            ],
        }


# ══════════════════════════════════════════════════════════════
#  3.  DÉMO
# ══════════════════════════════════════════════════════════════


def demo():
    """Démo qui interconnecte les 3 modules."""
    import json

    print("🔣 Symbolic Inference — le pont entre perception et action")
    print("=" * 55)
    print()

    inf = SymbolicInference()

    # Scénario 1 : pattern constant + symétrie horizontale
    print("1️⃣  CONSTANT + SYMMETRY_H → MIRROR_H")
    syms = inf.infer(
        temporal_pattern=(PatternType.CONSTANT, 0.9),
        spatial_pattern=(SpatialPatternType.SYMMETRY_H, 0.85),
        focus_position=(5.0, 10.0),
    )
    for s in syms[:3]:
        print(f"   🔣 {s.type.value:<20} confiance: {s.confidence:.0%}  "
              f"(temporel: {s.source_pattern.value if s.source_pattern else '—'}, "
              f"spatial: {s.source_spatial.value if s.source_spatial else '—'})")
    print("   → Skill recommandé:", inf.skill_recommendations()[0][0])
    print()

    # Scénario 2 : oscillating + ring
    inf2 = SymbolicInference()
    print("2️⃣  OSCILLATING + RING → TOGGLE + INSIDE_OUT")
    syms2 = inf2.infer(
        temporal_pattern=(PatternType.OSCILLATING, 0.95),
        spatial_pattern=(SpatialPatternType.RING, 0.7),
        focus_position=(3.0, 3.0),
    )
    for s in syms2[:3]:
        print(f"   🔣 {s.type.value:<20} confiance: {s.confidence:.0%}")
    print("   → Skills recommandés:", [r[0] for r in inf2.skill_recommendations()])
    print()

    # Scénario 3 : accelerating + grille
    inf3 = SymbolicInference()
    print("3️⃣  ACCELERATING + GRID → ROTATE + SCALE_UP")
    syms3 = inf3.infer(
        temporal_pattern=(PatternType.ACCELERATING, 0.7),
        spatial_pattern=(SpatialPatternType.GRID, 0.75),
        focus_position=(7.0, 7.0),
        deltas=[{"colors_added": {1}, "colors_removed": {2}}],
    )
    for s in syms3[:3]:
        print(f"   🔣 {s.type.value:<20} confiance: {s.confidence:.0%}")
    print("   → Skills recommandés:", [r[0] for r in inf3.skill_recommendations()])
    print()

    # Rapport complet
    print("📊 Rapport du dernier scénario :")
    print(json.dumps(inf3.summary(), indent=2))


if __name__ == "__main__":
    demo()
