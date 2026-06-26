#!/usr/bin/env python3
"""
Symbol Memory — reconnaissance, mémorisation et apprentissage de symboles.

Un système symbolique n'est PAS un dictionnaire codé en dur.
C'est une MÉMOIRE qui :
    1. RECONNAÎT des symboles dans les patterns perçus
    2. MÉMORISE les combinaisons pattern→symbole qui ont réussi
    3. APPREND de l'expérience (renforcement des mappings qui marchent)
    4. DÉCOUVRE de nouveaux symboles par combinaison de patterns connus
    5. OUBLIE les mappings qui ne marchent pas (décroissance)

La différence avec symbolic_inference.py :
    - symbolic_inference.py = mapping RÈGLE (codé en dur, statique)
    - symbol_memory.py = apprentissage (évolutif, persistant, adaptatif)

Architecture :
    Chaque symbole est un nœud dans un graphe :
        (pattern_temporel, pattern_spatial, contexte_attention)
            ↓ combinaison
        (symbole, confiance, taux_de_réussite)
            ↓ mapping
        (skill_id, nb_succès, nb_échecs)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from collections import defaultdict

# Same pattern as arc_agent.py: add parent to path, then absolute imports
_cogniarc_path = Path(__file__).resolve().parent
if str(_cogniarc_path) not in sys.path:
    sys.path.insert(0, str(_cogniarc_path))

from temporal_inference import PatternType
from spatial_inference import SpatialPatternType
from attention import AttentionEvent
from symbolic_inference import SymbolType


# ══════════════════════════════════════════════════════════════
#  1.  ENTRÉE SYMBOLIQUE — l'unité de mémoire
# ══════════════════════════════════════════════════════════════


@dataclass
class SymbolEntry:
    """Une entrée dans la mémoire symbolique.

    Attributes:
        temporal_pattern: Pattern temporel observé
        spatial_pattern: Pattern spatial observé
        attention_event: Événement d'attention associé
        symbol_type: Le symbole reconnu
        skill_id: Le skill recommandé
        confidence: Confiance actuelle [0-1]
        success_count: Nombre de fois que ce mapping a réussi
        fail_count: Nombre de fois qu'il a échoué
        last_used: Timestamp de la dernière utilisation
        created: Timestamp de création
    """
    temporal_pattern: str           # PatternType.value
    spatial_pattern: str            # SpatialPatternType.value (ou "")
    attention_event: str            # AttentionEvent.value (ou "")
    symbol_type: str                # SymbolType.value
    skill_id: str                   # SkillDAG ID
    confidence: float = 0.5
    success_count: int = 0
    fail_count: int = 0
    last_used: float = 0.0
    created: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def key(self) -> str:
        """Clé unique : combinaison des patterns + symbole."""
        return f"{self.temporal_pattern}|{self.spatial_pattern}|{self.attention_event}|{self.symbol_type}"

    def reinforce(self, success: bool):
        """Renforce ou affaiblit l'entrée selon le résultat."""
        if success:
            self.success_count += 1
            # La confiance augmente mais plafonne
            self.confidence = min(1.0, self.confidence + 0.1)
        else:
            self.fail_count += 1
            # La confiance diminue mais ne passe pas sous 0.05
            self.confidence = max(0.05, self.confidence - 0.05)
        self.last_used = time.time()


# ══════════════════════════════════════════════════════════════
#  2.  MÉMOIRE SYMBOLIQUE
# ══════════════════════════════════════════════════════════════


class SymbolMemory:
    """Mémoire symbolique persistante qui apprend de l'expérience.

    Stocke les mappings pattern→symbole→skill qui ont fonctionné,
    et les renforce ou les affaiblit selon les résultats.

    Principe : chaque fois qu'un symbole mène à une action qui résout
    la grille, le mapping est renforcé. Sinon, il est affaibli.
    Les mappings inutilisés finissent par être oubliés (décroissance).
    """

    def __init__(self, storage_path: str = "~/.cache/cogniarc/symbol_memory.json"):
        self.storage_path = Path(storage_path).expanduser()
        self.entries: dict[str, SymbolEntry] = {}  # key → entry
        self._dirty = False

        # Index par type de symbole
        self._by_symbol: dict[str, list[str]] = defaultdict(list)
        # Index par skill
        self._by_skill: dict[str, list[str]] = defaultdict(list)

        self._load()

    # ── Persistance ──

    def _load(self):
        """Charge la mémoire depuis le disque."""
        if not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text())
            for entry_dict in data.get("entries", []):
                entry = SymbolEntry(**entry_dict)
                self.entries[entry.key] = entry
                self._by_symbol[entry.symbol_type].append(entry.key)
                self._by_skill[entry.skill_id].append(entry.key)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def save(self):
        """Sauvegarde la mémoire sur le disque."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "count": len(self.entries),
            "entries": [asdict(e) for e in self.entries.values()],
            "saved_at": time.time(),
        }
        self.storage_path.write_text(json.dumps(data, indent=2))
        self._dirty = False

    # ── Enregistrement et récupération ──

    def record(self, symbol_type: SymbolType, skill_id: str,
               temporal_pattern: Optional[PatternType] = None,
               spatial_pattern: Optional[SpatialPatternType] = None,
               attention_event: Optional[AttentionEvent] = None,
               success: Optional[bool] = None) -> SymbolEntry:
        """Enregistre ou met à jour une entrée symbolique.

        Si l'entrée existe déjà, elle est mise à jour.
        Si success est fourni, la confiance est ajustée.
        """
        t = temporal_pattern.value if temporal_pattern else ""
        s = spatial_pattern.value if spatial_pattern else ""
        a = attention_event.value if attention_event else ""
        sym_val = symbol_type.value

        entry = SymbolEntry(
            temporal_pattern=t,
            spatial_pattern=s,
            attention_event=a,
            symbol_type=sym_val,
            skill_id=skill_id,
            created=time.time(),
            last_used=time.time(),
        )

        key = entry.key

        if key in self.entries:
            entry = self.entries[key]
            entry.last_used = time.time()
        else:
            self.entries[key] = entry
            self._by_symbol[sym_val].append(key)
            self._by_skill[skill_id].append(key)

        if success is not None:
            entry.reinforce(success)

        self._dirty = True
        return entry

    def lookup(self, temporal_pattern: Optional[PatternType] = None,
               spatial_pattern: Optional[SpatialPatternType] = None,
               attention_event: Optional[AttentionEvent] = None,
               min_confidence: float = 0.1) -> list[tuple[SymbolEntry, float]]:
        """Recherche les entrées correspondant à une combinaison de patterns.

        Returns:
            Liste de (entry, relevance_score) triée par pertinence.
        """
        t = temporal_pattern.value if temporal_pattern else ""
        s = spatial_pattern.value if spatial_pattern else ""
        a = attention_event.value if attention_event else ""

        results: list[tuple[SymbolEntry, float]] = []

        for key, entry in self.entries.items():
            if entry.confidence < min_confidence:
                continue

            # Score de correspondance
            score = 0.0
            total = 0

            if t and entry.temporal_pattern == t:
                score += 1.0
                total += 1
            if s and entry.spatial_pattern == s:
                score += 1.0
                total += 1
            if a and entry.attention_event == a:
                score += 0.5  # L'attention est moins discriminante
                total += 0.5

            if total > 0:
                match_score = (score / total) * entry.confidence
                results.append((entry, match_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_best_skill(self, symbol_type: SymbolType) -> Optional[tuple[str, float]]:
        """Trouve le meilleur skill pour un symbole donné, basé sur l'historique."""
        sym_val = symbol_type.value
        candidates: list[tuple[str, float, int]] = []  # (skill_id, avg_confidence, successes)

        for key in self._by_symbol.get(sym_val, []):
            entry = self.entries.get(key)
            if entry and entry.confidence > 0.1:
                candidates.append((entry.skill_id, entry.confidence, entry.success_count))

        if not candidates:
            return None

        # Trier par confiance, puis par nombre de succès
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return (candidates[0][0], candidates[0][1])

    # ── Apprentissage et décroissance ──

    def reinforce(self, symbol_type: SymbolType, skill_id: str, success: bool,
                  temporal_pattern: Optional[PatternType] = None,
                  spatial_pattern: Optional[SpatialPatternType] = None):
        """Renforce ou affaiblit un mapping après un résultat."""
        t = temporal_pattern.value if temporal_pattern else ""
        s = spatial_pattern.value if spatial_pattern else ""

        for entry in self.entries.values():
            if (entry.symbol_type == symbol_type.value and
                entry.skill_id == skill_id and
                (not t or entry.temporal_pattern == t) and
                (not s or entry.spatial_pattern == s)):
                entry.reinforce(success)
                self._dirty = True
                break

    def decay(self, factor: float = 0.99, min_confidence: float = 0.05):
        """Applique une décroissance à toutes les entrées inutilisées.

        Les entrées non utilisées depuis longtemps perdent de la confiance.
        En dessous de min_confidence, elles sont oubliées.
        """
        now = time.time()
        to_remove: list[str] = []

        for key, entry in self.entries.items():
            days_unused = (now - entry.last_used) / 86400
            if days_unused > 7:
                # Décroissance progressive
                entry.confidence *= factor ** days_unused
                if entry.confidence < min_confidence:
                    to_remove.append(key)
                self._dirty = True

        for key in to_remove:
            entry = self.entries.pop(key, None)
            if entry:
                if entry.symbol_type in self._by_symbol:
                    self._by_symbol[entry.symbol_type] = [
                        k for k in self._by_symbol[entry.symbol_type] if k != key
                    ]
                if entry.skill_id in self._by_skill:
                    self._by_skill[entry.skill_id] = [
                        k for k in self._by_skill[entry.skill_id] if k != key
                    ]

        if to_remove:
            print(f"  🧹 Oublié {len(to_remove)} entrées symboliques (confiance < {min_confidence})")

    # ── Découverte de nouveaux symboles ──

    def discover(self, symbol_type: SymbolType) -> bool:
        """Vérifie si un symbole est NOUVEAU (jamais vu).

        Utile pour déclencher une exploration : si on voit un symbole
        inconnu, l'agent peut essayer différents skills pour découvrir
        quel mapping fonctionne.
        """
        return symbol_type.value not in self._by_symbol

    def get_unexplored(self) -> list[SymbolType]:
        """Retourne les symboles qui n'ont jamais été essayés."""
        explored = set(self._by_symbol.keys())
        all_symbols = set(st.value for st in SymbolType)
        unexplored = all_symbols - explored
        return [SymbolType(s) for s in unexplored]

    # ── Stats ──

    def stats(self) -> dict:
        """Statistiques de la mémoire symbolique."""
        total = len(self.entries)
        if total == 0:
            return {"total": 0, "symbols": {}, "message": "Mémoire vide"}

        success_total = sum(e.success_count for e in self.entries.values())
        fail_total = sum(e.fail_count for e in self.entries.values())

        # Stats par symbole
        by_symbol = {}
        for sym, keys in self._by_symbol.items():
            entries = [self.entries[k] for k in keys if k in self.entries]
            if entries:
                avg_conf = sum(e.confidence for e in entries) / len(entries)
                successes = sum(e.success_count for e in entries)
                by_symbol[sym] = {
                    "count": len(entries),
                    "avg_confidence": round(avg_conf, 3),
                    "successes": successes,
                }

        return {
            "total": total,
            "unique_symbols": len(self._by_symbol),
            "unique_skills": len(self._by_skill),
            "total_successes": success_total,
            "total_fails": fail_total,
            "success_rate": round(success_total / max(success_total + fail_total, 1), 3),
            "by_symbol": by_symbol,
            "stored_at": str(self.storage_path),
        }

    def summary(self) -> str:
        """Rapport lisible."""
        s = self.stats()
        lines = [
            "🧠 Symbol Memory Report",
            f"  Entrées: {s['total']}",
            f"  Symboles uniques: {s['unique_symbols']}",
            f"  Skills associés: {s['unique_skills']}",
            f"  Succès/Échecs: {s['total_successes']}/{s['total_fails']}",
            f"  Taux de succès: {s['success_rate']:.0%}",
            f"  Fichier: {s['stored_at']}",
        ]
        if s.get('by_symbol'):
            lines.append("  Par symbole:")
            for sym, data in sorted(s['by_symbol'].items()):
                lines.append(f"    {sym:<20} {data['count']} entrées, "
                           f"confiance moyenne {data['avg_confidence']:.2f}, "
                           f"{data['successes']} succès")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  3.  DÉMO
# ══════════════════════════════════════════════════════════════


def demo():
    """Démo : apprendre des symboles par renforcement."""
    print("🧠 Symbol Memory — apprentissage par renforcement")
    print("=" * 55)
    print()

    mem = SymbolMemory(storage_path="/tmp/symbol_memory_demo.json")

    # Simulation : l'agent essaie des mappings, certains réussissent
    print("1️⃣  Phase d'apprentissage : 5 essais")
    trials = [
        (SymbolType.MIRROR_H, "rotate-to-goal", PatternType.CONSTANT,
         SpatialPatternType.SYMMETRY_H, True),   # ✅ Succès
        (SymbolType.MIRROR_V, "rotate-to-goal", PatternType.CONSTANT,
         SpatialPatternType.SYMMETRY_V, True),   # ✅ Succès
        (SymbolType.TOGGLE, "interact-with-object", PatternType.OSCILLATING,
         SpatialPatternType.RING, True),          # ✅ Succès
        (SymbolType.ROTATE_90_CW, "rotate-to-goal", PatternType.ACCELERATING,
         SpatialPatternType.GRID, False),          # ❌ Échec
        (SymbolType.SCALE_UP, "interact-with-object", PatternType.ACCELERATING,
         SpatialPatternType.GRID, True),           # ✅ Succès
    ]

    for sym, skill, t, s, success in trials:
        mem.record(sym, skill, temporal_pattern=t, spatial_pattern=s, success=success)
        print(f"   {'✅' if success else '❌'} {sym.value:<20} → {skill:<25} "
              f"(temporel: {t.value}, spatial: {s.value})")

    print()
    print("2️⃣  Recherche : 'CONSTANT + SYMMETRY_H'")
    results = mem.lookup(temporal_pattern=PatternType.CONSTANT,
                         spatial_pattern=SpatialPatternType.SYMMETRY_H)
    for entry, score in results:
        print(f"   🔗 {entry.symbol_type:<20} → {entry.skill_id:<25} "
              f"(score: {score:.2f}, confiance: {entry.confidence:.2f})")

    print()
    print("3️⃣  Meilleur skill pour MIRROR_H:")
    best = mem.get_best_skill(SymbolType.MIRROR_H)
    if best:
        print(f"   🏆 {best[0]} (confiance: {best[1]:.2f})")

    print()
    print("4️⃣  Symboles inexplorés:")
    unexplored = mem.get_unexplored()
    print(f"   {len(unexplored)} symboles jamais essayés")
    print(f"   Exemples: {[u.value for u in unexplored[:5]]}")

    print()
    print(mem.summary())

    # Nettoie
    Path("/tmp/symbol_memory_demo.json").unlink(missing_ok=True)


if __name__ == "__main__":
    demo()
