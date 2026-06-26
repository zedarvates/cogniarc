#!/usr/bin/env python3
"""
ARC-AGI-3 Agent v4 — Perception-guided search with full cognitive pipeline.

Pipeline:
  Phase 1:  Map action transforms
  Phase 1.5: Temporal analysis (patterns of change)
  Phase 1.6: Spatial analysis (regions, relations)
  Phase 1.7: Attention tracking (focus point, saliency)
  Phase 1.8: Symbolic inference (perception → symbols)
  Phase 1.9: Symbol memory (learn from outcomes)
  Phase 2:   Perception-guided BFS
  Phase 3:   Execute solution + reinforce symbols
"""

from __future__ import annotations

import json
import sys
import time
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent))

from transforms import Transform, map_transforms, _hash_grid
from skill_tree import SkillTree
from temporal_inference import TemporalReasoner, PatternType
from spatial_inference import SpatialReasoner, SpatialPatternType
from attention import AttentionModel, AttentionEvent
from symbolic_inference import SymbolicInference, SymbolType
from symbol_memory import SymbolMemory

import arc_agi
from arcengine import GameAction


def real_bfs(env, actions: List[int], max_depth: int = 10,
             max_states: int = 10000,
             priority_actions: Optional[List[int]] = None) -> Optional[Tuple[List[int], Any]]:
    """BFS with optional action prioritization from symbolic inference.

    Si priority_actions est fourni, ces actions sont essayées en premier
    (guidage symbolique de la recherche).
    """
    env.reset()
    obs = env.reset()
    start_hash = _hash_grid(obs.frame[0])
    steps = 0

    stack = [(start_hash, [], obs)]
    visited = {start_hash}
    best = None
    best_levels = 0

    # Ordonner les actions : prioritaires d'abord, puis le reste
    if priority_actions:
        ordered_actions = list(dict.fromkeys(priority_actions + actions))
    else:
        ordered_actions = list(actions)

    while stack and steps < max_states:
        sh, path, obs = stack.pop()

        if len(path) >= max_depth:
            continue

        for act_num in ordered_actions:
            env.reset()
            obs2 = env.reset()
            for a in path:
                obs2 = env.step(getattr(GameAction, f"ACTION{a}"))
            obs2 = env.step(getattr(GameAction, f"ACTION{act_num}"))
            steps += 1

            new_hash = _hash_grid(obs2.frame[0])
            new_path = path + [act_num]
            levels = getattr(obs2, 'levels_completed', 0)

            if levels > best_levels:
                best_levels = levels
                best = (new_path, obs2)
                print(f"  [BFS] Found level {levels} at depth {len(new_path)}: {new_path}")
                if str(getattr(obs2, 'state', '')) in ('WIN', 'FINISHED', 'GameState.WIN'):
                    return best

            if new_hash not in visited:
                visited.add(new_hash)
                stack.append((new_hash, new_path, obs2))

    print(f"  [BFS] Explored {steps} states, {len(visited)} unique, best_level={best_levels}")
    return best


class ArcAgentV4:
    """Agent ARC-AGI-3 avec perception cognitive complète."""

    def __init__(self, game_name: str, max_steps: int = 500,
                 memory_path: str = "~/.cache/cogniarc/symbol_memory.json"):
        self.game_name = game_name
        self.max_steps = max_steps
        self.arc = arc_agi.Arcade()
        self.env = self.arc.make(game_name)
        self.skills = SkillTree()
        self.transforms: Dict[int, Transform] = {}
        self.symbol_memory = SymbolMemory(memory_path)
        self.symbolic = SymbolicInference()
        self.stats = {
            "phases": {},
            "symbols_found": [],
            "actions_tried": 0,
        }

    def run(self) -> dict:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f" ARC AGENT V4 — {self.game_name}")
        print(f"{'='*60}")

        obs = self.env.reset()
        initial_grid = obs.frame[0].copy()
        actions = list(obs.available_actions or [])
        win_levels = getattr(obs, 'win_levels', 1)

        print(f"  Grid: {initial_grid.shape}, Actions: {actions}, Levels: {win_levels}")

        # ══════════════════════════════════════════════════════════
        # Phase 1: Map transforms
        # ══════════════════════════════════════════════════════════
        t1 = time.time()
        print("\n[Phase 1] Mapping transforms...")
        self.transforms = map_transforms(self.env, initial_grid, actions)
        for act_num, t in sorted(self.transforms.items()):
            print(f"  {t.describe()}")
        self.stats["phases"]["1_transforms"] = time.time() - t1

        # ══════════════════════════════════════════════════════════
        # Phase 1.5: Temporal analysis
        # ══════════════════════════════════════════════════════════
        t15 = time.time()
        print("\n[Phase 1.5] Temporal pattern analysis...")
        temporal = TemporalReasoner()
        temporal.add_frame(initial_grid)
        obs_temp = self.env.reset()
        for _ in range(min(3, len(actions))):
            action = np.random.choice(actions)
            obs_temp = self.env.step(getattr(GameAction, f"ACTION{action}"))
            temporal.add_frame(obs_temp.frame[0].copy())

        temporal_pattern = temporal.analyze()
        t_type = temporal_pattern.type
        t_conf = temporal_pattern.confidence
        print(f"  Pattern: {t_type.value} (confidence: {t_conf:.0%})")
        print(f"  Magnitudes: {[round(d.magnitude, 3) for d in temporal.deltas]}")
        self.stats["phases"]["1.5_temporal"] = time.time() - t15

        # ══════════════════════════════════════════════════════════
        # Phase 1.6: Spatial analysis
        # ══════════════════════════════════════════════════════════
        t16 = time.time()
        print("\n[Phase 1.6] Spatial analysis...")
        spatial = SpatialReasoner(initial_grid)
        regions = spatial.segment()
        relations = spatial.relate()
        spatial_pattern = spatial.analyze()
        s_type = spatial_pattern.type
        s_conf = spatial_pattern.confidence
        print(f"  Regions: {len(regions)}")
        for r in regions[:5]:
            print(f"    R{r.id}: color={r.color}, shape={r.shape}, "
                  f"area={r.area}, center=({r.center[0]:.1f},{r.center[1]:.1f})")
        print(f"  Relations: {len(relations)}")
        rel_types = set(r.type.value for r in relations)
        print(f"  Types: {sorted(rel_types)}")
        print(f"  Global pattern: {s_type.value} (confidence: {s_conf:.0%})")
        self.stats["phases"]["1.6_spatial"] = time.time() - t16

        # ══════════════════════════════════════════════════════════
        # Phase 1.7: Attention tracking
        # ══════════════════════════════════════════════════════════
        t17 = time.time()
        print("\n[Phase 1.7] Attention tracking...")
        attention = AttentionModel(initial_grid.shape, focus_radius=3.0)
        # Utilise le dernier delta temporel pour positionner le focus
        if temporal.deltas:
            last_delta = temporal.deltas[-1]
            focus = attention.update_from_delta(
                last_delta.changed, last_delta.added)
        else:
            focus = attention.update_from_regions(regions)

        print(f"  Focus: ({focus.position[0]:.1f}, {focus.position[1]:.1f})")
        print(f"  Event: {focus.event.value}")
        print(f"  Velocity: ({focus.velocity[0]:.2f}, {focus.velocity[1]:.2f})")
        focus_zone = attention.get_focus_region()
        print(f"  Focus zone: {np.sum(focus_zone)} pixels (radius {focus.radius})")
        self.stats["phases"]["1.7_attention"] = time.time() - t17

        # ══════════════════════════════════════════════════════════
        # Phase 1.8: Symbolic inference
        # ══════════════════════════════════════════════════════════
        t18 = time.time()
        print("\n[Phase 1.8] Symbolic inference...")

        # Chercher d'abord dans la mémoire
        memory_results = self.symbol_memory.lookup(
            temporal_pattern=t_type if t_conf > 0.3 else None,
            spatial_pattern=s_type if s_conf > 0.3 else None,
        )

        if memory_results:
            print(f"  🧠 Found {len(memory_results)} matches in symbol memory")
            for entry, score in memory_results[:3]:
                print(f"     {entry.symbol_type:<20} → {entry.skill_id:<25} "
                      f"(score: {score:.2f}, succès: {entry.success_count})")

        # Inférer les symboles depuis les patterns actuels
        symbols = self.symbolic.infer(
            temporal_pattern=(t_type, t_conf) if t_conf > 0.3 else None,
            spatial_pattern=(s_type, s_conf) if s_conf > 0.3 else None,
            attention_event=focus.event,
            focus_position=focus.position,
            deltas=[d.to_dict() for d in temporal.deltas],
            regions=regions,
        )

        print(f"  🔣 Inferred {len(symbols)} symbols:")
        for s in symbols[:4]:
            print(f"     {s.type.value:<20} (confidence: {s.confidence:.0%})")
            # Enregistrer dans la mémoire
            self.symbol_memory.record(
                s.type, 
                self.symbolic._skill_map.get(s.type, "select-skill-for-observation"),
                temporal_pattern=t_type if t_conf > 0.3 else None,
                spatial_pattern=s_type if s_conf > 0.3 else None,
            )

        # Recommandations de skills
        recommendations = self.symbolic.skill_recommendations()
        print(f"  🎯 Skill recommendations:")
        for skill_id, conf, reason in recommendations[:4]:
            # Vérifier si la mémoire a une meilleure suggestion
            mem_best = self.symbol_memory.get_best_skill(
                next((s.type for s in symbols if s.type.value == reason), SymbolType.NO_OP))
            effective_conf = mem_best[1] if mem_best else conf
            print(f"     {skill_id:<25} (confidence: {effective_conf:.0%}) via {reason}")

        self.stats["symbols_found"] = [s.type.value for s in symbols[:5]]
        self.stats["phases"]["1.8_symbolic"] = time.time() - t18

        # ══════════════════════════════════════════════════════════
        # Phase 1.9: Préparer le guidage symbolique
        # ══════════════════════════════════════════════════════════
        # Mapper les symboles vers des actions prioritaires
        priority_actions: List[int] = []
        for skill_id, conf, _ in recommendations:
            if conf > 0.3:
                # Trouver les actions correspondant à ce skill
                for act_num, transform in self.transforms.items():
                    if skill_id in transform.describe().lower():
                        priority_actions.append(act_num)
                # Fallback: prioriser les actions qui bougent vers le focus
                if not priority_actions:
                    focus_r, focus_c = focus.position
                    center_r, center_c = initial_grid.shape[0] / 2, initial_grid.shape[1] / 2
                    if abs(focus_c - center_c) > 1:
                        priority_actions.append(1)  # ACTION1 = déplacement horizontal

        # Reset environment for BFS
        self.env.reset()

        # ══════════════════════════════════════════════════════════
        # Phase 2: Perception-guided BFS
        # ══════════════════════════════════════════════════════════
        print(f"\n[Phase 2] Perception-guided BFS (max_depth=10)...")
        if priority_actions:
            print(f"  🧭 Guided search with priority actions: {priority_actions}")
        t2 = time.time()
        result = real_bfs(self.env, actions, max_depth=10, max_states=5000,
                          priority_actions=priority_actions if priority_actions else None)
        self.stats["phases"]["2_bfs"] = time.time() - t2

        # ══════════════════════════════════════════════════════════
        # Phase 3: Execute + Reinforce
        # ══════════════════════════════════════════════════════════
        total_steps = 0
        solved = False
        if result:
            plan, winning_obs = result
            solved = True
            print(f"\n[Phase 3] ✅ Solution found: {plan}")
            # Execute one more time to verify
            self.env.reset()
            obs = self.env.reset()
            for act_num in plan:
                obs = self.env.step(getattr(GameAction, f"ACTION{act_num}"))
                total_steps += 1
            levels = getattr(obs, 'levels_completed', 0)
            state = getattr(obs, 'state', '?')
            print(f"  Levels: {levels}/{win_levels}, State: {state}")

            # Renforcer les symboles qui ont réussi
            print("\n[Phase 3.5] Reinforcing symbol memory...")
            for s in symbols[:3]:
                self.symbol_memory.reinforce(
                    s.type,
                    self.symbolic._skill_map.get(s.type, "select-skill-for-observation"),
                    success=True,
                    temporal_pattern=t_type if t_conf > 0.3 else None,
                    spatial_pattern=s_type if s_conf > 0.3 else None,
                )
                print(f"  ✅ {s.type.value:<20} → reinforced (success)")
        else:
            print(f"\n[Phase 3] ❌ No solution found")
            # Affaiblir les symboles qui n'ont pas marché
            for s in symbols[:3]:
                self.symbol_memory.reinforce(
                    s.type,
                    self.symbolic._skill_map.get(s.type, "select-skill-for-observation"),
                    success=False,
                )
                print(f"  ❌ {s.type.value:<20} → weakened (failure)")

        # Sauvegarder la mémoire
        self.symbol_memory.save()
        print(f"  💾 Symbol memory saved ({self.symbol_memory.stats()['total']} entries)")

        elapsed = time.time() - t0
        result_dict = {
            "game": self.game_name,
            "steps": total_steps,
            "plan_found": solved,
            "elapsed": round(elapsed, 1),
            "symbols_used": [s.type.value for s in symbols[:3]],
            "memory_entries": self.symbol_memory.stats()["total"],
        }

        print(f"\n{'='*60}")
        print(f" RESULT: {self.game_name} — plan={'YES' if solved else 'NO'}, "
              f"{elapsed:.1f}s")
        print(f" Symbols: {', '.join(s.type.value for s in symbols[:3])}")
        print(f" Memory: {self.symbol_memory.stats()['total']} entries, "
              f"{self.symbol_memory.stats()['success_rate']:.0%} success rate")
        print(f"{'='*60}")

        return result_dict


if __name__ == "__main__":
    games = sys.argv[1:] if len(sys.argv) > 1 else ["ls20"]
    for game in games:
        agent = ArcAgentV4(game, max_steps=300)
        result = agent.run()
        # Afficher le résumé JSON
        print(f"\nJSON: {json.dumps(result, indent=2)}")
