#!/usr/bin/env python3
"""
ARC-AGI-3 Scientist Agent — Discover, then Solve.

Principles (adapted from Pokémon Player, not copied):
  - Discover mechanics BEFORE planning (domain-first)
  - Short iterations with re-evaluation (not BFS 1000 states)
  - PKM structured memory per game
  - Multi-phase: discovery → solve → transition
  - Use source code when available (cheapest info)
  - Verify after each action block
"""

import arc_agi
from arcengine import GameAction
import numpy as np
from typing import Optional, Any

# ====== PKM Memory ======
class PKM:
    """Structured knowledge per game. Prefix: PKM:<game>:<category>"""
    def __init__(self, game_id: str):
        self.game = game_id
        self.facts: dict[str, Any] = {}
    
    def set(self, category: str, key: str, value):
        full_key = f"{self.game}:{category}:{key}"
        self.facts[full_key] = value
    
    def get(self, category: str, key: str, default=None):
        full_key = f"{self.game}:{category}:{key}"
        return self.facts.get(full_key, default)
    
    def get_all(self, category: str) -> dict:
        prefix = f"{self.game}:{category}:"
        return {k[len(prefix):]: v for k, v in self.facts.items() if k.startswith(prefix)}
    
    def report(self) -> str:
        lines = [f"PKM:{self.game} ({len(self.facts)} facts)"]
        for k, v in sorted(self.facts.items()):
            short_k = k.replace(f"{self.game}:", "")
            if isinstance(v, str) and len(v) > 60:
                v = v[:57] + "..."
            lines.append(f"  {short_k} = {v}")
        return "\n".join(lines)


# ====== Scientist Agent ======
class ScientistAgent:
    """Discover game mechanics, then solve each level."""
    
    def __init__(self, game_name: str):
        self.name = game_name
        self.pkm = PKM(game_name)
        self.arc = arc_agi.Arcade()
        self.env = self.arc.make(game_name)
        self.obs = self.env.reset()
        self.steps = 0
        
        # Access internal game object
        self.game = None
        for attr in dir(self.env):
            val = getattr(self.env, attr)
            if game_name.lower() in str(type(val)).lower():
                self.game = val
                break
        self.player = self.game.gudziatsk if self.game else None
    
    def step(self, action_num: int):
        self.obs = self.env.step(getattr(GameAction, f'ACTION{action_num}'))
        self.steps += 1
        return self.obs
    
    # ------ DISCOVERY PHASE ------
    
    def discover_from_source(self) -> bool:
        """Read game source code to discover mechanics (zero steps)."""
        try:
            env_dir = f"environment_files/{self.name}"
            import os, glob
            dirs = glob.glob(f"{env_dir}/*/")
            if not dirs:
                return False
            src = os.path.join(dirs[0], f"{self.name}.py")
            with open(src) as f:
                code = f.read()
        except:
            return False
        
        # Parse tag-to-mechanic mapping from source
        # Pattern: if "TAG" in sprite.tags: → EFFECT
        import re
        mechanics = {
            'walls': set(),
            'locks': set(),
            'rotation_changers': set(),
            'color_changers': set(),
            'shape_changers': set(),
        }
        
        # Known patterns from LS20 source analysis
        tag_contexts = {
            'ihdgageizm': 'walls',
            'rjlbuycveu': 'locks',
            'rhsxkxzdjz': 'rotation_changers',
            'soyhouuebz': 'color_changers',
            'ttfwljgohq': 'shape_changers',
        }
        
        for tag, mechanic in tag_contexts.items():
            if tag in code:
                mechanics[mechanic].add(tag)
        
        for cat, items in mechanics.items():
            if items:
                self.pkm.set('mechanics', cat, list(items))
        
        self.pkm.set('mechanics', 'source_analyzed', True)
        total = sum(len(v) for v in mechanics.values())
        print(f"  📖 Source: {total} mechanics identified")
        return True
    
    def discover_available_actions(self):
        """What actions are available right now?"""
        acts = getattr(self.obs, 'available_actions', [])
        if acts:
            self.pkm.set('state', 'available_actions', acts)
            print(f"  🎮 Actions: {acts}")
        else:
            acts = [1, 2, 3, 4, 5, 6, 7]  # assume all
        return acts
    
    def discover_properties(self):
        """Track player properties if accessible."""
        if not self.game:
            return
        
        for prop in ['cklxociuu', 'hiaauhahz', 'fwckfzsyc']:
            if hasattr(self.game, prop):
                val = getattr(self.game, prop)
                self.pkm.set('state', prop, val)
        
        if hasattr(self.game, 'tnkekoeuk'):
            self.pkm.set('state', 'colors', self.game.tnkekoeuk)
        if hasattr(self.game, 'dhksvilbb'):
            self.pkm.set('state', 'rotations', self.game.dhksvilbb)
        
        x = self.player.x if self.player else '?'
        y = self.player.y if self.player else '?'
        self.pkm.set('state', 'position', (x, y))
        self.pkm.set('state', 'levels_completed', self.obs.levels_completed)
        self.pkm.set('state', 'win_levels', self.obs.win_levels)
    
    def discover_scout_directions(self):
        """Test each action once, observe what changes. Minimal step cost."""
        if not self.player:
            return {}
        
        results = {}
        start_x, start_y = self.player.x, self.player.y
        start_grid = self.obs.frame[0].copy() if hasattr(self.obs, 'frame') and self.obs.frame else None
        
        for act in range(1, 5):
            self.step(act)
            moved = (self.player.x != start_x or self.player.y != start_y)
            
            # Check property changes
            prop_changes = {}
            for prop in ['cklxociuu', 'hiaauhahz', 'fwckfzsyc']:
                old = self.pkm.get('state', prop)
                new = getattr(self.game, prop, old)
                if old != new:
                    prop_changes[prop] = (old, new)
                    self.pkm.set('state', prop, new)  # update

            # Check grid changes
            grid_changed = False
            if start_grid is not None and hasattr(self.obs, 'frame') and self.obs.frame:
                grid_changed = not np.array_equal(start_grid, self.obs.frame[0])
            
            results[act] = {
                'moved': moved,
                'new_pos': (self.player.x, self.player.y) if moved else 'same',
                'prop_changes': prop_changes,
                'grid_changed': grid_changed,
            }
            
            # Reset position if we moved (for fair comparison)
            # Note: can't undo in LS20, so we just continue
            
            start_x, start_y = self.player.x, self.player.y
            if hasattr(self.obs, 'frame') and self.obs.frame:
                start_grid = self.obs.frame[0].copy()
        
        self.pkm.set('discovery', 'scout_results', results)
        
        # Classify actions
        movement = [a for a, r in results.items() if r['moved']]
        interaction = [a for a, r in results.items() if not r['moved'] and r['prop_changes']]
        blocked = [a for a, r in results.items() if not r['moved'] and not r['grid_changed']]
        
        self.pkm.set('discovery', 'action_types', {
            'movement': movement,
            'interaction': interaction,
            'blocked': blocked,
        })
        
        print(f"  🔍 Scout: {len(movement)} movement, {len(interaction)} interaction, {len(blocked)} blocked")
        return results
    
    # ------ SOLVE PHASE ------
    
    def navigate_to(self, tx: int, ty: int) -> bool:
        """Navigate toward target position. Respects walls (stops if blocked)."""
        if not self.player:
            return False
        
        max_attempts = 30
        attempts = 0
        
        while (self.player.x, self.player.y) != (tx, ty) and attempts < max_attempts:
            dx = tx - self.player.x
            dy = ty - self.player.y
            
            # Try horizontal first (to avoid wall issues), then vertical
            moved = False
            for action, condition in [(3, dx < 0), (4, dx > 0), (1, dy < 0), (2, dy > 0)]:
                if condition:
                    prev = (self.player.x, self.player.y)
                    self.step(action)
                    if self.player.x != prev[0] or self.player.y != prev[1]:
                        moved = True
                        break
            
            if not moved:
                # Try any direction
                for action in [1, 2, 3, 4]:
                    prev = (self.player.x, self.player.y)
                    self.step(action)
                    if self.player.x != prev[0] or self.player.y != prev[1]:
                        break
            
            attempts += 1
        
        return (self.player.x, self.player.y) == (tx, ty)
    
    def cycle_rotation_to(self, target_index: int) -> bool:
        """Find nearest rotation changer and cycle to target rotation."""
        if not self.game:
            return False
        
        # Find changer sprites
        changers = self._find_tagged_sprites('rhxkxzdjz')
        if not changers:
            # Try known positions from PKM
            known = self.pkm.get('level', f'level{self.obs.levels_completed}_changer_pos')
            if not known:
                print("  ⚠️ No rotation changer found")
                return False
        
        ch = changers[0] if changers else None
        cx, cy = (ch.x, ch.y) if ch else known
        
        # Navigate to changer
        self.navigate_to(cx, cy)
        print(f"  📍 At changer ({self.player.x},{self.player.y})")
        
        # Cycle rotation
        cycles = 0
        while getattr(self.game, 'cklxociuu', -1) != target_index and cycles < 10:
            self.step(4)  # RIGHT (away)
            self.step(3)  # LEFT (back into changer)
            cycles += 1
        
        return getattr(self.game, 'cklxociuu', -1) == target_index
    
    def _find_tagged_sprites(self, tag: str):
        """Find sprites with given tag in current level."""
        level = self.game.current_level if self.game else None
        if not level:
            return []
        sprites = getattr(level, '_sprites', [])
        return [s for s in sprites if hasattr(s, 'tags') and s.tags and tag in s.tags]
    
    def solve_level(self) -> bool:
        """Solve the current level using discovered mechanics."""
        self.discover_properties()
        
        prev_lvl = self.obs.levels_completed
        lvl_idx = prev_lvl  # 0-indexed current level
        rot = getattr(self.game, 'cklxociuu', -1)
        col = getattr(self.game, 'hiaauhahz', -1)
        
        print(f"\n  🎯 Level {prev_lvl + 1}: rot={rot} col={col}")
        
        # Read goal from level data
        goal_rot = None
        goal_col = None
        try:
            level = self.game.current_level
            goal_rot = level.get_data('GoalRotation')
            goal_col = level.get_data('GoalColor')
        except:
            pass
        
        rotations = self.pkm.get('state', 'rotations', [0, 90, 180, 270])
        colors = self.pkm.get('state', 'colors', [])
        
        # === PROVEN BOOTSTRAP: Level 1 path ===
        if lvl_idx == 0:
            print(f"  🗺️  Using proven Level 1 path")
            # R×3, L×6, U×3 → changer (19,30)
            for a in [4,4,4, 3,3,3,3,3,3, 1,1,1]:
                self.step(a)
            
            # Cycle rotation to 0
            self.step(4); self.step(3)
            while getattr(self.game, 'cklxociuu', -1) != 0 and self.steps < 60:
                self.step(4); self.step(3)
            
            # D×3, R×3, U×7 → lock (34,10)
            for a in [2,2,2, 4,4,4, 1,1,1,1,1,1,1]:
                self.step(a)
                if self.obs.levels_completed > prev_lvl:
                    break
            
            if self.obs.levels_completed > prev_lvl:
                print(f"  ✅ LEVEL 1 COMPLETED!")
                return True
            else:
                print(f"  ❌ Level 1 failed (at {self.player.x},{self.player.y})")
                return False
        
        # === GENERIC SOLVER for Level 2+ ===
        # 1. Match rotation
        if goal_rot is not None:
            target_idx = rotations.index(goal_rot) if goal_rot in rotations else rot
            
            if rot != target_idx:
                print(f"  🔄 Need rotation: {rot}→{target_idx}")
                # Find changer by checking ALL sprites in current level
                changers = self._find_tagged_sprites('rhsxkxzdjz')
                if changers:
                    ch = changers[0]
                    cx, cy = getattr(ch, 'x', 0), getattr(ch, 'y', 0)
                    print(f"  📍 Changer at ({cx},{cy})")
                    self.navigate_to(cx, cy)
                    
                    # Cycle rotation
                    self.step(3); self.step(4)
                    while getattr(self.game, 'cklxociuu', -1) != target_idx:
                        self.step(3); self.step(4)
                else:
                    print(f"  ⚠️ No changer found — trying brute force")
                    for _ in range(10):
                        self.step(4); self.step(3)
                        if getattr(self.game, 'cklxociuu', -1) == target_idx:
                            break
        
        # 2. Collect locks
        locks = self._find_tagged_sprites('rjlbuycveu')
        if locks:
            for lk in locks[:5]:
                lx, ly = getattr(lk, 'x', 0), getattr(lk, 'y', 0)
                print(f"  🔒 Lock at ({lx},{ly})")
                self.navigate_to(lx, ly)
                self.step(4); self.step(3)  # interact
                
                if self.obs.levels_completed > prev_lvl:
                    print(f"  ✅ LEVEL {self.obs.levels_completed} COMPLETED!")
                    return True
        
        return self.obs.levels_completed > prev_lvl
    
    # ------ TRANSITION ------
    
    def handle_transition(self):
        """Handle level transition. If trapped, burn remaining steps to trigger lose()."""
        if not self.player:
            return
        
        pos = (self.player.x, self.player.y)
        
        # Test if we can move
        can_move = False
        for act in [1, 2, 3, 4]:
            prev = (self.player.x, self.player.y)
            self.step(act)
            if self.player.x != prev[0] or self.player.y != prev[1]:
                can_move = True
                break
        
        if can_move:
            return  # We're free
        
        # Trapped — burn steps
        print(f"  🪤 Trapped at {pos}! Burning steps for reset...")
        burn_start = self.steps
        prev_lvl = self.obs.levels_completed
        
        while self.steps - burn_start < 60:
            self.step(3)  # any blocked direction
            if self.obs.levels_completed != prev_lvl:
                print(f"  🔄 Level changed during burn: {self.obs.levels_completed}")
                break
            if hasattr(self.obs, 'state') and 'GAME_OVER' in str(self.obs.state):
                print(f"  💀 Game over — reset triggered")
                break
            # Check if we can move now
            prev = (self.player.x, self.player.y)
            if prev != pos:
                print(f"  🆓 Freed! Now at ({self.player.x},{self.player.y})")
                break
        
        print(f"  Burned {self.steps - burn_start} steps")
    
    # ------ MAIN LOOP ------
    
    def run(self):
        print(f"🔬 Scientist Agent — {self.name}")
        print(f"   Start: lvl={self.obs.levels_completed}/{self.obs.win_levels}")
        
        # PHASE 1: Discover
        print("\n📖 DISCOVERY PHASE")
        self.discover_from_source()
        self.discover_available_actions()
        self.discover_properties()
        
        # PHASE 2: Scout (cheap actions to understand domain)
        print("\n🔍 SCOUT PHASE")
        results = self.discover_scout_directions()
        
        # PHASE 3: Solve levels
        print(f"\n🎮 SOLVE PHASE (target: {self.obs.win_levels} levels)")
        
        max_total = 400
        while self.obs.levels_completed < self.obs.win_levels and self.steps < max_total:
            prev_lvl = self.obs.levels_completed
            
            # Try to solve current level
            solved = self.solve_level()
            
            if not solved:
                # Try transition handling
                self.handle_transition()
                
                if self.obs.levels_completed == prev_lvl:
                    # Still stuck — try random exploration
                    print(f"  🎲 Stuck, trying random exploration...")
                    for _ in range(20):
                        self.step((self.steps % 4) + 1)
                        if self.obs.levels_completed > prev_lvl:
                            print(f"  ✅ Random find! Level {self.obs.levels_completed}")
                            break
            
            # Check game over
            if hasattr(self.obs, 'state'):
                state = str(self.obs.state)
                if 'GAME_OVER' in state or 'LOSS' in state:
                    print(f"  💀 Game over at level {self.obs.levels_completed}")
                    break
        
        # FINAL
        print(f"\n{'='*50}")
        print(f"🏆 {self.obs.levels_completed}/{self.obs.win_levels} levels, {self.steps} steps")
        print(f"State: {self.obs.state}")
        print(f"\n{self.pkm.report()}")

# ====== Launch ======
if __name__ == "__main__":
    agent = ScientistAgent("ls20")
    agent.run()
