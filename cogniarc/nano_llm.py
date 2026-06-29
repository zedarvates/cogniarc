"""Nano-LLM tier — small transformer (Qwen2.5-0.5B, SmolLM-135M) via Ollama.
Sits between micro-NN (<1ms) and cloud LLM (expensive).

Tier 0.5: Cheap transformer that reads text/grid, proposes actions.
Wrapped in harnais for safety — deterministic checks validate output.
"""

import json
import numpy as np
from typing import Optional, Tuple, Dict, Any


class NanoLLM:
    """Small HF transformer loaded via Ollama.
    
    Models:
    - qwen2.5:0.5b (default) — Qwen2.5-0.5B-Instruct, Q4_K_M, ~400MB RAM
    - smollm:135m — not yet pulled
    
    Inference: <1s on CPU, 0 LLM tokens (local).
    Falls back to micro-NN if unavailable.
    """
    
    def __init__(self, model: str = "qwen2.5:0.5b", timeout: int = 5):
        self.model = model
        self.timeout = timeout
        self._available = False
        self._last_error = None
        
        try:
            import ollama
            self.ollama = ollama
            # Quick health check
            self.ollama.list()
            self._available = True
        except Exception as e:
            self._last_error = str(e)
    
    @property
    def available(self) -> bool:
        return self._available
    
    def ask(self, system: str, prompt: str, 
            temperature: float = 0.1,
            max_tokens: int = 50) -> Tuple[str, float]:
        """Ask nano-LLM a question. Returns (response, confidence).
        
        Confidence is estimated from response coherence:
        - 1.0 = parseable structured output
        - 0.5 = free text but relevant
        - 0.0 = error or empty
        """
        if not self._available:
            return "", 0.0
        
        try:
            response = self.ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            )
            
            text = response['message']['content'].strip()
            
            # Estimate confidence
            conf = 0.5  # Default
            if not text:
                conf = 0.0
            elif text[0] == '{' or text[0] == '[':
                conf = 0.8  # Structured
            elif len(text.split()) <= 3:
                conf = 0.7  # Short, likely directive
            
            return text, conf
            
        except Exception as e:
            self._last_error = str(e)
            return "", 0.0
    
    def classify_game(self, observation: Dict) -> Tuple[str, float]:
        """Ask nano-LLM to classify the game type from observation.
        
        Returns: (domain_type, confidence)
        """
        system = """You are a game classifier. Output ONLY a JSON object with a single field 'domain'.
Valid domains: movement, rotation, transform, hybrid, unknown.
Example: {"domain": "movement"}"""
        
        prompt = f"""Classify this ARC-AGI-3 game:
Available actions: {observation.get('available_actions', [])}
Grid shape: {observation.get('grid_shape', 'unknown')}
Levels completed: {observation.get('levels_completed', 0)}
Player position: {observation.get('player_pos', 'unknown')}

Domain:"""
        
        text, conf = self.ask(system, prompt, temperature=0.0, max_tokens=20)
        
        try:
            result = json.loads(text)
            return result.get('domain', 'unknown'), conf
        except:
            return 'unknown', conf * 0.5
    
    def propose_action(self, game_state: str, 
                       available_actions: list,
                       recent_history: str = "") -> Tuple[int, float, str]:
        """Ask nano-LLM to propose the next action.
        
        Returns: (action_number, confidence, reasoning)
        """
        system = """You control a game agent. Output ONLY a JSON object:
{"action": <int>, "reasoning": "<brief>"}

Available action numbers: the ONLY valid actions the game accepts.
Choose the action that makes the most progress toward the goal.
Be conservative — prefer safe moves over risky exploration."""
        
        prompt = f"""Game state:
{game_state}

Available actions: {available_actions}

Recent history:
{recent_history if recent_history else 'First move'}

Propose next action:"""
        
        text, conf = self.ask(system, prompt, temperature=0.2, max_tokens=80)
        
        try:
            result = json.loads(text)
            action = int(result.get('action', 1))
            reasoning = result.get('reasoning', '')
            
            # Validate action
            if action not in available_actions:
                # Fall back to first available action
                action = available_actions[0] if available_actions else 1
                conf *= 0.3
            
            return action, conf, reasoning
        except:
            return (available_actions[0] if available_actions else 1), 0.1, "parse error"


class NanoLLMHarness:
    """Safety harness for nano-LLM — validates proposals before execution.
    
    Checks:
    1. Action is in available_actions (deterministic)
    2. Action doesn't hit known walls (if pathfinder available)
    3. Action doesn't repeat a known failure (recent history check)
    """
    
    def __init__(self, nano_llm: NanoLLM):
        self.nano = nano_llm
        self.blocked_actions: Dict[Tuple, int] = {}  # (x,y) -> action that failed
        self.approved_count = 0
        self.blocked_count = 0
    
    def propose_safe(self, game_state: str,
                     available_actions: list,
                     player_pos: Tuple[int, int] = None,
                     recent_history: str = "",
                     wall_cells: set = None) -> Tuple[int, float, str, bool]:
        """Propose action with safety checks.
        
        Returns: (action, confidence, reasoning, is_safe)
        """
        if not self.nano.available:
            return (available_actions[0] if available_actions else 1), 0.0, "nano-LLM offline", False
        
        action, conf, reasoning = self.nano.propose_action(
            game_state, available_actions, recent_history
        )
        
        is_safe = True
        block_reason = ""
        
        # Check 1: Blocked actions at current position
        if player_pos and player_pos in self.blocked_actions:
            blocked = self.blocked_actions[player_pos]
            if action == blocked:
                is_safe = False
                block_reason = f"action {action} previously failed at {player_pos}"
        
        # Check 2: Wall collision (if wall data available)
        if wall_cells and player_pos:
            px, py = player_pos
            # LS20 action mapping: 1=UP,2=DOWN,3=LEFT,4=RIGHT
            dx, dy = {1:(0,-5), 2:(0,5), 3:(-5,0), 4:(5,0)}.get(action, (0,0))
            nx, ny = px + dx, py + dy
            if (nx, ny) in wall_cells:
                is_safe = False
                block_reason = f"action {action} leads to wall at ({nx},{ny})"
        
        if is_safe:
            self.approved_count += 1
        else:
            self.blocked_count += 1
            # Try safe fallback
            for alt_action in available_actions:
                if alt_action != action:
                    dx, dy = {1:(0,-5), 2:(0,5), 3:(-5,0), 4:(5,0)}.get(alt_action, (0,0))
                    nx, ny = (player_pos[0]+dx, player_pos[1]+dy) if player_pos else (0,0)
                    if wall_cells and (nx, ny) in wall_cells:
                        continue
                    return alt_action, 0.3, f"fallback (blocked: {block_reason})", False
        
        return action, conf, reasoning, is_safe
    
    def record_failure(self, position: Tuple[int, int], action: int):
        """Record that an action failed at this position."""
        self.blocked_actions[position] = action
    
    def stats(self) -> Dict:
        return {
            'approved': self.approved_count,
            'blocked': self.blocked_count,
            'blocked_positions': len(self.blocked_actions),
            'nano_available': self.nano.available,
        }
