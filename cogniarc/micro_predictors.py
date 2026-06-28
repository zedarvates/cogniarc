"""Micro-NN predictors for ScientistAgent — instant action success prediction.
Replaces V-JEPA world model queries (6s) with <1ms numpy inference.
"""

import json
import os
import numpy as np
from typing import Optional, Tuple


class ActionPredictor:
    """Predicts whether an action will succeed based on 8 game state features.
    
    Architecture: 8 → 16 → 1 (relu + sigmoid)
    Input: [dx, dy, dist, action_norm, wall_between, stagnation, near_target, steps]
    Output: success probability 0-1
    """
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), '..', 'micro_nn', 'action_predictor.json'
            )
        
        self._loaded = False
        if os.path.exists(model_path):
            try:
                with open(model_path) as f:
                    data = json.load(f)
                self.weights = [np.array(w).reshape(data['layers'][i+1], data['layers'][i])
                               for i, w in enumerate(data['weights'])]
                self.biases = [np.array(b) for b in data['biases']]
                self.activations = data['activations']
                self._loaded = True
            except Exception as e:
                print(f"[ActionPredictor] Failed to load model: {e}")
    
    @property
    def available(self) -> bool:
        return self._loaded
    
    def _activate(self, x: np.ndarray, name: str) -> np.ndarray:
        if name == 'relu':
            return np.maximum(0, x)
        if name == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        if name == 'softmax':
            ex = np.exp(x - np.max(x))
            return ex / ex.sum()
        return x
    
    def predict(self, features: np.ndarray) -> float:
        """Predict action success probability from 8 features.
        
        Args:
            features: [dx, dy, dist, action_norm, wall_between, stagnation, near_target, steps]
        
        Returns:
            success probability 0-1
        """
        if not self._loaded:
            return 0.5  # Neutral default
        
        x = features.reshape(1, -1)
        for i in range(len(self.weights)):
            x = x @ self.weights[i].T + self.biases[i]
            x = self._activate(x, self.activations[i])
        
        return float(x[0, 0])
    
    def predict_action(self, player_pos: Tuple[int, int], target_pos: Tuple[int, int],
                       action: int, wall_colors: set, grid: np.ndarray,
                       stagnation: int = 0, steps: int = 0) -> float:
        """High-level: predict if an action will succeed given game state.
        
        Args:
            player_pos: (px, py)
            target_pos: (tx, ty)  
            action: 1=right, 2=down, 3=left, 4=up
            wall_colors: set of wall color values
            grid: full grid (for wall detection)
            stagnation: current stagnation counter
            steps: steps taken so far
        
        Returns:
            success probability 0-1
        """
        px, py = player_pos
        tx, ty = target_pos
        
        # Distance features
        dx = (tx - px) / max(abs(tx - px) + abs(ty - py), 1)
        dy = (ty - py) / max(abs(tx - px) + abs(ty - py), 1)
        dist = np.sqrt((tx-px)**2 + (ty-py)**2) / 100.0  # normalize
        
        # Wall between player and target?
        wall_between = 0
        if grid is not None:
            # Check a few points along the line
            for t in [0.25, 0.5, 0.75]:
                cx = int(px + (tx - px) * t)
                cy = int(py + (ty - py) * t)
                if 0 <= cy < grid.shape[0] and 0 <= cx < grid.shape[1]:
                    if grid[cy, cx] in wall_colors:
                        wall_between = 1
                        break
        
        # Near target?
        near_target = 1 if abs(px - tx) + abs(py - ty) <= 2 else 0
        
        features = np.array([
            dx, dy, min(dist, 1.0),
            (action - 1) / 3.0,  # normalize 1-4 → 0-1
            wall_between,
            min(stagnation / 15.0, 1.0),
            near_target,
            min(steps / 200.0, 1.0),
        ])
        
        return self.predict(features)


class DomainPredictor:
    """Predicts ARC-AGI-3 game domain from 6 grid features.
    
    Architecture: 6 → 12 → 4 (relu + softmax)
    Output: [movement, rotation, transform, hybrid] probabilities
    """
    
    DOMAINS = ["movement", "rotation", "transform", "hybrid"]
    
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), '..', 'micro_nn', 'domain_classifier.json'
            )
        
        self._loaded = False
        if os.path.exists(model_path):
            try:
                with open(model_path) as f:
                    data = json.load(f)
                self.weights = [np.array(w).reshape(data['layers'][i+1], data['layers'][i])
                               for i, w in enumerate(data['weights'])]
                self.biases = [np.array(b) for b in data['biases']]
                self.activations = data['activations']
                self._loaded = True
            except Exception as e:
                print(f"[DomainPredictor] Failed: {e}")
    
    @property
    def available(self) -> bool:
        return self._loaded
    
    def _activate(self, x, name):
        if name == 'relu': return np.maximum(0, x)
        if name == 'softmax':
            ex = np.exp(x - np.max(x))
            return ex / ex.sum()
        return x
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Predict domain from 6 grid features.
        
        Returns:
            (domain_name, confidence)
        """
        if not self._loaded:
            return "unknown", 0.0
        
        x = features.reshape(1, -1)
        for i in range(len(self.weights)):
            x = x @ self.weights[i].T + self.biases[i]
            x = self._activate(x, self.activations[i])
        
        probs = x[0]
        idx = np.argmax(probs)
        return self.DOMAINS[idx], float(probs[idx])
