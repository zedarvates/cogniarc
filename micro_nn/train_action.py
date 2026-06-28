#!/usr/bin/env python3
"""Train a micro-NN to predict action success in ARC-AGI-3.
Replaces V-JEPA world model queries (6s) with 5µs Rust inference.

Input: 8 features from game state + action
Output: success probability (0-1, binary classifier)
"""

import numpy as np
import json
import os
from typing import Tuple

class TinyNN:
    """Minimal feedforward NN — same as botte-secrete pattern."""
    
    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        rng = np.random.default_rng(42)
        for i in range(len(layers) - 1):
            fan_in, fan_out = layers[i], layers[i+1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))
    
    def _activate(self, x, name):
        if name == 'relu': return np.maximum(0, x)
        if name == 'sigmoid': return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))
        if name == 'softmax':
            ex = np.exp(x - np.max(x))
            return ex / ex.sum()
        return x
    
    def _activate_derivative(self, x, name):
        if name == 'relu': return (x > 0).astype(float)
        if name == 'sigmoid':
            s = self._activate(x, 'sigmoid')
            return s * (1 - s)
        return np.ones_like(x)
    
    def forward(self, x):
        pre_acts, post_acts = [], [x]
        for i in range(len(self.weights)):
            z = post_acts[-1] @ self.weights[i] + self.biases[i]
            pre_acts.append(z)
            post_acts.append(self._activate(z, self.activations[i]))
        return post_acts[-1], pre_acts, post_acts
    
    def predict(self, x):
        out, _, _ = self.forward(x)
        return out
    
    def train(self, X, y, epochs=500, lr=0.01, batch_size=32, verbose=True):
        """Train with mini-batch SGD (vectorized for speed)."""
        n = len(X)
        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx].reshape(-1, 1)
            
            total_loss = 0
            n_batches = 0
            
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb = X_shuf[start:end]
                yb = y_shuf[start:end]
                m = len(Xb)
                
                # Forward (vectorized over batch)
                a = Xb  # activations[0]
                activations = [a]
                pre_acts = []
                
                for l in range(len(self.weights)):
                    z = a @ self.weights[l] + self.biases[l]
                    pre_acts.append(z)
                    a = self._activate(z, self.activations[l])
                    activations.append(a)
                
                # Loss
                error = activations[-1] - yb
                total_loss += np.mean(error ** 2)
                n_batches += 1
                
                # Backprop
                delta = error * self._activate_derivative(pre_acts[-1], self.activations[-1])
                
                for l in range(len(self.weights)-1, -1, -1):
                    dw = activations[l].T @ delta / m
                    db = delta.mean(axis=0)
                    
                    self.weights[l] -= lr * dw
                    self.biases[l] -= lr * db
                    
                    if l > 0:
                        delta = (delta @ self.weights[l].T) * self._activate_derivative(
                            pre_acts[l-1], self.activations[l-1]
                        )
            
            if verbose and epoch % 100 == 0:
                print(f"  Epoch {epoch:4d}: loss={total_loss/n_batches:.6f}")
    
    def accuracy(self, X, y, threshold=0.5):
        preds = np.array([self.predict(x)[0] for x in X])
        binary = (preds >= threshold).astype(int)
        return np.mean(binary == y.astype(int))
    
    def export_json(self):
        return {
            "layers": self.layers,
            "weights": [np.concatenate(w.T).tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "activations": self.activations,
        }


def generate_action_data(n_samples: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic action success data.
    
    Features (8):
        0: dx_to_target (normalized, -1 to 1)
        1: dy_to_target (normalized, -1 to 1)
        2: distance_to_target (normalized, 0 to 1)
        3: action (1-4 encoded as 0-1)
        4: wall_between (0 or 1, is there a wall color between player and target?)
        5: stagnation_count (normalized, 0 to 1)
        6: is_near_target (0 or 1, adjacent to target?)
        7: steps_taken (normalized, 0 to 1)
    
    Output:
        success probability (0 or 1)
    
    Rules:
        - Moving TOWARD target on same axis = success
        - Moving into wall = failure
        - At target = success (already there)
        - Far away + wall between = failure
        - Stagnation > 5 = likely failure
    """
    rng = np.random.default_rng(42)
    
    X = np.zeros((n_samples, 8))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Random starting position
        px, py = rng.uniform(0, 1, 2)
        tx, ty = rng.uniform(0, 1, 2)
        
        dx = tx - px
        dy = ty - py
        dist = np.sqrt(dx**2 + dy**2)
        
        action = rng.integers(1, 5)  # 1=right, 2=down, 3=left, 4=up
        wall = rng.choice([0, 1], p=[0.7, 0.3])
        stagnation = rng.integers(0, 15)
        near_target = 1 if dist < 0.15 else 0
        steps = rng.integers(0, 200)
        
        # Determine success
        success = 0
        
        if near_target and action in [1,2,3,4]:
            # Moving while at/near target — often succeeds
            success = 1 if rng.random() > 0.3 else 0
        elif wall and stagnation > 5:
            success = 0  # Wall + stuck = failure
        elif wall:
            success = 0  # Wall blocks
        elif stagnation > 8:
            success = 0  # Too stuck
        elif action == 1 and dx > 0:
            success = 1  # Moving right toward target
        elif action == 2 and dy > 0:
            success = 1  # Moving down toward target
        elif action == 3 and dx < 0:
            success = 1  # Moving left toward target
        elif action == 4 and dy < 0:
            success = 1  # Moving up toward target
        else:
            success = 0 if rng.random() > 0.3 else 1  # Occasionally succeeds anyway
        
        X[i] = [
            dx, dy, dist,
            (action - 1) / 3.0,  # normalize 1-4 → 0-1
            wall,
            stagnation / 15.0,
            near_target,
            steps / 200.0,
        ]
        y[i] = success
    
    return X, y


def main():
    print("🎯 Training Action Success Predictor micro-NN")
    print("=" * 50)
    
    X, y = generate_action_data(2000)
    
    # Split
    n_train = 1500
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    
    print(f"\n📊 Features (8): [dx, dy, dist, action, wall, stagnation, near_target, steps]")
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Success rate: {y.mean():.1%}")
    
    # Model: 8 → 16 → 1 (sigmoid output for probability)
    model = TinyNN([8, 16, 1], ["relu", "sigmoid"])
    
    print(f"\n🏗️  Model: [8, 16, 1] relu + sigmoid")
    print("🔄 Training (vectorized, 500 epochs)...")
    model.train(X_train, y_train, epochs=500, lr=0.01, batch_size=64)
    
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    print(f"\n📈 Accuracy: train={train_acc:.1%}, test={test_acc:.1%}")
    
    # Test specific scenarios
    print("\n🧪 Scenario tests:")
    scenarios = [
        ("Move right toward target", [0.3, 0.0, 0.3, 0.0, 0, 0.0, 0, 0.1]),
        ("Move right away from target", [-0.3, 0.0, 0.3, 0.0, 0, 0.0, 0, 0.1]),
        ("Move into wall", [0.3, 0.0, 0.3, 0.0, 1, 0.5, 0, 0.3]),
        ("Near target, move right", [0.05, 0.02, 0.06, 0.0, 0, 0.0, 1, 0.1]),
        ("Stuck + wall", [0.2, 0.1, 0.25, 0.33, 1, 0.8, 0, 0.5]),
    ]
    
    for desc, features in scenarios:
        prob = model.predict(np.array(features))[0]
        verdict = "✅" if prob > 0.5 else "❌"
        print(f"   {verdict} {desc:35s} → {prob:.3f}")
    
    # Export
    data = model.export_json()
    out = os.path.join(os.path.dirname(__file__), "action_predictor.json")
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n💾 Exported: {out} ({os.path.getsize(out)/1024:.1f} KB)")
    print(f"   Layers: {data['layers']}")


if __name__ == "__main__":
    main()
