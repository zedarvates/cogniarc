#!/usr/bin/env python3
"""Train a micro-NN domain classifier for ARC-AGI-3 games.
Pure numpy, no ML deps. Export JSON for Rust inference.

Input: 6 features from game grid
Output: 4 classes (movement, rotation, transform, hybrid)

Pattern: replicate botte-secrete TinyNN → JSON → Rust binary
"""

import numpy as np
import json
import math
import os
from typing import List, Tuple

# ═══════════════════════════════════════════════
# 1. SYNTHETIC TRAINING DATA
# ═══════════════════════════════════════════════

def generate_synthetic_data(n_samples: int = 800) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic ARC-AGI-3 domain classification data.
    
    Features (6):
        0: grid_width (normalized 0-1, 1=64px)
        1: grid_height (normalized 0-1)
        2: num_unique_colors (normalized 0-1, 1=10 colors)
        3: color_entropy (Shannon entropy, 0-1)
        4: spatial_variance (how spread out colors are, 0-1)
        5: connected_components_ratio (objects / total cells)
    
    Labels (4 classes):
        0: movement (actions 1-4, large grid, few objects)
        1: rotation (action 6, square grid, circular patterns)
        2: transform (actions 5+, small grid, many colors)
        3: hybrid (mixed)
    """
    rng = np.random.default_rng(42)
    n_per_class = n_samples // 4
    
    X = np.zeros((n_samples, 6))
    y = np.zeros(n_samples, dtype=int)
    
    for cls in range(4):
        start = cls * n_per_class
        end = start + n_per_class
        
        if cls == 0:  # MOVEMENT — large grid, few colors, low entropy, path-like
            X[start:end, 0] = rng.uniform(0.5, 1.0, n_per_class)      # wide
            X[start:end, 1] = rng.uniform(0.5, 1.0, n_per_class)      # tall
            X[start:end, 2] = rng.uniform(0.1, 0.4, n_per_class)      # 2-4 colors
            X[start:end, 3] = rng.uniform(0.1, 0.5, n_per_class)      # low entropy
            X[start:end, 4] = rng.uniform(0.2, 0.5, n_per_class)      # moderate spread
            X[start:end, 5] = rng.uniform(0.005, 0.05, n_per_class)   # very few objects
            y[start:end] = 0
            
        elif cls == 1:  # ROTATION — square, medium colors, high spatial var, circular
            X[start:end, 0] = rng.uniform(0.3, 0.6, n_per_class)
            X[start:end, 1] = X[start:end, 0] * rng.uniform(0.9, 1.1, n_per_class)  # near-square
            X[start:end, 2] = rng.uniform(0.2, 0.6, n_per_class)      # 3-6 colors
            X[start:end, 3] = rng.uniform(0.3, 0.7, n_per_class)      # medium entropy
            X[start:end, 4] = rng.uniform(0.5, 0.9, n_per_class)      # HIGH spatial var (ring)
            X[start:end, 5] = rng.uniform(0.03, 0.10, n_per_class)    # few objects
            y[start:end] = 1
            
        elif cls == 2:  # TRANSFORM — small grid, many colors, high entropy
            X[start:end, 0] = rng.uniform(0.1, 0.35, n_per_class)     # narrow
            X[start:end, 1] = rng.uniform(0.1, 0.35, n_per_class)     # short
            X[start:end, 2] = rng.uniform(0.5, 1.0, n_per_class)      # 5-10 colors
            X[start:end, 3] = rng.uniform(0.5, 1.0, n_per_class)      # HIGH entropy
            X[start:end, 4] = rng.uniform(0.1, 0.5, n_per_class)      # tight spread
            X[start:end, 5] = rng.uniform(0.05, 0.25, n_per_class)    # many small objects
            y[start:end] = 2
            
        else:  # HYBRID — medium everything, overlapping
            X[start:end, 0] = rng.uniform(0.2, 0.7, n_per_class)
            X[start:end, 1] = rng.uniform(0.2, 0.7, n_per_class)
            X[start:end, 2] = rng.uniform(0.3, 0.7, n_per_class)
            X[start:end, 3] = rng.uniform(0.3, 0.7, n_per_class)
            X[start:end, 4] = rng.uniform(0.2, 0.7, n_per_class)
            X[start:end, 5] = rng.uniform(0.02, 0.12, n_per_class)
            y[start:end] = 3
    
    # Add tiny noise
    X += rng.normal(0, 0.015, X.shape)
    X = np.clip(X, 0, 1)
    
    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]


def get_known_game_features() -> Tuple[np.ndarray, np.ndarray]:
    """Add features from known ARC-AGI-3 games for better training."""
    # These are approximate feature vectors for known games
    # [width/64, height/64, n_colors/10, entropy, spatial_var, objects_ratio]
    known = np.array([
        # MOVEMENT games
        [1.0, 1.0, 0.3, 0.35, 0.45, 0.02],   # LS20 (64x64, 3 wall colors)
        [0.47, 0.47, 0.4, 0.40, 0.50, 0.03],  # TR87 (30x30)
        [0.94, 0.94, 0.5, 0.45, 0.48, 0.04],  # RE86 (60x60)
        [0.78, 0.78, 0.3, 0.30, 0.42, 0.015], # G50T (50x50)
        
        # ROTATION games
        [0.47, 0.47, 0.5, 0.55, 0.75, 0.06],  # VC33 (30x30)
        [0.31, 0.31, 0.4, 0.50, 0.70, 0.05],  # R11L (20x20)
        [0.39, 0.39, 0.6, 0.60, 0.80, 0.07],  # LP85 (25x25)
        
        # TRANSFORM games
        [0.16, 0.16, 0.8, 0.75, 0.35, 0.20],  # FT09 (10x10, pixel transform)
        [0.23, 0.23, 0.7, 0.70, 0.40, 0.15],  # LF52 (15x15)
        
        # HYBRID games
        [0.63, 0.63, 0.6, 0.55, 0.60, 0.08],  # CD82 (40x40, movement+rotation)
        [0.31, 0.31, 0.5, 0.50, 0.55, 0.06],  # M0R0 (20x20)
        [0.47, 0.47, 0.7, 0.65, 0.65, 0.10],  # CN04 (30x30, transform+movement)
    ])
    
    labels = np.array([0,0,0,0,  1,1,1,  2,2,  3,3,3])
    return known, labels


def extract_features_from_grid(grid: np.ndarray) -> np.ndarray:
    """Extract 6 features from a real ARC-AGI-3 grid."""
    h, w = grid.shape
    
    # Normalize dimensions
    w_norm = w / 64.0
    h_norm = h / 64.0
    
    # Unique colors
    unique = np.unique(grid)
    n_colors = len(unique) / 10.0
    
    # Color entropy
    counts = np.bincount(grid.flatten().astype(int), minlength=10)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs + 1e-10)) / np.log2(10)  # normalized
    
    # Spatial variance
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    colored = grid > 0
    if colored.sum() > 0:
        cx = x_idx[colored].mean() / w
        cy = y_idx[colored].mean() / h
        spatial_var = np.sqrt(((x_idx[colored]/w - cx)**2 + (y_idx[colored]/h - cy)**2).mean())
    else:
        spatial_var = 0.0
    
    # Connected components ratio (simplified: count non-zero groups)
    binary = (grid > 0).astype(int)
    # Simple flood fill count
    visited = np.zeros_like(binary, dtype=bool)
    n_objects = 0
    for y in range(h):
        for x in range(w):
            if binary[y, x] and not visited[y, x]:
                n_objects += 1
                # Flood fill
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if 0 <= cy < h and 0 <= cx < w and binary[cy, cx] and not visited[cy, cx]:
                        visited[cy, cx] = True
                        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                            stack.append((cy+dy, cx+dx))
    
    objects_ratio = n_objects / (h * w + 1)
    
    return np.array([w_norm, h_norm, n_colors, entropy, spatial_var, objects_ratio])


# ═══════════════════════════════════════════════
# 2. TINY NEURAL NETWORK (pure numpy)
# ═══════════════════════════════════════════════

class TinyNN:
    """Minimal feedforward neural network. Pure numpy, no ML deps."""
    
    def __init__(self, layers: List[int], activations: List[str]):
        self.layers = layers
        self.activations = activations
        self.weights = []
        self.biases = []
        
        # Xavier initialization
        rng = np.random.default_rng(42)
        for i in range(len(layers) - 1):
            fan_in = layers[i]
            fan_out = layers[i+1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = rng.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)
    
    def _activate(self, x: np.ndarray, name: str) -> np.ndarray:
        if name == 'relu':
            return np.maximum(0, x)
        elif name == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
        elif name == 'softmax':
            ex = np.exp(x - np.max(x))
            return ex / ex.sum()
        elif name == 'linear':
            return x
        return x
    
    def _activate_derivative(self, x: np.ndarray, name: str) -> np.ndarray:
        if name == 'relu':
            return (x > 0).astype(float)
        elif name == 'sigmoid':
            s = self._activate(x, 'sigmoid')
            return s * (1 - s)
        elif name == 'linear':
            return np.ones_like(x)
        return np.ones_like(x)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Forward pass. Returns (output, pre_activations, post_activations)."""
        pre_acts = []
        post_acts = [x]
        
        for i in range(len(self.weights)):
            z = post_acts[-1] @ self.weights[i] + self.biases[i]
            pre_acts.append(z)
            a = self._activate(z, self.activations[i])
            post_acts.append(a)
        
        return post_acts[-1], pre_acts, post_acts
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Inference only — returns class probabilities."""
        out, _, _ = self.forward(x)
        return out
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              epochs: int = 2000, lr: float = 0.01, verbose: bool = True):
        """Train with SGD + MSE loss."""
        n = len(X)
        y_onehot = np.eye(self.layers[-1])[y]
        
        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(n)
            X_shuf = X[idx]
            y_shuf = y_onehot[idx]
            
            total_loss = 0
            for i in range(n):
                xi = X_shuf[i]
                yi = y_shuf[i]
                
                # Forward
                out, pre_acts, post_acts = self.forward(xi)
                
                # Loss
                error = out - yi
                total_loss += np.mean(error ** 2)
                
                # Backprop
                delta = error * self._activate_derivative(pre_acts[-1], self.activations[-1])
                
                for l in range(len(self.weights) - 1, -1, -1):
                    # Gradient
                    dw = np.outer(post_acts[l], delta)
                    db = delta
                    
                    # Update
                    self.weights[l] -= lr * dw
                    self.biases[l] -= lr * db
                    
                    # Propagate delta
                    if l > 0:
                        delta = (delta @ self.weights[l].T) * self._activate_derivative(
                            pre_acts[l-1], self.activations[l-1]
                        )
            
            if verbose and epoch % 200 == 0:
                print(f"  Epoch {epoch:4d}: loss={total_loss/n:.6f}")
    
    def export_json(self) -> dict:
        """Export weights in botte-secrete format for Rust inference."""
        return {
            "layers": self.layers,
            "weights": [np.concatenate(w.T).tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "activations": self.activations,
            "labels": ["movement", "rotation", "transform", "hybrid"],
        }
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = np.array([np.argmax(self.predict(x)) for x in X])
        return np.mean(preds == y)


# ═══════════════════════════════════════════════
# 3. TRAINING
# ═══════════════════════════════════════════════

def main():
    print("🧠 Training Domain Classifier micro-NN")
    print("=" * 50)
    
    # Generate data — mix synthetic + known games
    print("\n📊 Generating training data...")
    X_syn, y_syn = generate_synthetic_data(600)
    X_known, y_known = get_known_game_features()
    
    # Combine
    X_train = np.vstack([X_syn[:500], X_known])
    y_train = np.concatenate([y_syn[:500], y_known])
    X_test = X_syn[500:]
    y_test = y_syn[500:]
    
    print(f"   Train: {len(X_train)} samples (500 synth + {len(X_known)} known games)")
    print(f"   Test: {len(X_test)} samples (synth only)")
    print(f"   Features: [width, height, n_colors, entropy, spatial_var, objects]")
    print(f"   Classes: {dict(enumerate(['movement','rotation','transform','hybrid']))}")
    
    # Create and train model
    print("\n🏗️  Model: [6, 12, 4] with [relu, softmax]")
    model = TinyNN([6, 12, 4], ["relu", "softmax"])
    
    print("\n🔄 Training...")
    model.train(X_train, y_train, epochs=3000, lr=0.015)
    
    # Evaluate
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    print(f"\n📈 Accuracy: train={train_acc:.1%}, test={test_acc:.1%}")
    
    # Test on known games
    print("\n🧪 Testing on known ARC-AGI-3 games:")
    labels = ["movement", "rotation", "transform", "hybrid"]
    games = [
        ("LS20", 0), ("TR87", 0), ("VC33", 1), ("R11L", 1),
        ("FT09", 2), ("LF52", 2), ("CD82", 3), ("CN04", 3),
    ]
    
    all_correct = True
    for i, (name, expected) in enumerate(games):
        features = X_known[i]
        pred = model.predict(features)
        pred_class = np.argmax(pred)
        ok = "✅" if pred_class == expected else "❌"
        if pred_class != expected:
            all_correct = False
        print(f"   {ok} {name:6s} → {labels[pred_class]:10s} (expected {labels[expected]}) conf={pred[pred_class]:.3f}")
    
    if all_correct:
        print(f"\n   🎯 All known games classified correctly!")
    else:
        correct = sum(1 for i, (_, e) in enumerate(games) if np.argmax(model.predict(X_known[i])) == e)
        print(f"\n   📊 {correct}/{len(games)} known games correct")
    
    # Export
    print("\n💾 Exporting weights...")
    data = model.export_json()
    out_path = os.path.join(os.path.dirname(__file__), "domain_classifier.json")
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    size_kb = os.path.getsize(out_path) / 1024
    print(f"   Saved: {out_path} ({size_kb:.1f} KB)")
    print(f"   Layers: {data['layers']}")
    print(f"   Weights: {[len(w) for w in data['weights']]}")
    
    print("\n✅ Domain Classifier trained and exported!")


if __name__ == "__main__":
    main()
