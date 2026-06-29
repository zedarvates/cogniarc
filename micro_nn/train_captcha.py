#!/usr/bin/env python3
"""Train a micro-NN to classify CAPTCHA types from screenshots.
Pure numpy, no ML deps. Export JSON for Rust inference.

Architecture: 64 → 32 → 16 → 6 (relu×2 + softmax)
Input:  8×8 downsampled grayscale screenshot (64 features)
Output: reCAPTCHA_v2 | hCaptcha | Turnstile | text | math | none
"""

import numpy as np
import json
import os


class TinyNN:
    """Mini-batch SGD neural network (pure numpy)."""
    
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
        if name == 'softmax':
            ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return ex / ex.sum(axis=-1, keepdims=True)
        return x
    
    def _activate_derivative(self, x, name):
        if name == 'relu': return (x > 0).astype(float)
        return np.ones_like(x)
    
    def forward(self, x):
        a = x
        for i in range(len(self.weights)):
            a = self._activate(a @ self.weights[i] + self.biases[i], self.activations[i])
        return a
    
    def predict(self, x):
        return self.forward(x.reshape(1, -1))[0]
    
    def train(self, X, y, epochs=300, lr=0.005, batch_size=32, verbose=True):
        n = len(X)
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            Xs, ys = X[idx], y[idx]
            total_loss = 0
            n_batches = 0
            
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb, yb = Xs[start:end], ys[start:end]
                m = len(Xb)
                
                # Forward
                a = Xb
                activations = [a]
                pre_acts = []
                for l in range(len(self.weights)):
                    z = a @ self.weights[l] + self.biases[l]
                    pre_acts.append(z)
                    a = self._activate(z, self.activations[l])
                    activations.append(a)
                
                # Cross-entropy loss
                eps = 1e-10
                loss = -np.mean(np.sum(yb * np.log(activations[-1] + eps), axis=1))
                total_loss += loss
                n_batches += 1
                
                # Backprop
                delta = activations[-1] - yb
                for l in range(len(self.weights)-1, -1, -1):
                    dw = activations[l].T @ delta / m
                    db = delta.mean(axis=0)
                    dw = np.clip(dw, -0.5, 0.5)
                    db = np.clip(db, -0.5, 0.5)
                    self.weights[l] -= lr * dw
                    self.biases[l] -= lr * db
                    if l > 0:
                        delta = (delta @ self.weights[l].T) * self._activate_derivative(pre_acts[l-1], self.activations[l-1])
            
            if verbose and epoch % 50 == 0:
                print(f"  Epoch {epoch:3d}: loss={total_loss/n_batches:.6f}")
    
    def accuracy(self, X, y):
        return np.mean(self.forward(X).argmax(axis=1) == y.argmax(axis=1))
    
    def export_json(self):
        return {
            "layers": self.layers,
            "weights": [np.concatenate(w.T).tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "activations": self.activations,
        }


# CAPTCHA types
TYPES = [
    "recaptcha_v2",   # Google reCAPTCHA v2 (checkbox + image grid)
    "hcaptcha",       # hCaptcha
    "turnstile",      # Cloudflare Turnstile
    "text_captcha",   # Distorted text
    "math_captcha",   # Simple math (3+5=?)
    "none",           # No CAPTCHA detected
]


def generate_captcha_data(n_samples: int = 3000) -> tuple:
    """Generate synthetic CAPTCHA screenshot features.
    
    Each type has a distinct visual signature:
    - recaptcha_v2: blue/white theme, checkbox pattern
    - hcaptcha: dark theme, hexagonal logo
    - turnstile: minimal, spinning icon
    - text: random pixel noise, high contrast
    - math: digits + operator symbols
    - none: uniform, no patterns
    """
    rng = np.random.default_rng(42)
    X = []
    y = []
    
    for _ in range(n_samples):
        img = np.zeros((8, 8), dtype=np.float32)
        type_idx = rng.integers(0, 6)
        
        if type_idx == 0:  # recaptcha_v2 — blue/white gradient + checkbox center
            img[:, :] = rng.normal(0.7, 0.1, (8, 8))
            img[3:5, 3:5] = rng.normal(0.3, 0.05, (2, 2))  # checkbox
            img[0:2, 0:8] = rng.normal(0.5, 0.1, (2, 8))  # blue bar top
        elif type_idx == 1:  # hcaptcha — dark theme, hexagonal pattern
            img[:, :] = rng.normal(0.2, 0.1, (8, 8))  # dark bg
            img[2:6, 2:6] = rng.normal(0.6, 0.15, (4, 4))  # logo center
            img[1:3, 4:6] = rng.normal(0.4, 0.1, (2, 2))  # hex edge
            img[5:7, 4:6] = rng.normal(0.4, 0.1, (2, 2))
        elif type_idx == 2:  # turnstile — minimal, rotating pattern
            img[:, :] = rng.normal(0.95, 0.03, (8, 8))  # white bg
            # Spinning arc
            for i in range(3):
                angle = i * 2.09
                cx, cy = 3 + int(2*np.cos(angle)), 3 + int(2*np.sin(angle))
                if 0 <= cx < 8 and 0 <= cy < 8:
                    img[cy, cx] = rng.normal(0.2, 0.1)
        elif type_idx == 3:  # text_captcha — high contrast noise + letter shapes
            img[:, :] = rng.uniform(0, 1, (8, 8))  # random noise
            # Add horizontal line patterns (letter strokes)
            img[2:4, 1:7] = rng.normal(0.2, 0.1, (2, 6))
            img[5:7, 1:7] = rng.normal(0.8, 0.1, (2, 6))  # high contrast bottom
            img[1:7, 3:5] = rng.normal(0.3, 0.15, (6, 2))  # vertical stroke
        elif type_idx == 4:  # math_captcha — digit patterns
            img[:, :] = rng.normal(0.9, 0.05, (8, 8))  # light bg
            img[1:7, 1:4] = rng.normal(0.1, 0.1, (6, 3))  # left digit dark
            img[1:7, 5:7] = rng.normal(0.1, 0.1, (6, 2))  # right digit dark
            img[3:5, 4] = rng.normal(0.4, 0.1, (2,))  # operator
        else:  # none — uniform noise
            img[:, :] = rng.normal(0.5, 0.02, (8, 8))
        
        X.append(img.flatten())
        y_vec = np.zeros(6)
        y_vec[type_idx] = 1.0
        y.append(y_vec)
    
    # Normalize
    X_arr = np.array(X, dtype=np.float32)
    X_arr = np.clip(X_arr, 0, 1)
    return X_arr, np.array(y, dtype=np.float32)


def main():
    print("🧩 Training CAPTCHA Type Classifier")
    print("=" * 50)
    print(f"  Types: {TYPES}")
    
    print("\n📊 Generating synthetic training data...")
    X, y = generate_captcha_data(4000)
    
    # Split
    n_train = int(len(X) * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    print(f"\n🏗️  Model: [64, 32, 16, 6] relu + softmax")
    model = TinyNN([64, 32, 16, 6], ["relu", "relu", "softmax"])
    
    print("🔄 Training...")
    model.train(X_train, y_train, epochs=300, lr=0.005, batch_size=32)
    
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    print(f"\n📈 Accuracy: train={train_acc:.1%}, test={test_acc:.1%}")
    
    # Per-class accuracy
    print("\n📊 Per-class test accuracy:")
    for i, name in enumerate(TYPES):
        mask = y_test.argmax(axis=1) == i
        if mask.sum() > 0:
            acc = model.accuracy(X_test[mask], y_test[mask])
            print(f"  {name:15s}: {acc:.1%} ({mask.sum()} samples)")
    
    # Export
    data = model.export_json()
    data["types"] = TYPES
    out = os.path.join(os.path.dirname(__file__), "captcha_classifier.json")
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)
    
    size_kb = os.path.getsize(out) / 1024
    print(f"\n💾 Exported: {out} ({size_kb:.1f} KB)")
    print(f"   Architecture: {data['layers']}")
    print(f"   Types: {TYPES}")
    print(f"\n✅ CAPTCHA classifier trained!")
    print(f"   Use with existing Rust binary: ./domain-classifier captcha_classifier.json")


if __name__ == "__main__":
    main()
