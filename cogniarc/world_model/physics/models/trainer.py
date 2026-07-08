"""
World Model Trainer — Small model that learns physics state transitions.
Learns to predict next state from current state using ground truth data.
Designed for small LLMs (few million params) with approximate reasoning.
"""

import numpy as np
import json
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import sys
sys.path.insert(0, '/home/redgamer/projects/world-model-tool')
from simulator.physics import PhysicsWorld, Vec2, SCENARIOS, create_ramp_scenario, generate_training_data


@dataclass 
class WorldModelConfig:
    input_size: int = 20      # Number of state variables
    hidden_size: int = 64     # Small hidden layer (for small LLM)
    num_layers: int = 2       # Very shallow
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    train_split: float = 0.8
    
    # Domain-specific knowledge (approximations)
    gravity_direction: int = -1  # Y-axis negative
    collision_restitution: float = 0.3
    max_velocity: float = 50.0   # Clamp velocities (game-style)


class SimpleWorldModel:
    """
    Minimal neural network that learns (state → next_state) mapping.
    Uses MLP with residual connections for small parameter count.
    Approximate reasoning like a video game physics engine.
    """
    
    def __init__(self, config: WorldModelConfig):
        self.config = config
        self.params = self._init_params()
        self.trained = False
        self.training_history = []
    
    def _init_params(self):
        """Initialize weights for 2-layer residual network (small initialization for stability)"""
        rng = np.random.RandomState(42)
        D = self.config.input_size
        H = self.config.hidden_size
        
        # Small initialization to prevent NaN
        scale = 0.01
        w1 = rng.randn(D, H) * scale
        b1 = np.zeros(H)
        w2 = rng.randn(H, H) * scale
        b2 = np.zeros(H)
        w3 = rng.randn(H, D) * scale
        b3 = np.zeros(D)
        
        # Domain priors: slight gravity bias on vy
        b3[3::4] += self.config.gravity_direction * 0.001
        
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}
    
    def _normalize(self, X: np.ndarray, Y: np.ndarray = None):
        """Normalize states to [-1, 1] range"""
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0) + 1e-10
        X_norm = (X - self._x_mean) / self._x_std
        if Y is not None:
            Y_norm = (Y - self._x_mean) / self._x_std
            return X_norm, Y_norm
        return X_norm
    
    def _denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        return X_norm * self._x_std + self._x_mean
    
    def _relu(self, x): return np.maximum(0, x)
    def _relu_deriv(self, x): return (x > 0).astype(float)
    
    def forward(self, x_norm: np.ndarray) -> np.ndarray:
        """Predict next state from normalized current state. Returns normalized prediction."""
        p = self.params
        z1 = x_norm @ p["w1"] + p["b1"]
        a1 = self._relu(z1)
        z2 = a1 @ p["w2"] + p["b2"]
        a2 = self._relu(z2) + a1
        z3 = a2 @ p["w3"] + p["b3"]
        return x_norm + z3  # predict delta in normalized space
    
    def train_batch(self, X: np.ndarray, Y: np.ndarray, lr: float) -> float:
        """One training step, returns loss"""
        N = X.shape[0]
        p = self.params
        
        # Forward
        z1 = X @ p["w1"] + p["b1"]
        a1 = self._relu(z1)
        z2 = a1 @ p["w2"] + p["b2"]
        a2 = self._relu(z2) + a1  # residual
        z3 = a2 @ p["w3"] + p["b3"]
        pred = X + z3  # predict delta
        
        # Loss
        diff = pred - Y
        loss = np.mean(diff ** 2)
        
        # Backward (manual for speed)
        d_pred = 2 * diff / N  # dL/dpred
        # Layer 3
        d_z3 = d_pred
        p["w3"] -= lr * a2.T @ d_z3
        p["b3"] -= lr * d_z3.sum(axis=0)
        # Layer 2 (residual)
        d_a2 = d_z3 @ p["w3"].T
        d_a2_reduced = d_a2  # same as d_a2 since residual is additive
        d_z2 = d_a2_reduced * self._relu_deriv(z2)
        p["w2"] -= lr * a1.T @ d_z2
        p["b2"] -= lr * d_z2.sum(axis=0)
        # Layer 1
        d_a1 = d_z2 @ p["w2"].T + d_a2_reduced  # + residual gradient
        d_z1 = d_a1 * self._relu_deriv(z1)
        p["w1"] -= lr * X.T @ d_z1
        p["b1"] -= lr * d_z1.sum(axis=0)
        
        return float(loss)
    
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray,
            X_val: np.ndarray = None, Y_val: np.ndarray = None,
            verbose: bool = True, lr_decay: float = 0.99) -> dict:
        """Train the world model with normalization"""
        # Normalize data
        X_train_norm, Y_train_norm = self._normalize(X_train, Y_train)
        X_val_norm = self._normalize(X_val) if X_val is not None else None
        Y_val_norm = (Y_val - self._x_mean) / self._x_std if Y_val is not None else None
        
        N = X_train_norm.shape[0]
        lr = self.config.learning_rate
        
        for epoch in range(self.config.epochs):
            idx = np.random.permutation(N)
            total_loss = 0
            batches = 0
            
            for i in range(0, N, self.config.batch_size):
                batch_idx = idx[i:i + self.config.batch_size]
                X_batch = X_train_norm[batch_idx]
                Y_batch = Y_train_norm[batch_idx]
                loss = self.train_batch(X_batch, Y_batch, lr)
                total_loss += loss
                batches += 1
            
            lr *= lr_decay  # Decay learning rate
            avg_loss = total_loss / batches
            
            # Denormalize to get real MSE
            val_loss = None
            if X_val_norm is not None and self._x_std is not None:
                pred_norm = self.forward(X_val_norm)
                pred = self._denormalize(pred_norm)
                val_loss = float(np.mean((pred - Y_val) ** 2))
            
            self.training_history.append({"epoch": epoch, "train_loss": avg_loss, "val_loss": val_loss})
            
            if verbose and epoch % 20 == 0:
                info = f"Epoch {epoch}: train_loss={avg_loss:.6f}"
                if val_loss: info += f" val_mse={val_loss:.6f}"
                print(info)
        
        self.trained = True
        return {"final_loss": avg_loss, "history": self.training_history}
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict next state (denormalized output)"""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_norm = self._normalize(x)
        pred_norm = self.forward(x_norm)
        pred = self._denormalize(pred_norm)
        # Clamp velocities
        out = np.array(pred)
        for i in range(0, out.shape[1], 4):
            out[:, i+2:] = np.clip(out[:, i+2:], -self.config.max_velocity, self.config.max_velocity)
        return out
    
    def predict_multi(self, x: np.ndarray, steps: int = 10) -> List[np.ndarray]:
        """Predict multiple steps ahead (autoregressive)"""
        predictions = []
        current = x.copy()
        for _ in range(steps):
            current = self.predict(current.reshape(1, -1)).reshape(-1)
            predictions.append(current.copy())
        return predictions
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                "config_dict": {
                    "input_size": self.config.input_size,
                    "hidden_size": self.config.hidden_size,
                    "num_layers": self.config.num_layers,
                    "learning_rate": self.config.learning_rate,
                    "max_velocity": self.config.max_velocity,
                    "gravity_direction": self.config.gravity_direction,
                },
                "params": self.params, "trained": self.trained,
                "history": self.training_history,
                "x_mean": getattr(self, '_x_mean', None),
                "x_std": getattr(self, '_x_std', None)
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "SimpleWorldModel":
        with open(path, 'rb') as f:
            data = pickle.load(f)
        config = WorldModelConfig(**data["config_dict"])
        model = cls(config)
        model.params = data["params"]
        model.trained = data["trained"]
        model.training_history = data.get("history", [])
        if data.get("x_mean") is not None:
            model._x_mean = np.array(data["x_mean"])
            model._x_std = np.array(data["x_std"])
        return model


def train_on_scenario(scenario_name: str, steps: int = 5000, epochs: int = 200):
    """Train a world model on a specific physics scenario"""
    print(f"\n{'='*50}")
    print(f"Training World Model: {scenario_name}")
    print(f"{'='*50}")
    
    if scenario_name in SCENARIOS:
        world = SCENARIOS[scenario_name]()
    else:
        world = create_ramp_scenario()
    
    state_size = world.state_size()
    
    print(f"Generating {steps} frames of ground truth data...")
    data = generate_training_data(world, steps)
    X = np.array([d[0] for d in data])
    Y = np.array([d[1] for d in data])
    
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]
    
    print(f"Training: {len(X_train)} samples, Validation: {len(X_val)} samples")
    print(f"State size: {state_size} variables")
    
    config = WorldModelConfig(input_size=state_size, hidden_size=64, num_layers=2,
                              epochs=epochs, learning_rate=0.0001)
    model = SimpleWorldModel(config)
    results = model.fit(X_train, Y_train, X_val, Y_val)
    
    print(f"Final validation loss: {results['final_loss']:.6f}")
    
    test_state = X_val[0]
    actual = Y_val[0]
    predicted = model.predict(test_state.reshape(1, -1)).reshape(-1)
    mse = np.mean((predicted - actual) ** 2)
    print(f"Single step MSE: {mse:.6f}")
    
    predictions = model.predict_multi(test_state, steps=30)
    last_actual = Y_val[min(29, len(Y_val)-1)]
    last_pred = predictions[min(29, len(predictions)-1)]
    multi_mse = np.mean((last_pred - last_actual) ** 2)
    print(f"30-step MSE: {multi_mse:.6f}")
    
    return model


if __name__ == "__main__":
    model = train_on_scenario("ramp", steps=3000, epochs=150)
    model.save("/home/redgamer/projects/world-model-tool/models/ramp_model.pkl")
    print("Saved: models/ramp_model.pkl")
