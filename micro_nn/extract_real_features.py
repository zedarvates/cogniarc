#!/usr/bin/env python3
"""Extract real features from ARC-AGI-3 game grids and train the domain classifier.
Uses actual game environments to get ground-truth features + domains.
"""

import numpy as np
import json
import sys
import os

# Add cogniarc to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogniarc.scientist_agent import ScientistAgent
from micro_nn.train_domain import TinyNN, extract_features_from_grid

# Known games with their domains
GAMES = {
    # Movement games (actions 1-4)
    'ls20-9607627b': 0,
    'tr87-3592c1ab': 0,
    're86-7f4a1b2c': 0,
    'g50t-8e3d9f1a': 0,
    
    # Rotation games (action 6)
    'vc33-4b7c2a1d': 1,
    'r11l-9f3e8d2c': 1,
    'lp85-6a4b7c3d': 1,
    
    # Transform games (actions 5+)
    'ft09-2c8d4e6f': 2,
    'lf52-7b3a9c1d': 2,
    
    # Hybrid games (movement + rotation/transform)
    'cd82-5e1f7a3b': 3,
    'm0r0-8d6c4b2a': 3,
    'cn04-3f9e1d5c': 3,
}

def extract_features_from_game(game_id: str) -> np.ndarray:
    """Load a game, get its initial grid, extract features."""
    try:
        agent = ScientistAgent(game_id, enable_benchmark=False, 
                               enable_skill_tree=False, enable_world_model=False)
        
        if agent.obs.frame and len(agent.obs.frame) > 0:
            grid = agent.obs.frame[0]
            features = extract_features_from_grid(grid)
            return features
    except Exception as e:
        print(f"  ⚠️ {game_id}: {e}")
    return None

def main():
    print("🔬 Extracting real features from ARC-AGI-3 games...")
    print("=" * 55)
    
    X = []
    y = []
    game_names = []
    
    for game_id, domain in GAMES.items():
        short = game_id.split('-')[0].upper()
        print(f"  {short:6s} ({game_id})...", end=' ')
        
        features = extract_features_from_game(game_id)
        if features is not None:
            X.append(features)
            y.append(domain)
            game_names.append(short)
            print(f"✅ {features}")
        else:
            print("❌ failed")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n📊 Extracted {len(X)}/{len(GAMES)} games")
    print(f"   Features shape: {X.shape}")
    print(f"   Domains: {dict(zip(game_names, y))}")
    
    # Also generate synthetic data for augmentation
    from micro_nn.train_domain import generate_synthetic_data
    X_syn, y_syn = generate_synthetic_data(400)
    
    # Combine real + synthetic
    X_train = np.vstack([X, X_syn[:200]])
    y_train = np.concatenate([y, y_syn[:200]])
    
    # Shuffle
    idx = np.random.default_rng(42).permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]
    
    print(f"\n🏗️  Training with {len(X_train)} samples ({len(X)} real + {200} synth)")
    
    # Train
    model = TinyNN([6, 12, 4], ["relu", "softmax"])
    model.train(X_train, y_train, epochs=3000, lr=0.015, verbose=False)
    
    # Test on real games
    print("\n🧪 Testing on real games:")
    labels = ["movement", "rotation", "transform", "hybrid"]
    correct = 0
    for i, (name, expected) in enumerate(zip(game_names, y)):
        pred = model.predict(X[i])
        pred_class = np.argmax(pred)
        ok = "✅" if pred_class == expected else "❌"
        if pred_class == expected:
            correct += 1
        print(f"   {ok} {name:6s} → {labels[pred_class]:10s} (expected {labels[expected]}) conf={pred[pred_class]:.3f}")
    
    acc = correct / len(X)
    print(f"\n   Accuracy: {correct}/{len(X)} = {acc:.0%}")
    
    # Export for Rust
    data = model.export_json()
    out_path = os.path.join(os.path.dirname(__file__), "domain_classifier.json")
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Exported to {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")
    
    # Save real features for future use
    features_path = os.path.join(os.path.dirname(__file__), "real_features.json")
    with open(features_path, 'w') as f:
        json.dump({
            "features": X.tolist(),
            "labels": y.tolist(),
            "games": game_names,
        }, f, indent=2)
    print(f"💾 Real features saved to {features_path}")
    
    return model, acc

if __name__ == "__main__":
    main()
