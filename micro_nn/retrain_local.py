#!/usr/bin/env python3
"""Quick: extract features from locally available ARC-AGI-3 games."""
import numpy as np
import json, os, sys, glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from micro_nn.train_domain import TinyNN, extract_features_from_grid, generate_synthetic_data, get_known_game_features

# Find available games locally
env_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'environment_files')
games_available = []
for game_type in os.listdir(env_dir):
    game_dir = os.path.join(env_dir, game_type)
    if os.path.isdir(game_dir):
        for hash_dir in os.listdir(game_dir):
            py_files = glob.glob(os.path.join(game_dir, hash_dir, '*.py'))
            if py_files:
                games_available.append(f'{game_type}-{hash_dir}')

print(f"📂 Found {len(games_available)} local games")
print(f"   {games_available[:5]}...")

# Build features from LS20 only (we know it works), plus synthetic
# Use the known features from train_domain.py as ground truth
X_known, y_known = get_known_game_features()

# Generate synthetic
X_syn, y_syn = generate_synthetic_data(600)

# Train with synthetic + known features
X_train = np.vstack([X_syn[:400], X_known])
y_train = np.concatenate([y_syn[:400], y_known])

# Shuffle
idx = np.random.default_rng(42).permutation(len(X_train))
X_train, y_train = X_train[idx], y_train[idx]

X_test = X_syn[400:]
y_test = y_syn[400:]

print(f"\n🏗️  Training: {len(X_train)} samples ({len(X_known)} known + {400} synth)")

model = TinyNN([6, 12, 4], ["relu", "softmax"])
model.train(X_train, y_train, epochs=5000, lr=0.01, verbose=False)

train_acc = model.accuracy(X_train, y_train)
test_acc = model.accuracy(X_test, y_test)
print(f"📈 Accuracy: train={train_acc:.1%}, test={test_acc:.1%}")

# Test on known games
print("\n🧪 Known games:")
labels = ["movement", "rotation", "transform", "hybrid"]
games = ["LS20","TR87","RE86","G50T","VC33","R11L","LP85","FT09","LF52","CD82","M0R0","CN04"]
correct = 0
for i, (name, expected) in enumerate(zip(games, y_known)):
    pred = model.predict(X_known[i])
    pred_class = np.argmax(pred)
    ok = "✅" if pred_class == expected else "❌"
    if pred_class == expected: correct += 1
    print(f"   {ok} {name:6s} → {labels[pred_class]:10s} (exp {labels[expected]})")

print(f"\n   {correct}/{len(games)} correct")

# Export
data = model.export_json()
out = os.path.join(os.path.dirname(__file__), 'domain_classifier.json')
with open(out, 'w') as f:
    json.dump(data, f, indent=2)
print(f"\n💾 Exported: {out} ({os.path.getsize(out)/1024:.1f} KB)")
