"""Distillation pipeline — use deterministic extractors to label real data, then retrain micro-NN.

Pattern:
  1. Deterministic extractor produces (features, label) pairs from real data
  2. These pairs train a micro-NN
  3. Micro-NN learns to replicate the extractor at <1ms

This replaces hand-crafted synthetic rules with REAL learned patterns.
"""

import numpy as np
import json
import os
from typing import Callable, List, Tuple, Dict, Any


class DistillationPipeline:
    """Generic distillation: teacher → features → student micro-NN.
    
    Teacher = deterministic extractor (heuristic, CSS selector, regex, etc.)
    Student = micro-NN (numpy feedforward)
    """
    
    def __init__(self, name: str, architecture: List[int], activations: List[str]):
        self.name = name
        self.architecture = architecture
        self.activations = activations
        self.X: List[np.ndarray] = []
        self.y: List[np.ndarray] = []
    
    def collect(self, teacher_fn: Callable, data_source: Any,
                extract_features: Callable, n_labels: int) -> int:
        """Collect labeled data from teacher.
        
        Args:
            teacher_fn: (data_item) -> label_index (deterministic, no noise)
            data_source: iterable of real data items
            extract_features: (data_item) -> np.ndarray of features
            n_labels: number of output classes
        
        Returns:
            Number of samples collected
        """
        count = 0
        for item in data_source:
            try:
                label_idx = teacher_fn(item)
                features = extract_features(item)
                
                y_vec = np.zeros(n_labels, dtype=np.float32)
                y_vec[label_idx] = 1.0
                
                self.X.append(features.astype(np.float32))
                self.y.append(y_vec)
                count += 1
            except Exception as e:
                continue
        
        return count
    
    def train(self, epochs=200, lr=0.003, batch_size=32, test_split=0.2):
        """Train student micro-NN on collected data."""
        if len(self.X) == 0:
            raise ValueError("No data collected. Call collect() first.")
        
        from micro_nn.train_pathfinder import TinyNN
        
        X_arr = np.array(self.X)
        y_arr = np.array(self.y)
        
        # Shuffle + split
        idx = np.random.default_rng(42).permutation(len(X_arr))
        X_arr, y_arr = X_arr[idx], y_arr[idx]
        
        n_test = int(len(X_arr) * test_split)
        X_train, y_train = X_arr[n_test:], y_arr[n_test:]
        X_test, y_test = X_arr[:n_test], y_arr[:n_test]
        
        # Normalize
        self.feature_mean = X_train.mean(axis=0)
        self.feature_std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - self.feature_mean) / self.feature_std
        X_test_norm = (X_test - self.feature_mean) / self.feature_std
        
        print(f"  Training: {len(X_train)} samples, testing: {len(X_test)}")
        print(f"  Architecture: {self.architecture}")
        
        self.model = TinyNN(self.architecture, self.activations)
        self.model.train(X_train_norm, y_train, epochs=epochs, lr=lr, 
                        batch_size=batch_size, verbose=(epochs >= 100))
        
        train_acc = self.model.accuracy(X_train_norm, y_train)
        test_acc = self.model.accuracy(X_test_norm, y_test)
        
        print(f"  Accuracy: train={train_acc:.1%}, test={test_acc:.1%}")
        
        return {'train_acc': train_acc, 'test_acc': test_acc}
    
    def export(self, path: str, metadata: Dict = None):
        """Export trained model to JSON (same format as other micro-NNs)."""
        if not hasattr(self, 'model'):
            raise ValueError("No trained model. Call train() first.")
        
        data = self.model.export_json()
        data['feature_mean'] = self.feature_mean.tolist()
        data['feature_std'] = self.feature_std.tolist()
        if metadata:
            data.update(metadata)
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        size_kb = os.path.getsize(path) / 1024
        print(f"  Exported: {path} ({size_kb:.1f} KB)")
        return path
    
    def evaluate(self, teacher_fn: Callable, test_items: List,
                 extract_features: Callable, n_labels: int) -> Dict:
        """Compare student vs teacher on test set."""
        if not hasattr(self, 'model'):
            raise ValueError("No trained model. Call train() first.")
        
        correct = 0
        total = 0
        disagreements = []
        
        for item in test_items:
            try:
                teacher_label = teacher_fn(item)
                features = extract_features(item)
                
                x = (features - self.feature_mean) / self.feature_std
                probs = self.model.predict(x)
                student_label = np.argmax(probs)
                
                total += 1
                if student_label == teacher_label:
                    correct += 1
                else:
                    disagreements.append({
                        'teacher': int(teacher_label),
                        'student': int(student_label),
                        'confidence': float(probs[student_label]),
                    })
            except Exception:
                continue
        
        acc = correct / max(total, 1)
        print(f"  Student vs Teacher: {acc:.1%} ({correct}/{total})")
        if disagreements:
            print(f"  Disagreements: {len(disagreements)}")
            for d in disagreements[:3]:
                print(f"    teacher={d['teacher']} student={d['student']} conf={d['confidence']:.3f}")
        
        return {'accuracy': acc, 'total': total, 'disagreements': disagreements}


# ═══ Pre-built distillation recipes ═══

def distill_pathfinder_from_heuristic(grid, wall_colors, n_samples=1000):
    """Distill pathfinder micro-NN from heuristic wall-circumvention.
    
    Teacher: heuristic_path.heuristic_navigate()
    Student: PathfinderPredictor (105 -> 64 -> 32 -> 4)
    """
    from cogniarc.heuristic_path import heuristic_navigate
    from micro_nn.train_pathfinder import extract_patch_features
    
    h, w = grid.shape
    rng = np.random.default_rng(42)
    
    def generate_sample(_):
        # Pick random start and target positions
        px = rng.integers(1, w-1)
        py = rng.integers(1, h-1)
        tx = rng.integers(1, w-1)
        ty = rng.integers(1, h-1)
        
        # Skip if on wall
        if int(grid[py, px]) in wall_colors or int(grid[ty, tx]) in wall_colors:
            return None
        
        # Get teacher action
        path = heuristic_navigate(grid, px, py, tx, ty, wall_colors, max_steps=1)
        if not path:
            return None
        
        action, _ = path[0]
        features = extract_patch_features(grid, px, py, tx, ty, wall_colors)
        
        return {
            'features': features,
            'label': action - 1,  # Convert 1-4 to 0-3
        }
    
    # Collect
    samples = []
    for _ in range(n_samples * 2):  # Oversample to account for skips
        s = generate_sample(None)
        if s:
            samples.append(s)
        if len(samples) >= n_samples:
            break
    
    pipeline = DistillationPipeline(
        'pathfinder', [105, 64, 32, 4], ['relu', 'relu', 'softmax']
    )
    
    # Convert samples to pipeline format
    pipeline.X = [s['features'] for s in samples]
    pipeline.y = [np.eye(4)[s['label']].astype(np.float32) for s in samples]
    
    return pipeline, samples


def distill_captcha_from_css(screenshots_with_css: List[Tuple[np.ndarray, str]]):
    """Distill CAPTCHA classifier from CSS-detected labels.
    
    Teacher: CSS selector (finds iframe, identifies CAPTCHA type)
    Student: CaptchaPredictor (256 -> 64 -> 32 -> 6)
    
    Args:
        screenshots_with_css: list of (screenshot_array, css_type_string)
            where css_type_string is one of TYPES
    
    Returns:
        DistillationPipeline ready for training
    """
    TYPES = ['recaptcha_v2', 'hcaptcha', 'turnstile', 'text_captcha', 'math_captcha', 'none']
    type_to_idx = {t: i for i, t in enumerate(TYPES)}
    
    pipeline = DistillationPipeline(
        'captcha', [256, 64, 32, 6], ['relu', 'relu', 'softmax']
    )
    
    for screenshot, css_type in screenshots_with_css:
        if css_type not in type_to_idx:
            continue
        
        # Downsample to 16x16
        if len(screenshot.shape) == 3:
            gray = np.mean(screenshot, axis=2)
        else:
            gray = screenshot
        
        h, w = gray.shape
        downsampled = np.zeros((16, 16))
        for i in range(16):
            for j in range(16):
                y0, y1 = i * h // 16, (i+1) * h // 16
                x0, x1 = j * w // 16, (j+1) * w // 16
                downsampled[i, j] = np.mean(gray[y0:max(y0+1, y1), x0:max(x0+1, x1)])
        
        features = downsampled.flatten() / 255.0
        label_idx = type_to_idx[css_type]
        
        pipeline.X.append(features.astype(np.float32))
        pipeline.y.append(np.eye(6, dtype=np.float32)[label_idx])
    
    return pipeline
