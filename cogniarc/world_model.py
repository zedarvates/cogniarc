"""
World Model Tool — V-JEPA based simulator for ARC-AGI-3.

Answers "what happens if I take action X in state S?"
Uses pretrained V-JEPA 2.1 ViT-B/16 encoder + k-NN predictor.

Inspired by "Einstein World Models" concept:
The world model is a TOOL that simulates, not the entire architecture.
The agent queries it: "simulate action X, what state do you predict?"
"""

import os
import sys
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

# Optional torch import — world model is available but not mandatory
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class WorldModelConfig:
    """Configuration for the world model tool."""
    checkpoint_path: str = os.path.expanduser("~/.hermes/models/vjepa2_vitb_384.pt")
    img_size: int = 384
    patch_size: int = 16
    tubelet_size: int = 2
    num_frames: int = 16
    use_rope: bool = True
    uniform_power: bool = True
    checkpoint_key: str = "ema_encoder"
    latent_dim: int = 768  # ViT-B/16 output
    knn_k: int = 3  # Number of neighbors for prediction
    max_memory: int = 10000  # Max transitions to remember
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)


class WorldModelTool:
    """World model as a tool: encode states, predict transitions via k-NN.
    
    Usage:
        wm = WorldModelTool()
        latent = wm.encode(grid_observation)
        wm.remember(latent_before, action=1, next_latent=latent_after)
        predicted, confidence = wm.predict(grid_observation, action=1)
    """
    
    def __init__(self, config: Optional[WorldModelConfig] = None):
        self.config = config or WorldModelConfig()
        self.encoder = None
        self.transform = None
        self.memory: List[Tuple[np.ndarray, int, np.ndarray]] = []  # (latent, action, next_latent)
        self._loaded = False
        
        if HAS_TORCH and os.path.exists(self.config.checkpoint_path):
            self._load_encoder()
    
    def _load_encoder(self):
        """Load the V-JEPA 2.1 ViT-B/16 encoder."""
        try:
            # Import from vjepa2 package (editable install)
            sys.path.insert(0, os.path.expanduser("~/.hermes/src/vjepa2"))
            from src.models.vision_transformer import vit_base
            
            cfg = self.config
            self.encoder = vit_base(
                patch_size=cfg.patch_size,
                tubelet_size=cfg.tubelet_size,
                use_rope=cfg.use_rope,
                uniform_power=cfg.uniform_power,
                num_frames=cfg.num_frames,
                img_size=(cfg.img_size, cfg.img_size),
            )
            
            # Load checkpoint
            state_dict = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
            encoder_dict = state_dict.get(cfg.checkpoint_key, state_dict.get("encoder", state_dict))
            encoder_dict = {
                k.replace("module.", "").replace("backbone.", ""): v
                for k, v in encoder_dict.items()
            }
            self.encoder.load_state_dict(encoder_dict, strict=False)
            self.encoder.eval()
            
            # Build transform
            short_side = int(256.0 / 224 * cfg.img_size)
            self.transform = T.Compose([
                T.Resize(short_side, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop((cfg.img_size, cfg.img_size)),
                T.ToTensor(),
                T.Normalize(mean=cfg.mean, std=cfg.std),
            ])
            
            self._loaded = True
        except Exception as e:
            print(f"[WorldModel] Failed to load encoder: {e}")
            self._loaded = False
    
    @property
    def available(self) -> bool:
        """Check if the world model is loaded and ready."""
        return self._loaded and self.encoder is not None
    
    def grid_to_image(self, grid: np.ndarray) -> Image.Image:
        """Convert an ARC grid (integers) to an RGB PIL Image.
        
        Grid values are mapped to grayscale (0=black, 9=white).
        Returns a 384x384 RGB image via nearest-neighbor upscale.
        """
        h, w = grid.shape
        # Normalize to 0-255 grayscale
        img_array = (grid.astype(np.float32) / max(9, grid.max())) * 255
        img_array = img_array.astype(np.uint8)
        
        # Create PIL image and resize to target size
        img = Image.fromarray(img_array, mode='L').convert('RGB')
        img = img.resize((self.config.img_size, self.config.img_size), Image.NEAREST)
        return img
    
    def encode(self, observation: np.ndarray) -> np.ndarray:
        """Encode an ARC grid observation into a latent vector.
        
        Args:
            observation: 2D numpy array (H, W) with integer values 0-9
        
        Returns:
            latent vector of shape (768,)
        """
        if not self.available:
            # Fallback: simple statistical encoding
            return self._fallback_encode(observation)
        
        # Convert grid to image
        img = self.grid_to_image(observation)
        
        # Create a static "video" of 16 identical frames
        tensor = self.transform(img)  # [3, H, W]
        video_tensor = tensor.unsqueeze(1).repeat(1, self.config.num_frames, 1, 1)  # [3, T, H, W]
        video_tensor = video_tensor.unsqueeze(0)  # [1, 3, T, H, W]
        
        # Forward through encoder
        with torch.inference_mode():
            features = self.encoder(video_tensor)  # [1, num_patches, 768]
        
        # Mean pool over patches → single latent vector
        latent = features.mean(dim=1).squeeze(0).numpy()  # [768]
        return latent
    
    def _fallback_encode(self, observation: np.ndarray) -> np.ndarray:
        """Fallback encoding when torch/V-JEPA is unavailable.
        
        Uses simple statistical features: histogram, spatial moments, gradients.
        Returns a 768-dim vector (same size as ViT for compatibility).
        """
        h, w = observation.shape
        
        # Color histogram (10 bins for values 0-9) → 10 dims
        hist = np.bincount(observation.flatten().astype(int), minlength=10) / (h * w)
        
        # Spatial moments (center of mass per color) → 10*2 = 20 dims
        moments = []
        for color in range(10):
            mask = (observation == color)
            if mask.sum() > 0:
                ys, xs = np.where(mask)
                moments.extend([xs.mean() / w, ys.mean() / h])
            else:
                moments.extend([0.0, 0.0])
        
        # Simple gradients (edge detection) → 4 dims
        grad_x = np.abs(np.diff(observation.astype(float), axis=1)).mean()
        grad_y = np.abs(np.diff(observation.astype(float), axis=0)).mean()
        grad_mean = (grad_x + grad_y) / 2
        
        # Compress to 768 dims via repetition + noise
        base = np.concatenate([hist, np.array(moments), [grad_x, grad_y, grad_mean, observation.mean() / 9.0]])
        # Repeat and add tiny noise to reach 768 dims
        repeats = 768 // len(base) + 1
        result = np.tile(base, repeats)[:768]
        result += np.random.default_rng(42).normal(0, 0.001, 768)  # Seed fixed for determinism
        result = result / (np.linalg.norm(result) + 1e-8)
        
        return result
    
    def remember(self, latent: np.ndarray, action: int, next_latent: np.ndarray):
        """Store a transition in memory: (state_latent, action, next_state_latent).
        
        The world model learns from real experience. Each transition adds to the
        k-NN database, enabling future predictions.
        """
        self.memory.append((latent.copy(), action, next_latent.copy()))
        
        # Evict oldest if over capacity
        if len(self.memory) > self.config.max_memory:
            self.memory = self.memory[-self.config.max_memory:]
    
    def predict(self, observation: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """Predict the next latent state if we take the given action.
        
        Uses k-NN: finds the k most similar past states with the same action,
        and returns a weighted average of their outcomes.
        
        Args:
            observation: Current ARC grid (H, W)
            action: Action to simulate (1=up, 2=down, 3=left, 4=right, 5=interact, 6=rotate)
        
        Returns:
            (predicted_next_latent, confidence)
            confidence: 0.0 = no memory, 1.0 = exact match
        """
        current_latent = self.encode(observation)
        
        # Find past transitions with the same action
        candidates = [
            (mem_latent, mem_next)
            for mem_latent, mem_action, mem_next in self.memory
            if mem_action == action
        ]
        
        if not candidates:
            return current_latent.copy(), 0.0
        
        # Compute cosine distances
        distances = []
        for mem_latent, _ in candidates:
            # Cosine distance = 1 - cosine_similarity
            dot = np.dot(current_latent, mem_latent)
            norm = np.linalg.norm(current_latent) * np.linalg.norm(mem_latent) + 1e-8
            cos_sim = dot / norm
            distances.append(1.0 - cos_sim)
        
        # Get top-k
        k = min(self.config.knn_k, len(candidates))
        top_indices = np.argsort(distances)[:k]
        
        # Weighted average of next_latents (inverse distance weighting)
        weights = 1.0 / (np.array([distances[i] for i in top_indices]) + 1e-8)
        weights = weights / weights.sum()
        
        predicted = np.zeros_like(current_latent)
        for idx, w in zip(top_indices, weights):
            predicted += w * candidates[idx][1]
        
        # Confidence: how close is the nearest neighbor?
        confidence = 1.0 / (1.0 + distances[top_indices[0]])
        
        return predicted, float(confidence)
    
    def memory_size(self) -> int:
        """Number of transitions stored."""
        return len(self.memory)
    
    def forget(self):
        """Clear all memory (fresh start)."""
        self.memory.clear()
