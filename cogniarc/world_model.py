"""
World Model Tool — V-JEPA based simulator for ARC-AGI-3.

Answers "what happens if I take action X in state S?"
Uses pretrained V-JEPA 2.1 ViT-B/16 encoder + k-NN predictor.

Inspired by "Einstein World Models" concept:
The world model is a TOOL that simulates, not the entire architecture.
The agent queries it: "simulate action X, what state do you predict?"

Storage/perf note: transitions live in preallocated contiguous numpy ring
buffers (not a Python list of per-transition arrays), and predict()'s cosine
distance uses einsum instead of np.linalg.norm(axis=1) (which takes a slow
generic path on rectangular arrays). Net effect at 10k transitions / 768-dim:
predict() ~13.4ms -> ~9.7ms (measured; see tests/test_world_model.py for the
numerical-equivalence proof against the original loop implementation). The
remaining cost is the boolean-mask copy selecting same-action transitions
(~30MB at this scale) — further gains would need per-action ring buffers to
avoid that copy, not attempted here to keep the change minimal and reviewable.
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
    
    def __init__(self, config: Optional[WorldModelConfig] = None, game_id: Optional[str] = None):
        self.config = config or WorldModelConfig()
        self.encoder = None
        self.transform = None
        # Transitions stored as preallocated contiguous arrays (ring buffer) so
        # predict() is pure-numpy with no per-call list building. See `memory`
        # property for the (latent, action, next_latent) tuple view.
        self._lat = None   # (cap, D) state latents
        self._nxt = None   # (cap, D) next-state latents
        self._act = None   # (cap,)   actions
        self._n = 0        # number of valid entries
        self._head = 0     # next write index (ring)
        self._loaded = False
        self.game_id = game_id
        
        if HAS_TORCH and os.path.exists(self.config.checkpoint_path):
            self._load_encoder()
        
        # Auto-load memory for this game if game_id provided
        if game_id:
            self.load(game_id)
    
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
    
    def _ensure_arrays(self, dim: int):
        """Lazily allocate the ring buffers once the latent dimension is known."""
        if self._lat is None:
            cap = self.config.max_memory
            self._lat = np.empty((cap, dim), dtype=np.float64)
            self._nxt = np.empty((cap, dim), dtype=np.float64)
            self._act = np.empty(cap, dtype=np.int64)

    @property
    def memory(self):
        """Tuple view (latent, action, next_latent) for save/inspection/back-compat."""
        return [
            (self._lat[i].copy(), int(self._act[i]), self._nxt[i].copy())
            for i in range(self._n)
        ]

    def remember(self, latent: np.ndarray, action: int, next_latent: np.ndarray):
        """Store a transition: (state_latent, action, next_state_latent).

        O(1) append into a preallocated ring buffer; once at capacity the oldest
        entry is overwritten. The world model learns from real experience — each
        transition enriches the k-NN database.
        """
        latent = np.asarray(latent, dtype=np.float64)
        next_latent = np.asarray(next_latent, dtype=np.float64)
        self._ensure_arrays(latent.shape[0])

        i = self._head
        self._lat[i] = latent
        self._nxt[i] = next_latent
        self._act[i] = action
        self._head = (i + 1) % self.config.max_memory
        self._n = min(self._n + 1, self.config.max_memory)
    
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

        if self._n == 0:
            return current_latent.copy(), 0.0

        # Select stored transitions with the same action via a boolean mask
        # (single C-level fancy index — no Python loop, no per-call list build).
        mask = self._act[:self._n] == action
        if not mask.any():
            return current_latent.copy(), 0.0

        latents = self._lat[:self._n][mask]   # (M, D)
        nexts = self._nxt[:self._n][mask]     # (M, D)

        # Cosine distance = 1 - cosine_similarity, computed for all M at once.
        # np.linalg.norm(axis=1) takes a slow generic path on rectangular arrays;
        # einsum's diagonal dot-product is the same math, ~10x faster here.
        cur_norm = np.sqrt(np.dot(current_latent, current_latent))
        denom = np.sqrt(np.einsum('ij,ij->i', latents, latents)) * cur_norm + 1e-8
        cos_sim = (latents @ current_latent) / denom
        distances = 1.0 - cos_sim           # (M,)

        # Top-k nearest (argpartition is O(M); sort only the k for nearest-first).
        k = min(self.config.knn_k, distances.shape[0])
        part = np.argpartition(distances, k - 1)[:k]
        top = part[np.argsort(distances[part])]

        # Inverse-distance-weighted average of the k next-latents.
        weights = 1.0 / (distances[top] + 1e-8)
        weights = weights / weights.sum()
        predicted = (weights[:, None] * nexts[top]).sum(axis=0)

        # Confidence: how close is the nearest neighbor?
        confidence = 1.0 / (1.0 + distances[top[0]])

        return predicted, float(confidence)
    
    def memory_size(self) -> int:
        """Number of transitions stored."""
        return self._n
    
    def save(self, game_id: Optional[str] = None):
        """Persist world model memory to disk for a specific game.
        
        Format: compressed numpy arrays (.npz)
        Path: ~/.cache/cogniarc/world_model/<game_id>.npz
        
        Saves: latents_before [N, D], actions [N], latents_after [N, D]
        """
        gid = game_id or self.game_id
        if not gid:
            return

        if self._n == 0:
            return

        cache_dir = os.path.expanduser("~/.cache/cogniarc/world_model")
        os.makedirs(cache_dir, exist_ok=True)

        latents_before = self._lat[:self._n]
        actions = self._act[:self._n].astype(np.int8)
        latents_after = self._nxt[:self._n]

        path = os.path.join(cache_dir, f"{gid}.npz")
        np.savez_compressed(path,
            latents_before=latents_before,
            actions=actions,
            latents_after=latents_after,
            game_id=gid,
            timestamp=np.datetime64('now')
        )
    
    def load(self, game_id: str):
        """Load world model memory from disk for a specific game.
        
        If the file exists, populates self.memory from saved transitions.
        If not, starts fresh (empty memory) — discovery mode.
        """
        cache_dir = os.path.expanduser("~/.cache/cogniarc/world_model")
        path = os.path.join(cache_dir, f"{game_id}.npz")
        
        if not os.path.exists(path):
            return  # Fresh start — no prior knowledge of this game
        
        try:
            data = np.load(path, allow_pickle=True)
            latents_before = np.asarray(data['latents_before'], dtype=np.float64)
            actions = np.asarray(data['actions'])
            latents_after = np.asarray(data['latents_after'], dtype=np.float64)

            n = len(actions)
            cap = self.config.max_memory
            if n > cap:  # keep the most recent `cap` transitions
                latents_before, latents_after = latents_before[-cap:], latents_after[-cap:]
                actions = actions[-cap:]
                n = cap

            self._ensure_arrays(latents_before.shape[1])
            self._lat[:n] = latents_before
            self._nxt[:n] = latents_after
            self._act[:n] = actions
            self._n = n
            self._head = n % cap

        except Exception as e:
            print(f"[WorldModel] Failed to load memory for {game_id}: {e}")
    
    def forget(self):
        """Clear all memory (fresh start)."""
        self._n = 0
        self._head = 0
