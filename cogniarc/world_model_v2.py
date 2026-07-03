"""
WorldModelTool v2 — Multi-Modal JEPA World Model.

Extensions over v1 (cogniarc/world_model.py):
  1. Multi-modal encoder: image, video, audio, sensor → unified latent
  2. Noise injection: gaussian noise on latents → robust generalization
  3. Latent bottleneck: random projection 768→128 → forces abstraction
  4. Inverse Dynamics Model: discover latent actions without labels
  5. Rollout prediction: N-step forward simulation with stability score

Inspired by:
  - JEPA Overview (I-JEPA → V-JEPA 2 → VL-JEPA)
  - Latent Action World Models "In The Wild" (arXiv:2601.05230)
  - Einstein World Models concept

Key insight from the videos:
  "All the heavy lifting is in perception, not prediction."
  "Noise injection forces the model to shout the big important words."
  "You need the model to be a little bit dumb to generalize."
"""

import os
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

# Reuse base encoder from v1
from cogniarc.world_model import WorldModelTool, WorldModelConfig


class Modality(Enum):
    """Supported input modalities."""
    IMAGE = "image"       # 2D grid or RGB image
    VIDEO = "video"       # Sequence of frames
    AUDIO = "audio"       # Audio waveform or mel-spectrogram
    SENSOR = "sensor"     # Scalar/vector sensor readings (air quality, temp, etc.)


@dataclass
class WorldModelConfigV2(WorldModelConfig):
    """Extended configuration for multi-modal world model v2."""
    # Noise injection
    noise_sigma: float = 0.05          # Gaussian noise std on latents (0=disabled)
    noise_decay: float = 0.995         # Decay noise per transition (more data → less noise)
    
    # Latent bottleneck
    bottleneck_dim: int = 128          # Compressed latent dim (0=no bottleneck)
    bottleneck_projection: Optional[np.ndarray] = None  # Random projection matrix
    
    # Rollout
    rollout_steps: int = 5             # Max steps to simulate forward
    rollout_divergence_threshold: float = 0.5  # Cosine distance > this → diverged
    
    # IDM (Inverse Dynamics Model)
    idm_num_actions: int = 8           # Max latent actions to discover
    idm_cluster_threshold: float = 0.3 # Cosine distance to merge actions
    
    # Multi-modal encoders
    audio_sample_rate: int = 16000
    audio_n_mels: int = 64
    sensor_dim: int = 8                # Expected sensor vector dimension


class MultiModalWorldModel(WorldModelTool):
    """World model v2: multi-modal inputs, noise injection, bottleneck, IDM, rollout.
    
    Extends the base V-JEPA ViT-B/16 encoder with:
      - Audio: mel-spectrogram → ViT (treated as image)
      - Sensor: MLP → concatenated to latent
      - Noise: gaussian injection for robust generalization
      - Bottleneck: random projection for forced abstraction
      - IDM: discovers latent actions from state transitions
      - Rollout: N-step prediction with stability scoring
    
    Usage:
        wm = MultiModalWorldModel()
        
        # Multi-modal encoding
        img_latent = wm.encode({"type": "image", "data": grid_array})
        aud_latent = wm.encode({"type": "audio", "data": audio_waveform})
        sen_latent = wm.encode({"type": "sensor", "data": sensor_vector})
        
        # Combined encoding
        combined = wm.encode_multimodal(
            image=grid_array,
            audio=audio_clip,
            sensors={"co2": 450, "temp": 22.5, "humidity": 60}
        )
        
        # Latent action discovery
        lat_actions, clusters = wm.discover_actions(num_actions=4)
        
        # Rollout
        future_latents, stability = wm.rollout(state, action_sequence, steps=5)
    """
    
    def __init__(self, config: Optional[WorldModelConfigV2] = None, game_id: Optional[str] = None):
        if config is None:
            config = WorldModelConfigV2()
        super().__init__(config, game_id)
        self.cfg: WorldModelConfigV2 = config
        
        # Noise tracking
        self._noise_level = config.noise_sigma
        
        # Bottleneck projection matrix (lazy-init)
        self._bottleneck_matrix: Optional[np.ndarray] = None
        
        # IDM state
        self._latent_actions: Optional[np.ndarray] = None  # (K, bottleneck_dim)
        self._latent_action_counts: Optional[np.ndarray] = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # Multi-Modal Encoding
    # ═══════════════════════════════════════════════════════════════════════
    
    def encode(self, observation: Union[np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """Encode an observation into a latent vector.
        
        Accepts:
          - np.ndarray: legacy mode (2D ARC grid) — delegates to v1 encode()
          - dict: multi-modal input with 'type' and 'data' keys
        
        Multi-modal dict format:
          {"type": "image", "data": ndarray(H,W) or ndarray(H,W,3)}
          {"type": "video", "data": ndarray(T,H,W)}
          {"type": "audio", "data": ndarray(samples,) or ndarray(n_mels, T)}
          {"type": "sensor", "data": ndarray(D,) or list}
        """
        if isinstance(observation, np.ndarray):
            # Legacy mode: standard ARC grid
            return super().encode(observation)
        
        if not isinstance(observation, dict):
            raise ValueError(f"Expected ndarray or dict, got {type(observation)}")
        
        modality = observation.get("type", "image")
        data = observation["data"]
        
        if modality == "image":
            return self._encode_image(data)
        elif modality == "video":
            return self._encode_video(data)
        elif modality == "audio":
            return self._encode_audio(data)
        elif modality == "sensor":
            return self._encode_sensor(data)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def encode_multimodal(self, image: Optional[np.ndarray] = None,
                          video: Optional[np.ndarray] = None,
                          audio: Optional[np.ndarray] = None,
                          sensors: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Encode multiple modalities into a single fused latent vector.
        
        Returns a weighted average of available modalities' latent vectors.
        """
        latents = []
        weights = []
        
        if image is not None:
            latents.append(self._encode_image(image))
            weights.append(1.0)
        if video is not None:
            latents.append(self._encode_video(video))
            weights.append(2.0)  # Video carries more information
        if audio is not None:
            latents.append(self._encode_audio(audio))
            weights.append(1.5)
        if sensors is not None:
            latents.append(self._encode_sensor(sensors))
            weights.append(0.5)  # Sensors are low-dimensional
        
        if not latents:
            raise ValueError("At least one modality must be provided")
        
        weights = np.array(weights) / sum(weights)
        fused = sum(w * l for w, l in zip(weights, latents))
        return fused
    
    def _encode_image(self, data: np.ndarray) -> np.ndarray:
        """Encode image: 2D grid or RGB image → V-JEPA latent."""
        if data.ndim == 2:
            return super().encode(data)  # Legacy encode for 2D grids
        elif data.ndim == 3 and data.shape[2] == 3:
            # RGB image: normalize to [0,9] grayscale then encode
            gray = (data.mean(axis=2) * 9).astype(np.float32)
            return super().encode(gray)
        else:
            raise ValueError(f"Unexpected image shape: {data.shape}")
    
    def _encode_video(self, data: np.ndarray) -> np.ndarray:
        """Encode video: sequence of frames → V-JEPA latent.
        
        Takes the mean latent over frames for efficiency.
        """
        if data.ndim != 3:
            raise ValueError(f"Expected video shape (T, H, W), got {data.shape}")
        
        n_frames = min(data.shape[0], self.cfg.num_frames)
        latents = []
        for i in range(n_frames):
            latents.append(super().encode(data[i]))
        
        result = np.mean(latents, axis=0)
        return result
    
    def _encode_audio(self, data: np.ndarray) -> np.ndarray:
        """Encode audio: waveform or mel-spectrogram → V-JEPA latent.
        
        If raw waveform is provided, converts to mel-spectrogram first,
        then encodes as an image.
        """
        if data.ndim == 1:
            # Raw waveform → mel-spectrogram → image-like encoding
            try:
                import librosa
                mel = librosa.feature.melspectrogram(
                    y=data.astype(np.float32),
                    sr=self.cfg.audio_sample_rate,
                    n_mels=self.cfg.audio_n_mels,
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                # Normalize to [0, 9] for grid encoding
                mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8) * 9
                return super().encode(mel_norm.astype(np.float32))
            except ImportError:
                # librosa not available — use simpler fallback
                return self._fallback_encode_audio(data)
        elif data.ndim == 2:
            # Already mel-spectrogram
            mel_norm = (data - data.min()) / (data.max() - data.min() + 1e-8) * 9
            return super().encode(mel_norm.astype(np.float32))
        else:
            raise ValueError(f"Unexpected audio shape: {data.shape}")
    
    def _fallback_encode_audio(self, data: np.ndarray) -> np.ndarray:
        """Simple audio encoding when librosa is unavailable.
        
        Uses statistical features: RMS energy, zero-crossing rate,
        spectral centroid proxy, and time-domain statistics.
        """
        # Downsample to manageable size
        if len(data) > 44100:
            data = data[::len(data) // 44100]
        
        features = []
        # RMS energy in chunks
        chunk_size = len(data) // 16
        for i in range(16):
            chunk = data[i * chunk_size:(i + 1) * chunk_size]
            features.append(np.sqrt(np.mean(chunk ** 2)))
        
        # Zero-crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(data)))) / 2
        features.append(zcr * 10)
        
        # Pad to 768 dims (same as ViT latent)
        result = np.array(features, dtype=np.float64)
        result = result / (np.linalg.norm(result) + 1e-8)
        
        # Expand via repetition + noise to match 768
        repeats = 768 // len(result) + 1
        padded = np.tile(result, repeats)[:768]
        return padded.astype(np.float64)
    
    def _encode_sensor(self, data: Union[np.ndarray, Dict[str, float], list]) -> np.ndarray:
        """Encode sensor readings → latent vector.
        
        Accepts: ndarray, dict of name→value, or list.
        Uses a simple MLP-like expansion to 768 dims.
        
        Sensor examples: CO2 (ppm), temperature (°C), humidity (%),
        particulate matter (PM2.5), air pressure (hPa), VOC (ppb).
        """
        # Convert to array
        if isinstance(data, dict):
            values = list(data.values())
        elif isinstance(data, (list, tuple)):
            values = list(data)
        elif isinstance(data, np.ndarray):
            values = data.flatten().tolist()
        else:
            raise ValueError(f"Unexpected sensor type: {type(data)}")
        
        sensor_vec = np.array(values, dtype=np.float64)
        
        # Normalize: clip to reasonable ranges, then standardize
        if len(sensor_vec) > 0:
            # Simple z-score normalization on the vector itself
            mean = sensor_vec.mean()
            std = sensor_vec.std()
            if std > 1e-8:
                sensor_vec = (sensor_vec - mean) / std
            sensor_vec = np.tanh(sensor_vec)  # Bound to [-1, 1]
        
        # Create a pseudo-learned expansion
        n = len(sensor_vec)
        if n == 0:
            return np.zeros(768, dtype=np.float64)
        
        # Repeat to fill 768 dims
        result = np.zeros(768, dtype=np.float64)
        for i in range(768):
            result[i] = sensor_vec[i % n]
        
        # Add sinusoidal encoding to make each dimension unique
        pos = np.arange(768, dtype=np.float64)
        for i in range(min(4, len(sensor_vec))):
            freq = 2.0 ** i
            result += 0.1 * sensor_vec[i % n] * np.sin(pos * freq / 768 * np.pi)
        
        result = result / (np.linalg.norm(result) + 1e-8)
        return result
    
    # ═══════════════════════════════════════════════════════════════════════
    # Noise Injection
    # ═══════════════════════════════════════════════════════════════════════
    
    def _apply_noise(self, latent: np.ndarray, force: bool = False) -> np.ndarray:
        """Apply gaussian noise to latent for robust generalization.
        
        Inspired by JEPA 'In The Wild': noise injection forces the model
        to ignore fine-grained pixel details — only large structural
        movements survive the noise.
        
        The noise level decays over time as more transitions are collected,
        simulating the model gaining confidence in its world representation.
        """
        if self.cfg.noise_sigma <= 0 and not force:
            return latent
        
        sigma = self._noise_level
        noise = np.random.default_rng().normal(0, sigma, latent.shape)
        noisy = latent + noise
        
        # Re-normalize
        noisy = noisy / (np.linalg.norm(noisy) + 1e-8)
        
        # Decay noise for next time
        self._noise_level *= self.cfg.noise_decay
        self._noise_level = max(self._noise_level, 0.001)  # Floor
        
        return noisy
    
    # ═══════════════════════════════════════════════════════════════════════
    # Latent Bottleneck
    # ═══════════════════════════════════════════════════════════════════════
    
    def _init_bottleneck(self):
        """Initialize random projection matrix for bottleneck.
        
        Uses a fixed random projection (Gaussian) to compress 768-dim
        latent to bottleneck_dim. This is deterministic after first init.
        
        Inspired by JEPA: "starve the model" — restrict information capacity
        to force learning of abstract, transferable features.
        """
        if self.cfg.bottleneck_dim <= 0:
            return
        
        if self._bottleneck_matrix is None:
            rng = np.random.default_rng(42)  # Fixed seed for determinism
            # Gaussian random projection
            mat = rng.normal(0, 1 / np.sqrt(self.cfg.bottleneck_dim),
                            (self.cfg.latent_dim, self.cfg.bottleneck_dim))
            self._bottleneck_matrix = mat
    
    def _apply_bottleneck(self, latent: np.ndarray) -> np.ndarray:
        """Compress latent through bottleneck.
        
        768-dim → bottleneck_dim (e.g., 128) via random projection.
        If latent is already at bottleneck_dim, return as-is.
        """
        if self.cfg.bottleneck_dim <= 0:
            return latent
        
        if len(latent) == self.cfg.bottleneck_dim:
            return latent  # Already compressed
        
        self._init_bottleneck()
        compressed = latent @ self._bottleneck_matrix  # (768,) @ (768, B) → (B,)
        compressed = compressed / (np.linalg.norm(compressed) + 1e-8)
        return compressed
    
    # ═══════════════════════════════════════════════════════════════════════
    # Override remember() and predict() with noise + bottleneck
    # ═══════════════════════════════════════════════════════════════════════
    
    def remember(self, latent: np.ndarray, action: int, next_latent: np.ndarray):
        """Store a transition with noise injection + bottleneck compression."""
        # Apply bottleneck
        latent_c = self._apply_bottleneck(np.asarray(latent, dtype=np.float64))
        next_latent_c = self._apply_bottleneck(np.asarray(next_latent, dtype=np.float64))
        
        # Apply noise to stored latents (makes retrieval more robust)
        latent_n = self._apply_noise(latent_c)
        next_latent_n = self._apply_noise(next_latent_c)
        
        # Store compressed + noisy versions
        super().remember(latent_n, action, next_latent_n)
    
    def predict(self, observation: Union[np.ndarray, Dict[str, Any]],
                action: int) -> Tuple[np.ndarray, float]:
        """Predict next state with bottleneck + noise.
        
        Override: encodes with bottleneck, then uses base k-NN for prediction.
        """
        current_latent = self.encode(observation)
        current_c = self._apply_bottleneck(np.asarray(current_latent, dtype=np.float64))
        
        if self._n == 0:
            return current_c.copy(), 0.0
        
        mask = self._act[:self._n] == action
        if not mask.any():
            return current_c.copy(), 0.0
        
        latents = self._lat[:self._n][mask]
        nexts = self._nxt[:self._n][mask]
        
        # Cosine distance (from base impl)
        cur_norm = np.sqrt(np.dot(current_c, current_c))
        denom = np.sqrt(np.einsum('ij,ij->i', latents, latents)) * cur_norm + 1e-8
        cos_sim = (latents @ current_c) / denom
        distances = 1.0 - cos_sim
        
        k = min(self.config.knn_k, distances.shape[0])
        part = np.argpartition(distances, k - 1)[:k]
        top = part[np.argsort(distances[part])]
        
        weights = 1.0 / (distances[top] + 1e-8)
        weights = weights / weights.sum()
        predicted = (weights[:, None] * nexts[top]).sum(axis=0)
        
        confidence = 1.0 / (1.0 + distances[top[0]])
        
        return predicted, float(confidence)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Inverse Dynamics Model (IDM) — Latent Action Discovery
    # ═══════════════════════════════════════════════════════════════════════
    
    def discover_actions(self, num_actions: int = 4,
                         min_transitions: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Discover latent actions from stored transitions without labels.
        
        Inspired by JEPA 'In The Wild': the Inverse Dynamics Model looks at
        state(t) and state(t+1) and infers "what must have happened in between?"
        
        Algorithm:
          1. Compute delta vectors: D_i = latent(t+1) - latent(t)
          2. Cluster deltas into K groups via k-means-like assignment
          3. Each cluster center = a "latent action"
        
        Args:
            num_actions: Number of latent actions to discover
            min_transitions: Minimum transitions needed
        
        Returns:
            (latent_actions, cluster_assignments)
            latent_actions: (K, D) cluster centers
            cluster_assignments: (N,) action index per transition
        """
        if self._n < min_transitions:
            return np.array([]), np.array([])
        
        D = self._lat.shape[1]
        N = self._n
        
        # Compute deltas
        deltas = self._nxt[:N] - self._lat[:N]  # (N, D)
        
        # Simple k-means-like clustering
        # Initialize centroids: pick random deltas
        rng = np.random.default_rng(42)
        indices = rng.choice(N, min(num_actions, N), replace=False)
        centroids = deltas[indices].copy()  # (K, D)
        
        # Assign + update for a few iterations
        for _ in range(10):
            # Compute cosine similarity to each centroid
            # Using einsum for efficiency: (N, K)
            d_norm = np.sqrt(np.einsum('ij,ij->i', deltas, deltas)) + 1e-8
            c_norm = np.sqrt(np.einsum('ij,ij->i', centroids, centroids)) + 1e-8
            sim = (deltas @ centroids.T) / (d_norm[:, None] * c_norm[None, :] + 1e-8)
            
            # Assign each delta to nearest centroid
            assignments = np.argmax(sim, axis=1)
            
            # Update centroids
            for k in range(num_actions):
                mask = assignments == k
                if mask.sum() > 0:
                    centroids[k] = deltas[mask].mean(axis=0)
        
        # Merge similar centroids
        merged_centroids, merged_labels = self._merge_similar_actions(
            centroids, assignments, deltas
        )
        
        self._latent_actions = merged_centroids
        self._latent_action_counts = np.bincount(merged_labels, minlength=len(merged_centroids))
        
        return merged_centroids, merged_labels
    
    def _merge_similar_actions(self, centroids: np.ndarray,
                               assignments: np.ndarray,
                               deltas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Merge centroids that are too similar (cosine distance < threshold)."""
        K = len(centroids)
        if K <= 1:
            return centroids, assignments
        
        # Compute pairwise cosine similarity
        norms = np.sqrt(np.einsum('ij,ij->i', centroids, centroids)) + 1e-8
        sim = (centroids @ centroids.T) / (norms[:, None] * norms[None, :] + 1e-8)
        
        merged = list(range(K))
        for i in range(K):
            for j in range(i + 1, K):
                if sim[i, j] > (1.0 - self.cfg.idm_cluster_threshold):
                    merged[j] = merged[i]
        
        # Build mapping
        unique = sorted(set(merged))
        new_labels = np.array([unique.index(merged[a]) for a in assignments])
        
        # Recompute centroids from original deltas using new labels
        new_centroids = np.array([
            deltas[new_labels == l].mean(axis=0)
            for l in range(len(unique))
        ])
        
        return new_centroids, new_labels
    
    def predict_action(self, state: np.ndarray, next_state: np.ndarray) -> int:
        """Predict which latent action connects state → next_state.
        
        Returns the index of the closest latent action centroid,
        or -1 if no actions discovered yet.
        """
        if self._latent_actions is None or len(self._latent_actions) == 0:
            return -1
        
        delta = (np.asarray(next_state, dtype=np.float64) -
                 np.asarray(state, dtype=np.float64))
        
        # Find closest centroid
        d_norm = np.linalg.norm(delta) + 1e-8
        c_norms = np.sqrt(np.einsum('ij,ij->i', self._latent_actions,
                                    self._latent_actions)) + 1e-8
        sim = (delta @ self._latent_actions.T) / (d_norm * c_norms + 1e-8)
        
        return int(np.argmax(sim))
    
    # ═══════════════════════════════════════════════════════════════════════
    # Rollout Prediction
    # ═══════════════════════════════════════════════════════════════════════
    
    def rollout(self, observation: Union[np.ndarray, Dict[str, Any]],
                action_sequence: List[int],
                steps: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
        """Simulate N steps forward using the world model.
        
        At each step, feeds its OWN prediction back as input for the next step.
        This is the key difference from teacher forcing — the model must
        practice dealing with its own imperfections.
        
        Inspired by JEPA robotics: rollout loss trains the model to be
        robust to its own prediction errors over long sequences.
        
        Args:
            observation: Starting state (grid, dict, or already-encoded latent)
            action_sequence: List of actions to simulate
            steps: Max steps (defaults to len(action_sequence))
        
        Returns:
            (predicted_latents, stability_score)
            predicted_latents: List of latent vectors (one per step)
            stability_score: 0-1 where 1 = perfectly stable, 0 = diverged
        """
        if steps is None:
            steps = min(len(action_sequence), self.cfg.rollout_steps)
        else:
            steps = min(steps, len(action_sequence), self.cfg.rollout_steps)
        
        if steps == 0 or self._n == 0:
            return [], 1.0
        
        # Encode if observation is not already a latent vector
        if isinstance(observation, np.ndarray) and len(observation.shape) == 1:
            current = observation  # Already a latent vector
        else:
            current = self.encode(observation)
        
        predicted_latents = []
        divergence_count = 0
        
        for i in range(steps):
            pred, confidence = self._predict_from_latent(current, action_sequence[i])
            predicted_latents.append(pred)
            
            if confidence < 0.3:
                divergence_count += 1
            
            current = pred
        
        stability = 1.0 - (divergence_count / steps) if steps > 0 else 1.0
        return predicted_latents, stability
    
    def _predict_from_latent(self, latent: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """Predict next state from an already-encoded latent vector.
        
        Bypasses the encode() step — used internally by rollout().
        """
        current_c = self._apply_bottleneck(np.asarray(latent, dtype=np.float64))
        
        if self._n == 0:
            return current_c.copy(), 0.0
        
        mask = self._act[:self._n] == action
        if not mask.any():
            return current_c.copy(), 0.0
        
        latents = self._lat[:self._n][mask]
        nexts = self._nxt[:self._n][mask]
        
        cur_norm = np.sqrt(np.dot(current_c, current_c))
        denom = np.sqrt(np.einsum('ij,ij->i', latents, latents)) * cur_norm + 1e-8
        cos_sim = (latents @ current_c) / denom
        distances = 1.0 - cos_sim
        
        k = min(self.config.knn_k, distances.shape[0])
        part = np.argpartition(distances, k - 1)[:k]
        top = part[np.argsort(distances[part])]
        
        weights = 1.0 / (distances[top] + 1e-8)
        weights = weights / weights.sum()
        predicted = (weights[:, None] * nexts[top]).sum(axis=0)
        confidence = 1.0 / (1.0 + distances[top[0]])
        
        return predicted, float(confidence)
    
    def rollout_to_divergence(self, observation: Union[np.ndarray, Dict[str, Any]],
                              action: int, max_steps: int = 20) -> int:
        """Count how many steps before the model diverges.
        
        Useful for measuring world model quality: a good model stays stable
        for many steps; a poor model diverges quickly.
        """
        current = self.encode(observation)
        
        for step in range(max_steps):
            pred, confidence = self.predict(current, action)
            if confidence < self.cfg.rollout_divergence_threshold:
                return step
            current = pred
        
        return max_steps
    
    # ═══════════════════════════════════════════════════════════════════════
    # Statistics & Reporting
    # ═══════════════════════════════════════════════════════════════════════
    
    def report(self) -> str:
        """Human-readable status report."""
        lines = [
            f"🌍 WorldModel v2 [{self.game_id or 'no game'}]",
            f"   Transitions: {self._n}/{self.config.max_memory}",
            f"   Noise sigma: {self._noise_level:.4f}",
            f"   Bottleneck: {self.cfg.bottleneck_dim}d "
            f"({'initialized' if self._bottleneck_matrix is not None else 'lazy'})",
        ]
        
        if self._latent_actions is not None and len(self._latent_actions) > 0:
            lines.append(f"   Latent actions: {len(self._latent_actions)}")
            if self._latent_action_counts is not None:
                counts = self._latent_action_counts
                for i in range(len(self._latent_actions)):
                    cnt = counts[i] if i < len(counts) else 0
                    lines.append(f"     Action {i}: {cnt} transitions")
        
        if self._n >= 4:
            # Estimate prediction quality: self-predict on last 4 transitions
            correct = 0
            for i in range(max(0, self._n - 4), self._n):
                latent = self._lat[i]
                action = int(self._act[i])
                expected = self._nxt[i]
                pred, conf = self._predict_from_latent(latent, action)
                cos_sim = np.dot(pred, expected) / (
                    np.linalg.norm(pred) * np.linalg.norm(expected) + 1e-8
                )
                if cos_sim > 0.95:
                    correct += 1
            lines.append(f"   Self-predict accuracy: {correct}/4")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Quick diagnostic: test all modalities
# ═══════════════════════════════════════════════════════════════════════

def demo_multimodal():
    """Demonstrate multi-modal world model capabilities."""
    print("=== MultiModalWorldModel Demo ===\n")
    
    wm = MultiModalWorldModel(game_id="demo")
    cfg = wm.cfg
    
    # 1. Image encoding (ARC grid)
    print("1. Image encoding (ARC grid 40×40):")
    grid = np.random.randint(0, 10, (40, 40))
    img_latent = wm.encode({"type": "image", "data": grid})
    print(f"   Shape: {img_latent.shape}, norm: {np.linalg.norm(img_latent):.4f}")
    
    # 2. Audio encoding
    print("\n2. Audio encoding (sine wave 1s @ 440Hz):")
    sr = cfg.audio_sample_rate
    t = np.linspace(0, 1, sr, endpoint=False)
    sine = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    aud_latent = wm.encode({"type": "audio", "data": sine})
    print(f"   Shape: {aud_latent.shape}, norm: {np.linalg.norm(aud_latent):.4f}")
    
    # 3. Sensor encoding
    print("\n3. Sensor encoding (air quality):")
    sensors = {"co2": 450, "temperature": 22.5, "humidity": 60,
               "pm25": 12, "voc": 150, "pressure": 1013}
    sen_latent = wm.encode({"type": "sensor", "data": sensors})
    print(f"   Shape: {sen_latent.shape}, norm: {np.linalg.norm(sen_latent):.4f}")
    
    # 4. Multi-modal fusion
    print("\n4. Multi-modal fusion (image + audio + sensors):")
    fused = wm.encode_multimodal(image=grid, audio=sine, sensors=sensors)
    print(f"   Shape: {fused.shape}, norm: {np.linalg.norm(fused):.4f}")
    
    # 5. Noise + bottleneck demo
    print(f"\n5. Noise injection (sigma={cfg.noise_sigma}):")
    noisy = wm._apply_noise(fused, force=True)
    cos_sim = np.dot(fused, noisy) / (np.linalg.norm(fused) * np.linalg.norm(noisy) + 1e-8)
    print(f"   Cosine similarity original→noisy: {cos_sim:.4f}")
    
    print(f"\n6. Bottleneck: {cfg.latent_dim}→{cfg.bottleneck_dim}")
    compressed = wm._apply_bottleneck(fused)
    print(f"   Shape: {compressed.shape}")
    
    # 7. Remember + Predict
    print("\n7. Remember transitions:")
    for action in range(1, 5):
        latent_before = wm._apply_bottleneck(img_latent.copy())
        latent_after = wm._apply_bottleneck(
            wm.encode({"type": "image", "data": np.random.randint(0, 10, (40, 40))})
        )
        wm.remember(latent_before, action, latent_after)
    print(f"   Stored: {wm.memory_size()} transitions")
    
    # 8. Predict
    print("\n8. Predict action=1 from image:")
    pred, conf = wm.predict({"type": "image", "data": grid}, 1)
    print(f"   Confidence: {conf:.4f}, pred norm: {np.linalg.norm(pred):.4f}")
    
    # 9. Latent action discovery
    print("\n9. Latent action discovery:")
    actions, clusters = wm.discover_actions(num_actions=4)
    print(f"   Discovered {len(actions)} latent actions")
    for i, count in enumerate(np.bincount(clusters, minlength=len(actions))):
        print(f"     Action {i}: {count} transitions")
    
    # 10. Rollout
    print("\n10. Rollout (5 steps, action cycle 1,2,3,4,1):")
    latents, stability = wm.rollout(
        {"type": "image", "data": grid},
        [1, 2, 3, 4, 1], steps=5
    )
    print(f"    Stability: {stability:.4f}, predicted {len(latents)} steps")
    
    print("\n" + wm.report())
    print("\n=== Demo complete ===")


if __name__ == "__main__":
    demo_multimodal()
