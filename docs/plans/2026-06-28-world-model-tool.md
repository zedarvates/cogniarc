# World Model Tool — Plan d'implémentation

> **Pour Hermes:** Implémente task par task en TDD.

**Goal:** Créer un outil world-model minimal qui donne à ScientistAgent la capacité de simuler "que se passe-t-il si je fais l'action X ?" sans exécuter l'action réelle dans l'environnement.

**Architecture:** 
- `WorldModelTool` charge l'encodeur V-JEPA 2.1 ViT-B/16 pré-entraîné
- Interface: `predict(observation, action) → (predicted_latent, confidence)`
- k-NN predictor: cherche dans l'historique des états latents celui qui ressemble le plus à "état courant + action X appliquée"
- Intégration optionnelle dans ScientistAgent

**Stack:** PyTorch (encodeur V-JEPA 2.1), numpy (k-NN), pas de GPU requis (CPU OK pour inférence ViT-B/16)

**Pourquoi k-NN et pas un predictor entraîné:** On n'a pas encore entraîné le predictor V-JEPA. Le k-NN est un "simulateur" zero-shot: il mémorise les transitions observées et les rejoue. C'est exactement ce que font les humains novices — "j'ai déjà vu cette situation, la dernière fois j'ai fait X et il s'est passé Y."

---

## Task 1: Créer `world_model.py` avec l'encodeur V-JEPA 2.1

**Objectif:** Charger le modèle ViT-B/16 pré-entraîné et exposer `encode(observation) → latent`.

**Files:**
- Create: `cogniarc/world_model.py`

**Step 1: Écrire le test d'encodage**
```python
# tests/test_world_model.py
def test_encoder_loads_and_encodes():
    from cogniarc.world_model import WorldModelTool
    wm = WorldModelTool()
    obs = np.random.randint(0, 10, (8, 8))  # ARC grid 8x8
    latent = wm.encode(obs)
    assert latent.shape == (768,)  # ViT-B/16 output
    assert np.isfinite(latent).all()
```

**Step 2: Implémenter l'encodeur**
```python
class WorldModelTool:
    def __init__(self, checkpoint_path=None):
        # Charger ViT-B/16 V-JEPA 2.1
        self.encoder = self._load_encoder(checkpoint_path)
        self.memory = []  # [(latent, action, next_latent), ...]
    
    def encode(self, observation: np.ndarray) -> np.ndarray:
        # Resize 8x8 grid → 224x224 image (ou 384)
        # Forward through ViT
        # Return latent vector
```

**Step 3: Run test → FAIL (classe n'existe pas)**

**Step 4: Implémenter le chargement du modèle**
- Télécharger `vjepa2_1_vitb_dist_vitG_384.pt` si absent (~320MB)
- Utiliser le code d'inférence existant de `vjepa-encoder/scripts/vjepa2_infer.py`
- Exposer via une classe propre

**Step 5: Run test → PASS**

---

## Task 2: Ajouter le k-NN predictor + mémoire

**Objectif:** `predict(observation, action) → (predicted_latent, confidence)` en cherchant dans l'historique.

**Step 1: Écrire le test**
```python
def test_predictor_knn():
    wm = WorldModelTool()
    
    # Simuler des transitions mémorisées
    obs1 = np.zeros((8,8)); obs2 = np.ones((8,8))
    latent1 = wm.encode(obs1)
    latent2 = wm.encode(obs2)
    wm.remember(latent1, action=1, next_latent=latent2)
    
    # Prédire ce qui se passe si on fait action 1 depuis obs1
    pred, conf = wm.predict(obs1, action=1)
    assert conf > 0.5  # Devrait trouver une correspondance
```

**Step 2: Implémenter**
```python
def remember(self, latent, action, next_latent):
    self.memory.append((latent, action, next_latent))

def predict(self, observation, action, k=3):
    latent = self.encode(observation)
    # Trouver les k plus proches voisins dans self.memory
    # qui ont la même action
    candidates = [(l, a, nl) for l, a, nl in self.memory if a == action]
    if not candidates:
        return latent, 0.0  # Aucune mémoire → pas de prédiction
    
    # Distance cosinus
    distances = [1 - cosine_similarity(latent, l) for l, _, _ in candidates]
    top_k = np.argsort(distances)[:k]
    
    # Moyenne pondérée des next_latents
    weights = 1 / (np.array(distances)[top_k] + 1e-8)
    weights /= weights.sum()
    predicted = sum(w * candidates[i][2] for i, w in zip(top_k, weights))
    
    confidence = 1 / (1 + distances[top_k[0]])
    return predicted, confidence
```

**Step 3: Test → FAIL**

**Step 4: Implémenter**

**Step 5: Test → PASS**

---

## Task 3: Intégrer dans ScientistAgent

**Objectif:** Ajouter un flag `enable_world_model=True` et intégrer `world_model.predict()` dans le cycle de décision.

**Files:**
- Modify: `cogniarc/scientist_agent.py`

**Step 1: Ajouter l'intégration**
```python
class ScientistAgent:
    def __init__(self, ..., enable_world_model=False):
        if enable_world_model:
            from cogniarc.world_model import WorldModelTool
            self.world_model = WorldModelTool()
        else:
            self.world_model = None
    
    def _world_model_simulate(self, action):
        """Simule l'effet d'une action sans l'exécuter."""
        if not self.world_model:
            return None, 0.0
        obs = self.obs.frame[0] if self.obs.frame else np.zeros((8,8))
        return self.world_model.predict(obs, action)
    
    def _record_transition(self, obs_before, action, obs_after):
        """Enregistre une transition réelle dans la mémoire du world model."""
        if not self.world_model:
            return
        latent_before = self.world_model.encode(obs_before)
        latent_after = self.world_model.encode(obs_after)
        self.world_model.remember(latent_before, action, latent_after)
```

**Step 2: Tester sur LS20**
```python
def test_world_model_integration():
    agent = ScientistAgent('ls20', enable_world_model=True)
    agent.step(1)  # Move right
    # Après quelques steps, le world model devrait pouvoir prédire
    pred, conf = agent._world_model_simulate(1)
    assert pred is not None
```

**Step 3: Test → ok**
