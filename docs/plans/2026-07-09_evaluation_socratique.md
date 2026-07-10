# 🔍 Évaluation Socratique de CogniARC

> Appliqué au repo lui-même : examiner les hypothèses, les lacunes,
> les faux dilemmes, et proposer un plan qui maximise la valeur réelle.

---

## 1. Hypothèses non examinées

| Hypothèse | Examen |
|-----------|--------|
| *"Plus on ajoute de modules, plus l'agent devient intelligent"* | ❌ Faux. 156k LOC mais l'agent résout toujours 1 niveau sur 1 jeu. |
| *"Le physics engine 20 domaines sert à résoudre ARC-AGI"* | ⚠️ Non démontré. Les jeux ARC-AGI sont des puzzles 2D, pas des simulations physiques. |
| *"human_skills/ est dans CogniARC donc il sert CogniARC"* | ⚠️ 0 imports de human_skills dans cogniarc/. C'est un colocataire, pas un organe. |
| *"LS20 solved" = progression significative* | ⚠️ 1 niveau sur combien ? Les holdout ne sont pas résolus. |

## 2. Lacunes de preuves

- **Où est le benchmark holdout ?** Le plan en parle (docs/EVALUATION.md) mais on n'a jamais vu les chiffres.
- **Où est la preuve que le physics engine sert à quelque chose dans ARC-AGI ?** Les jeux sont des grilles 2D avec des règles logiques, pas des chutes de billes.
- **Où est la preuve que l'abacus aide l'agent ?** C'est un module isolé. Aucun mode de raisonnement ne l'appelle.
- **Où est la preuve que les 9 reasoning modes aident réellement ?** Qui vérifie que le mode sélectionné est le bon ?

## 3. Faux dilemme

> *"Soit on ajoute des fonctionnalités, soit on résout les jeux."*

C'est faux. Le vrai plan doit prioriser CE QUI DÉBLOQUE LES JEUX :
- La perception (trouver le joueur) → débloque la navigation
- La navigation (best_action_toward) → débloque l'exploration
- L'exploration (ObjectTracker + hypothèses) → débloque la résolution

Tout le reste (écriture, boulier, physique, maison, V-JEPA à 60%) est **hors du chemin critique**.

## 4. Sur-généralisation

- *"L'agent est générique"* — Non, il marche sur LS20. Sur les holdout, la chaîne B1+B2 était cassée. Maintenant elle est débloquée mais aucun niveau n'est résolu.
- *"Le world model est un outil"* — Oui, mais il est utilisé à 60% d'acc et en opt-in. C'est comme avoir une scie circularie qu'on utilise que pour couper du beurre.

---

# 🎯 Plan d'action

## Phase urgente — Débloquer les holdout (maintenant)

```
NIVEAU DE PRIORITÉ : CRITIQUE
```

### 1. Mesurer la perception gap sur les VRAIS jeux holdout
Le T1.1 existe mais sur données synthétiques. On doit lancer l'ObjectTracker sur 3 vrais jeux holdout et mesurer le vrai taux de régions disparues.

→ Sortie : vrai taux de gap. Si <5% : la perception n'est pas le problème.

### 2. Si perception OK mais navigation KO :
Le vrai problème post-B2 c'est que `best_action_toward()` existe mais la navigation multi-step n'est pas fiable. Un planificateur de chemin générique (A* sur la grille perçue) manque encore — le Pathfinder existe dans world_model_physics mais n'est pas utilisé par l'agent.

→ Action : brancher le pathfinding existant dans la boucle de décision.

### 3. Si navigation OK mais résolution KO :
Le solver sait bouger mais ne sait pas QUOI faire. C'est le problème le plus dur. Les 9 reasoning modes doivent être testés sur holdout pour voir lequel aide.

→ Action : benchmark des modes sur holdout.

---

## Phase valeur — Connecter ce qui existe (cette semaine)

```
NIVEAU DE PRIORITÉ : ÉLEVÉ
```

### 4. WIRER Mode 10 (SIMULATION_PHYSIQUE)
Le physics engine est prêt, le mode existe, mais il n'est JAMAIS appelé. La mailbox/trigger du ReasonModeManager doit être câblée pour que `causal_ambiguity > threshold` active réellement la simulation.

### 5. WIRER l'abacus comme Mode 12 (COMPTAGE_VISUEL)
L'abacus est un mode de raisonnement parfait pour les jeux qui nécessitent de compter des pas, des distances, des cycles. Mode 12 = l'agent pose les nombres sur le boulier visuel et « voit » le résultat au lieu de calculer abstraitement.

### 6. WIRER l'écriture dans le diagnostic
Quand l'agent explore une grille, il pourrait *dessiner* son hypothèse (le chemin prévu, la zone explorée) plutôt que de la décrire en texte. La `SceneGraph` du physics engine + `strokes_to_svg` = diagnostic visuel des hypothèses.

---

## Phase consolidation — Mesurer, documenter, fermer

```
NIVEAU DE PRIORITÉ : MOYEN
```

### 7. Benchmark holdout formel
- 5 jeux holdout
- 10 runs chacun
- Métriques : levels_completed, steps, perception_gap, player_color_found, stagnation_count
- Résultat : tableau clair de ce qui marche et ce qui ne marche pas

### 8. Nettoyage de la dette
- `attention.py` et `vision_filters.py` sont orphelins (jamais importés)
- Le `world_model.py` (V-JEPA) vs `world_model_physics/` — deux visions du monde qui coexistent sans communication
- Les scripts `hailo_vision.py` dépendent du matériel Hailo-8 — documenter ou retirer

---

## Résumé exécutif

| Priorité | Action | Impact attendu | Effort |
|----------|--------|----------------|--------|
| 🔴 | Mesurer perception gap sur vrais holdout | Savoir si la vision neuronale est justifiée | 1h |
| 🔴 | Brancher pathfinding dans boucle agent | Débloquer la navigation multi-step | 2h |
| 🟠 | WIRER Mode 10 physique | Donner un usage réel au physics engine | 2h |
| 🟠 | WIRER abacus Mode 12 | Donner un usage réel au boulier | 2h |
| 🟡 | Benchmark holdout formel | Savoir où on en est VRAIMENT | 3h |
| 🟡 | Nettoyer orphelins | Réduire la dette cognitive | 1h |
