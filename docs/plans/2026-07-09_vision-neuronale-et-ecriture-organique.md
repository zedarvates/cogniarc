# Plan : Mode « Vision Neuronale » + Écriture Organique + Dessin Vectoriel

> **Pour Hermes/Claude :** Implémenter phase par phase, tester chaque étape,
> valider sur holdout AVANT de laisser piloter (discipline docs/EVALUATION.md).

**Objectif global :** Donner à l'agent une perception apprise (pas seulement
symbolique), puis appliquer le même principe d'apprentissage moteur à
l'écriture manuscrite et au dessin vectoriel — apprendre à écrire/dessiner
comme un enfant humain : par essai, évaluation, correction. Livrable final :
dessiner réellement « notre maison cluster ensoleillée » en SVG organique.

---

## Phase 1 — Mode « Vision Neuronale » (ReasoningMode.VISION, mode 11)

### État des lieux (ce qui existe déjà, à réutiliser — ne rien réinventer)

| Brique | Fichier | État |
|--------|---------|------|
| Segmentation symbolique (couleur/connexité) | `object_perception.py` (ObjectTracker) | ✅ câblée, live-vérifiée |
| Encodeur latent V-JEPA 2.1 + k-NN | `world_model.py` (WorldModelTool) | ✅ vectorisé, **sous-utilisé** (~60% acc, opt-in) |
| Fallback statistique 768-dim | `world_model.py` (_fallback_encode) | ✅ mais « 0% real grid understanding » (README) |
| Filtres visuels | `vision_filters.py` | ⚠️ orphelin (jamais importé) |
| Attention (focus follows changes) | `attention.py` | ⚠️ orphelin |
| Micro-NN infra (numpy→JSON→inférence) | `micro_predictors.py`, `micro_nn/` | ✅ pipeline éprouvé |
| Mode 10 SIMULATION_PHYSIQUE | `scientist_agent.py` (ReasonModeManager) | ✅ précédent : comment ajouter un mode |

### Principe directeur (leçon logic-vs-NN, README §Logic vs Micro-NN)

Le neuronal doit être réservé aux mappings **sans règle fermée** :
- ✅ « ces deux régions sont-elles le même objet après téléportation/changement de couleur ? » (l'ObjectTracker échoue : il matche par couleur identique uniquement)
- ✅ « cette configuration de grille ressemble-t-elle à une déjà vue ? » (similarité perceptuelle)
- ❌ PAS pour re-détecter le joueur/murs quand la corrélation symbolique suffit (elle marche, elle est exacte, elle est gratuite)

**La vision neuronale complète l'ObjectTracker là où il échoue, elle ne le remplace pas.**

### Tâche 1.1 — Cartographier les échecs de l'ObjectTracker (mesure d'abord)

Instrumenter `observe()` : compter les régions « disparues » (aucun match
même-couleur — aujourd'hui ignorées ligne `continue  # region vanished`).
Logger (game_id, step, couleur, aire) dans un JSONL. Lancer sur 3 jeux
holdout bornés. **Si <5% des régions posent problème, la Phase 1 s'arrête
là** — pas de neuronal sans besoin mesuré.

- Fichiers : `object_perception.py` (+compteur), `scripts/measure_perception_gaps.py` (nouveau)
- Test : compteur exact sur grilles synthétiques avec disparition/changement de couleur

### Tâche 1.2 — Matching perceptuel de régions (micro-NN siamois)

Si 1.1 montre un vrai besoin : micro-NN siamois `(patch_A 8×8, patch_B 8×8) →
même_objet ∈ [0,1]`, entraîné sur paires positives/négatives **générées par
l'ObjectTracker lui-même** sur les jeux dev (self-supervision : les matchs
même-couleur-proche sont des positifs sûrs ; les paires aléatoires distantes
des négatifs). Pipeline existant : train numpy → JSON → inférence
`micro_predictors.py`. Baseline à battre (test obligatoire) : matching par
histogramme de couleur + distance. Si le NN ne bat pas la baseline sur des
transitions holdout tenues à l'écart → on garde la baseline (règle du repo).

- Fichiers : `micro_nn/train_region_matcher.py`, `cogniarc/micro_predictors.py` (+RegionMatcher), `tests/test_region_matcher.py`

### Tâche 1.3 — Mémoire de configurations via V-JEPA (réutiliser l'existant)

Le WorldModelTool encode déjà des grilles entières en 768-dim. L'utiliser
pour un `have_i_seen_this_before(grid) → (similar_game_state, distance)` :
détection de boucles/retours à un état connu, complémentaire au hash exact
de `cognitive_player.py` (qui rate les états *presque* identiques).

- Fichiers : `world_model.py` (+méthode `nearest_state`), test avec fallback encoder

### Tâche 1.4 — Mode 11 VISION dans ReasonModeManager

Même pattern que le mode 10 : `ReasoningMode.VISION = "vision"`, trigger =
`perception_gap_detected` (régions disparues ce niveau > seuil) OU
`object_tracker.player_color is None après N steps`. Priorité entre
GOAL_INFERENCE et EXPLORATION. Le mode oriente vers les outils 1.2/1.3.
**Advisory d'abord** (log la recommandation), pilote seulement après
validation holdout — comme tout le reste.

- Fichiers : `scientist_agent.py` (enum + stratégie), tests pattern `test_mode_driven_decisions.py`

### Critère de succès Phase 1
`player_color` identifié sur ≥8/10 jeux holdout en ≤20 steps (aujourd'hui :
à mesurer via 1.1 — probablement ~5-6/10), sans régression LS20.

---

## Phase 2 — Écriture organique : apprendre à écrire comme un enfant

### Concept (spécification utilisateur)

Chaque lettre/chiffre = **squelette de points de contrôle** + **variance
positionnelle par point** → rendu organique. Le système *apprend* : on ne
code pas la belle lettre, on code le squelette idéal + un σ (tremblement)
par point, et une boucle de pratique réduit σ là où l'évaluateur détecte des
erreurs — exactement l'apprentissage moteur d'un enfant : gribouillage →
lettres tremblées → écriture assurée.

### Architecture (nouveau package `human_skills/`, ce repo)

```
human_skills/
├── __init__.py
├── glyphs.py          # Squelettes : {char: [Stroke]} ; Stroke = [(x, y) normalisés 0-1]
│                      #   v1 : chiffres 0-9 + A-Z majuscules bâton (traits droits + arcs)
├── organic.py         # Jitter organique :
│                      #   - bruit gaussien corrélé le long du trait (pas i.i.d. — un
│                      #     tremblement de main est lisse, pas du bruit blanc)
│                      #   - variance de pente (slant), d'échelle, de ligne de base
│                      #   - paramètre global "âge moteur" : σ_global de 0.15 (enfant) → 0.01 (adulte)
├── render_svg.py      # Points jitterés → chemins SVG (Catmull-Rom → Bézier cubiques)
│                      #   épaisseur variable (pression simulée), pas de dépendance GUI
├── evaluate.py        # Évaluateur géométrique PUR (pas de vision nécessaire) :
│                      #   - distance de Fréchet discrète trait tracé vs squelette idéal
│                      #   - fermeture des boucles (<tolérance), proportions, ordre des traits
│                      #   - score 0-100 par glyphe
└── practice.py        # Boucle d'apprentissage moteur :
│                      #   1. écrire le glyphe avec σ courant
│                      #   2. évaluer → erreurs localisées PAR POINT
│                      #   3. réduire σ des points fautifs (consolidation), garder un
│                      #      σ_min résiduel = la "main" reste organique à jamais
│                      #   4. maîtrise = 5 essais consécutifs ≥ 80 (règle SkillDAG existante)
└── tests/             # Tout est pur numpy/math → testable sans écran
```

### Points de conception clés

1. **La variance ne descend jamais à 0** : σ_min ≈ 0.008 — une écriture
   parfaitement géométrique est un échec du projet, pas une réussite.
2. **Corrélation du bruit** : offset(t) = somme de 2-3 harmoniques sinus à
   phase aléatoire (pseudo-Perlin 1D simple, zéro dépendance) — c'est ce qui
   fait « main humaine » vs « imprimante qui tremble ».
3. **Courbe d'apprentissage mesurable** : `practice.py` exporte le score par
   essai en JSONL → on peut TRACER la courbe enfant→adulte, c'est le
   livrable scientifique de la phase.
4. **Mots ensuite** : espacement inter-lettres avec sa propre variance,
   ligature simple optionnelle.

### Critère de succès Phase 2
Les 36 glyphes atteignent la maîtrise (5×≥80) en <200 essais chacun, ET un
humain reconnaît chaque lettre rendue à σ_final (test : export d'une planche
SVG « alphabet appris » lisible).

---

## Phase 3 — Dessin vectoriel : la maison cluster ensoleillée 🏠☀️

Le morceau de graduation. Réutilise `organic.py`/`render_svg.py` — **le
dessin est de l'écriture avec d'autres squelettes**.

1. **Primitives** (`human_skills/shapes.py`) : ligne, rectangle, triangle,
   cercle/ellipse, arc — chacune = squelette de points + jitter organique,
   comme les glyphes. Niveaux 0-1 du curriculum arc-human-skills du README.
2. **Composition** (`human_skills/scenes.py`) : une scène = liste de formes
   posées avec positions/tailles relatives + ordre de dessin (fond → détails,
   comme un enfant : le corps de la maison d'abord, la fumée en dernier).
3. **La maison** (`scripts/draw_cluster_house.py`) :
   - corps principal + 2-3 modules accolés (le « cluster »)
   - toits triangulaires, porte, fenêtres à croisillons
   - ☀️ soleil : cercle + rayons (traits organiques rayonnants)
   - rayons de lumière vers la maison, un nuage, ligne de sol herbeuse
   - le tout tracé avec le σ **appris en Phase 2** — la maison a la même
     « main » que l'écriture, signée en lettres apprises : « notre maison »
4. **Sortie** : `outputs/maison_cluster_ensoleillee.svg` (+ PNG via pillow
   si simple). Bonus : version σ_enfant vs σ_adulte côte à côte.

### Critère de succès Phase 3
SVG s'ouvre dans un navigateur, scène reconnaissable, traits visiblement
organiques (pas de lignes parfaites), signature manuscrite issue de la
boucle d'apprentissage réelle (pas d'une police).

---

## Ordre d'exécution & garde-fous

1. Phase 1 T1.1 (mesure) → décision GO/NO-GO neuronal sur données réelles
2. Phase 2 en parallèle possible (indépendante du runtime arc_agi — pur numpy)
3. Phase 3 après maîtrise Phase 2
4. **Jamais** de câblage pilote en Phase 1 sans validation holdout préalable
5. Chaque module : tests pur-python d'abord, exactement comme les 157 tests actuels

## Hors périmètre (explicitement)

- Pas de gros modèle de vision (CNN profond, ViT entraîné) tant que le
  micro-NN siamois n'a pas prouvé/infirmé le besoin
- Pas d'automation MS Paint (pywinauto) en v1 — le SVG est le médium :
  reproductible, testable, diffable
- Pas de lettres cursives liées en v1 (bâton d'abord, comme à l'école)
