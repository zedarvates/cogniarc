# Prochaine Session — Pipeline Vidéo Hnoss Propre

## Prérequis : Étudier ces ressources AVANT de coder

### LTX 2.3
- [ ] docs.comfy.org/tutorials/video/ltx/ltx-2-3 — workflows officiels
- [ ] NVIDIA Blender→ComfyUI pipeline (nvidia.com guide)
- [ ] First & Last Frame technique (YouTube: LTX 2.3 First Last Frame)
- [ ] LTX 2.3 IC LoRA pour référence sheet

### Blender → IA Video
- [ ] Blender render comme depth map / control
- [ ] Textures, accessoires, vêtements AVANT render
- [ ] Images de référence : bureau/test influenceuse tech/
- [ ] Modèle Hnoss : bureau/hnoss/ (3d/, images/, textures/)

### TTS
- [ ] Français : voix féminine française (pas nova)
- [ ] Anglais : voix féminine US

## Pipeline Corrigé

```
1. Étudier les références (test influenceuse tech/)
2. Blender : nettoyer scène, habiller Hnoss (textures/accessoires)
3. Blender : setup décors + éclairage
4. Render image(s) de référence
5. ComfyUI : workflow LTX 2.3 officiel
6. Technique : first frame + last frame
7. TTS français voix féminine
8. Assemblage final vidéo + audio
```

## Erreurs à ne pas répéter
- ❌ Laisser le cube par défaut dans Blender
- ❌ Ignorer les images de référence
- ❌ Utiliser un prompt hors-sujet (Egyptian royal...)
- ❌ Lancer des workflows sans comprendre les noeuds
- ❌ Bricoler au lieu d'étudier d'abord
