# Plan Studio 3D Hnoss — Bridge Hermes → Blender MCP + ComfyUI

## Objectif
Permettre à Hermes de piloter Blender via MCP et d'améliorer le rendu via ComfyUI sur EUREKAI, en s'inspirant des techniques de Stefan 3D AI et PixelArtistry.

## Architecture cible

```
Hermes (WSL/GLYPH)
    │
    ├── MCP Bridge ──► Blender MCP Server ──► Blender (ODIN-PC Windows)
    │                    (uvx blender-mcp)       │
    │                                             ├── Geometry Nodes
    │                                             ├── Animation
    │                                             ├── Shaders/Textures
    │                                             └── Imprinting vêtements
    │
    └── MCP Bridge ──► ComfyUI (EUREKAI :8188)
                        ├── Workflows Hnoss (lip sync, expression)
                        ├── Texturing IA
                        └── Rendu final
```

## Phase 1 — Blender MCP Bridge

### 1.1 Installer Blender MCP sur ODIN-PC (Windows)
```powershell
# Dans PowerShell (ODIN-PC)
irm https://astral.sh/uv/install.ps1 | iex
$env:Path += ";$env:USERPROFILE\.local\bin"
uvx blender-mcp
```

### 1.2 Installer l'addon Blender
- Télécharger depuis https://www.blender.org/lab/mcp-server/
- Installer dans Blender (Edit → Preferences → Add-ons → Install)
- Activer l'addon

### 1.3 Créer le bridge MCP Hermes
Fichier: `~/.hermes/skills/blender-mcp/scripts/blender_bridge.py`

```python
# Bridge MCP Hermes → Blender
# Utilise le standard MCP pour communiquer avec blender-mcp
# Outils exposés à Hermes :
# - create_object(name, type, location, rotation, scale)
# - modify_mesh(object, operation, params)
# - apply_material(object, material_name, properties)
# - run_animation(sequence)
# - render_scene(output_path, engine, samples)
# - imprint_clothing(target_model, clothing_source)
# - geometry_nodes_apply(modifier_name, params)
```

### 1.4 Exposer les outils Hermes
Via `mcp_servers` dans config.yaml ou via un script Hermes qui appelle le serveur Blender MCP directement.

## Phase 2 — ComfyUI Amélioration

### 2.1 Audit ComfyUI actuel
```bash
# Check workflows existants sur EUREKAI
ssh sylvain@192.168.1.47 "ls ~/ComfyUI/user/default/workflows/"
```

### 2.2 Workflows à créer
- `hnoss_lipsync.json` — pipeline lip sync ComfyUI
- `hnoss_texturing.json` — texturing IA pour vêtements Hnoss
- `hnoss_rendering.json` — rendu final avec contrôle d'expression

### 2.3 Noeuds customs à ajouter
- ControlNet pour pose guidance
- IP-Adapter pour consistency du personnage
- AnimateDiff pour animation

## Phase 3 — Techniques Stefan 3D AI à implémenter

| Vidéo | Technique | Implémentation |
|-------|-----------|----------------|
| Imprinting vêtements | Texture projection → geometry | Blender Geometry Nodes + ComfyUI inpainting |
| Studio 3D | Éclairage + environnement | Three.js scène + HDRI |
| Claude+Blender+Comfy | Pipeline IA→3D | Bridge Hermes MCP |
| Animation | Lip sync + expressions | Rhubarb + blendshapes |

## Phase 4 — Pipeline Hnoss Vtuber Final

```
Texte (Hermes) → TTS (LocalAI) → Rhubarb (visèmes) → Blender (animation) → ComfyUI (rendu) → OBS (stream)
                                                                                    ↓
                                                                            Hnoss avatar 3D
```

## Dépendances

- **Blender MCP**: github.com/ahujasid/blender-mcp
- **ComfyUI**: EUREKAI :8188 (déjà opérationnel)
- **UV**: astral.sh/uv (pour lancer blender-mcp)
- **Rhubarb**: déjà installé (~/.local/bin/rhubarb)
- **LocalAI TTS**: EUREKAI :8080 (tts-1)
- **Problème F:\\**: Les projets Blender sont sur F:\_Serv ULtimate Od\ — inaccessible depuis WSL. Solution : lancer Blender + MCP sur ODIN-PC (Windows natif), Hermes communique via réseau.
