#!/usr/bin/env python3
"""
Cartographie Sonore — Carte des paramètres audio, leurs effets perceptifs 
et leurs significations symboliques.

Comme pour les jeux ARC-AGI-3 et les CAPTCHAs, chaque paramètre sonore
est catalogué avec :
  - Son effet sur la perception humaine
  - Sa signification symbolique (qu'est-ce que ça VEUT dire)
  - Son usage possible en synthèse / analyse
  - Sa connexion aux émotions / états

Structure: PARAMÈTRE → EFFET PERÇU → SYMBOLE → APPLICATION
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ════════════════════════════════════════════════
#  1. PARAMÈTRES FONDAMENTAUX
# ════════════════════════════════════════════════

class AudioParameter(Enum):
    """Les paramètres fondamentaux du son (la palette du peintre sonore)."""
    # ── Dynamique ──
    GAIN = "gain"                     # Volume / intensité
    ENVELOPE_ATTACK = "attack"        # Temps d'attaque
    ENVELOPE_DECAY = "decay"          # Temps de déclin
    ENVELOPE_SUSTAIN = "sustain"      # Niveau de maintien
    ENVELOPE_RELEASE = "release"      # Temps de relâchement
    
    # ── Spectral ──
    FREQUENCY = "frequency"           # Hauteur (Hz)
    TIMBRE = "timbre"                 # Contenu harmonique
    BRIGHTNESS = "brightness"         # Ratio aigu/grave (spectral centroid)
    WARMTH = "warmth"                 # Richesse en graves
    AIR = "air"                       # Présence d'hautes fréquences >8kHz
    
    # ── Spatial ──
    PAN = "pan"                       # Position stéréo (L/R)
    DEPTH = "depth"                   # Profondeur (avant/arrière via reverb + volume)
    WIDTH = "width"                   # Largeur stéréo
    REVERB_TIME = "reverb_time"       # Temps de réverbération (RT60)
    REVERB_PREDELAY = "pre_delay"     # Délai avant réverb
    REVERB_DAMPING = "damping"        # Amortissement hautes fréquences de la reverb
    ECHO_DELAY = "echo_delay"         # Temps d'écho
    ECHO_FEEDBACK = "echo_feedback"   # Feedback d'écho (répétitions)
    
    # ── Modulation ──
    VIBRATO_RATE = "vibrato_rate"     # Taux de vibrato (Hz)
    VIBRATO_DEPTH = "vibrato_depth"   # Profondeur de vibrato
    TREMOLO_RATE = "tremolo_rate"     # Taux de trémolo (variation de volume)
    TREMOLO_DEPTH = "tremolo_depth"   # Profondeur de trémolo
    PHASER_RATE = "phaser_rate"       # Taux de phaser
    FLANGER_RATE = "flanger_rate"     # Taux de flanger
    CHORUS_DEPTH = "chorus_depth"     # Profondeur de chorus
    WAH_FREQ = "wah_freq"             # Fréquence centrale du filtre wah
    WAH_RESONANCE = "wah_resonance"   # Résonance du filtre wah
    
    # ── Distorsion / Texture ──
    DISTORTION_AMOUNT = "distortion"  # Taux de distorsion
    SATURATION = "saturation"         # Saturation douce (tape, tube)
    BITCRUSH = "bitcrush"             # Réduction de résolution (lo-fi)
    NOISE_FLOOR = "noise_floor"       # Bruit de fond
    COMPRESSION_RATIO = "comp_ratio"  # Ratio de compression
    COMPRESSION_THRESHOLD = "comp_threshold"  # Seuil de compression
    
    # ── Psychoacoustique ──
    DOPPLER_SHIFT = "doppler"         # Effet Doppler (variation de hauteur)
    HAAS_EFFECT = "haas"              # Effet Haas (délai interaural)
    SHEPARD_TONE = "shepard"          # Tonalité de Shepard (illusion infinie)
    MISSING_FUNDAMENTAL = "missing_fundamental"  # Fondamental manquant
    BINAURAL_BEAT = "binaural"        # Battements binauraux
    AUDITORY_STREAMING = "streaming"  # Streaming auditif (ségrégation de flux)
    PRECEDENCE_EFFECT = "precedence"  # Effet de précédence (Haas localisation)
    FRISBIE = "frisbie"               # Effet Frisbie (poursuite auditive)


@dataclass
class ParameterMap:
    """Carte complète d'un paramètre audio → effet → symbole."""
    parameter: AudioParameter
    display_name: str
    unit: str
    range_human: str                    # Plage typique en valeur humaine
    
    # ── Effet perceptif ──
    what_it_does: str                   # Description concise de l'effet
    emotional_effect: str               # Impact émotionnel
    physical_sensation: str             # Sensation corporelle associée
    
    # ── Signification symbolique ──
    symbolic_meaning: str               # Qu'est-ce que ça représente
    narrative_use: str                  # Usage narratif (cinéma, jeux)
    archetype: str                      # Archétype sonore associé
    
    # ── Application technique ──
    detection_method: str               # Comment le détecter/analyser
    synthesis_method: str               # Comment le générer
    cogniarc_skill: str                 # Skill CogniArc associée


# ════════════════════════════════════════════════
#  2. CATALOGUE COMPLET
# ════════════════════════════════════════════════

SOUND_CARTOGRAPHY: dict[AudioParameter, ParameterMap] = {}

def _register(pm: ParameterMap):
    SOUND_CARTOGRAPHY[pm.parameter] = pm

# ─── 2.1 DYNAMIQUE ───

_register(ParameterMap(
    parameter=AudioParameter.GAIN,
    display_name="Gain / Volume",
    unit="dB",
    range_human="-60 dB (silence) à +12 dB (saturation)",

    what_it_does="Intensité perçue. Un son plus fort = plus proche, plus urgent, plus présent. "
                 "La perception du volume est logarithmique (loi de Weber-Fechner).",
    emotional_effect="Fort → alerte, puissance, danger, urgence. "
                     "Faible → calme, distance, secret, humilité.",
    physical_sensation="Le son fort fait vibrer le corps (poitrine, tympans). "
                       "Le son faible force l'attention, le rapprochement.",

    symbolic_meaning="Proximité vs distance. Puissance vs discrétion. "
                     "Un gain qui augmente = quelque chose qui APPROCHE (danger, révélation). "
                     "Un gain qui diminue = quelque chose qui S'ÉLOIGNE (espoir, souvenir).",
    narrative_use="Crescendo = montée de tension. Diminuendo = apaisement. "
                  "Silence soudain = choc, révélation, mort.",
    archetype="Le souffle divin (fort = présence, faible = absence).",

    detection_method="RMS + Peak meters. Crête-à-crête sur la waveform.",
    synthesis_method="Multiplicateur linéaire (amplitude). Taper graduel.",
    cogniarc_skill="attention — le volume dirige l'attention auditive",
))

_register(ParameterMap(
    parameter=AudioParameter.ENVELOPE_ATTACK,
    display_name="Attaque",
    unit="ms",
    range_human="0.1 ms (percussif) à 500 ms (doux)",

    what_it_does="Temps que met un son à atteindre son amplitude maximale. "
                 "Attaque rapide = percussif, frappant. Attaque lente = doux, enveloppant.",
    emotional_effect="Attaque rapide → surprise, choc, impact, agression. "
                     "Attaque lente → douceur, nostalgie, transition progressive.",
    physical_sensation="Attaque rapide = impulsion physique (coup de pied). "
                       "Attaque lente = caresse, vague qui monte.",

    symbolic_meaning="L'immédiateté de l'événement. "
                     "Attaque rapide = événement SOUDAIN (naissance, explosion, apparition). "
                     "Attaque lente = événement PROGRESSIF (aube, sentiment qui grandit, mort lente).",
    narrative_use="Attaque rapide = coup de feu, impact. Attaque lente = lever de soleil, crescendo orchestral.",
    archetype="La naissance (attaque rapide) vs l'émergence (attaque lente).",

    detection_method="Détection de transitoire. Dérivée de l'enveloppe.",
    synthesis_method="ADSR envelope. Paramètre 'attack' du synthétiseur.",
    cogniarc_skill="temporal_inference — l'attaque = l'instant T0 d'un événement temporel",
))

_register(ParameterMap(
    parameter=AudioParameter.ENVELOPE_RELEASE,
    display_name="Release / Chute",
    unit="ms",
    range_human="10 ms (sec) à 5 s (trainant)",

    what_it_does="Temps que met le son à s'éteindre après la fin de la note. "
                 "Release court = sec, précis. Release long = trainant, réverbérant.",
    emotional_effect="Release court → netteté, discipline, finitude. "
                     "Release long → nostalgie, infini, tristesse, espace.",
    physical_sensation="Release long = vibration qui persiste dans le corps. "
                       "Release court = coup sec, point final.",

    symbolic_meaning="La MORT du son, sa disparition. "
                     "Release court = mort nette (coupé, terminé). "
                     "Release long = agonie, évanescence, souvenir qui s'estompe. "
                     "Release infini (reverb) = éternité, transcendance.",
    narrative_use="Release long sur la dernière note = fin ouverte. Release court = conclusion définitive.",
    archetype="L'écho de l'âme (ce qui persiste après le départ).",

    detection_method="Décroissance de l'enveloppe après le relâchement.",
    synthesis_method="Paramètre 'release' de l'ADSR. Aussi temps de reverb.",
    cogniarc_skill="temporal_inference — le release = le DECAY temporel d'un événement",
))

_register(ParameterMap(
    parameter=AudioParameter.ENVELOPE_DECAY,
    display_name="Decay / Déclin",
    unit="ms",
    range_human="10 ms à 2 s",

    what_it_does="Temps que met le son à passer de l'attaque maximale au niveau de maintien (sustain). "
                 "Définit la 'queue' de l'impact initial.",
    emotional_effect="Decay court → percussif, net, tranchant. "
                     "Decay long → lourd, trainant, massif.",
    physical_sensation="Decay court = pincement, impact sec. Decay long = poids qui s'installe.",

    symbolic_meaning="L'INSTALLATION d'une présence. Decay court = événement soudain qui disparaît vite. "
                     "Decay long = événement qui s'installe, qui prend de la place. "
                     "Le decay d'une cloche = solennité qui persiste.",
    narrative_use="Cloche = decay long. Coup de feu = decay court. "
                  "Decay qui s'allonge = événement qui prend plus d'importance.",
    archetype="L'impact (court) vs la cloche (long).",

    detection_method="Pente de l'enveloppe entre le pic et le sustain.",
    synthesis_method="Paramètre 'decay' de l'ADSR.",
    cogniarc_skill="temporal_inference — le DECELERATING pattern (décélération après l'impact)",
))

_register(ParameterMap(
    parameter=AudioParameter.ENVELOPE_SUSTAIN,
    display_name="Sustain / Maintien",
    unit="Niveau (0-1)",
    range_human="0 (aucun) à 1 (maintien total)",

    what_it_does="Niveau du son PENDANT la phase de maintien (après l'attaque, avant le release). "
                 "Définit si le son 'reste présent' ou s'il s'éteint progressivement.",
    emotional_effect="Sustain haut → persistance, obstination, présence continue. "
                     "Sustain bas → évanescence, légèreté, fragilité.",
    physical_sensation="Sustain haut = vibration continue dans le corps. "
                       "Sustain bas = caresse qui passe.",

    symbolic_meaning="LA PERSISTANCE. Sustain = capacité à DURER. "
                     "Haut = endurance, vie éternelle, présence inébranlable. "
                     "Bas = éphémère, instant fragile, papillon. "
                     "Zéro = percussif pur (l'instant qui ne dure pas).",
    narrative_use="Note d'orgue = sustain infini. Piano = sustain qui décroît. "
                  "Sustain qui monte = un personnage qui gagne en endurance.",
    archetype="L'éternité (sustain infini) vs l'instant (sustain zéro).",

    detection_method="Niveau moyen de l'enveloppe pendant la phase de maintien.",
    synthesis_method="Paramètre 'sustain' de l'ADSR.",
    cogniarc_skill="temporal_inference — le CONSTANT pattern (maintien stable)",
))

# ─── 2.2 SPECTRAL ───

_register(ParameterMap(
    parameter=AudioParameter.BRIGHTNESS,
    display_name="Brightness / Aiguë",
    unit="Hz (spectral centroid)",
    range_human="200 Hz (sombre) à 4000 Hz (brillant)",

    what_it_does="Ratio d'énergie dans les hautes fréquences. "
                 "Bright = présence, clarté, définition. Dark = sourd, voilé, chaud.",
    emotional_effect="Bright → joie, alerte, clarté, vérité. "
                     "Dark → mystère, tristesse, danger, introspection.",
    physical_sensation="Bright = piqûre, picotement, présence dans la tête (tympan). "
                       "Dark = vibration dans la poitrine, poids, gravité.",

    symbolic_meaning="LUMIÈRE vs OBSCURITÉ. Bright = clarté mentale, révélation, conscience. "
                     "Dark = mystère, inconscient, profondeur, secret. "
                     "Un son qui devient plus bright = une vérité qui se révèle. "
                     "Un son qui devient plus dark = un mystère qui s'épaissit.",
    narrative_use="Bright = héroïque, triomphant. Dark = menaçant, mystérieux.",
    archetype="Le feu (bright) vs la terre (dark).",

    detection_method="Spectral centroid = moyenne pondérée des fréquences.",
    synthesis_method="Filtre passe-haut pour plus bright, passe-bas pour plus dark.",
    cogniarc_skill="spatial_inference — bright = contrastes élevés, dark = contrastes faibles",
))

_register(ParameterMap(
    parameter=AudioParameter.WARMTH,
    display_name="Warmth / Grave / Chaleur",
    unit="Ratio harmonique",
    range_human="Ratio 2-3e harmoniques vs fondamental",

    what_it_does="Richesse en basses fréquences + harmoniques impaires douces. "
                 "Warm = rond, charnu, enveloppant. Cold = maigre, distant.",
    emotional_effect="Warm → confort, nostalgie, intimité, humanité. "
                     "Cold → distance, technologie, isolement, précision.",
    physical_sensation="Warm = vibration dans la poitrine, étreinte sonore. "
                       "Cold = picotement dans la tête, distance.",

    symbolic_meaning="CHAUD vs FROID. Warm = vie, chair, battement de cœur, humanité. "
                     "Cold = machine, vide, espace, mort, pureté. "
                     "La transition warm→cold = un personnage qui perd son humanité. "
                     "Cold→warm = une machine qui s'éveille.",
    narrative_use="Warm = scènes intimes, flashbacks, humanité. Cold = IA, espace, antagonistes.",
    archetype="Le cœur qui bat (warm) vs le verre qui se brise (cold).",

    detection_method="Ratio d'énergie <500Hz vs >2000Hz + analyse harmonique.",
    synthesis_method="Filtre passe-bas doux + saturation tube (harmoniques paires).",
    cogniarc_skill="spatial_inference — warm = grandes régions homogènes, cold = détails fins",
))

_register(ParameterMap(
    parameter=AudioParameter.AIR,
    display_name="Air / Souffle",
    unit="dB above 8kHz",
    range_human="-40 dB à -10 dB dans la bande 8-20kHz",

    what_it_does="Présence d'hautes fréquences extrêmes. "
                 "Air = ouverture, espace, respiration. Pas d'air = fermé, terre-à-terre.",
    emotional_effect="Aérien → liberté, spiritualité, légèreté, transcendance. "
                     "Pas d'air → lourdeur, réalisme, oppression.",
    physical_sensation="Air = sensation de fraîcheur, vent sur la peau. "
                       "Pas d'air = chaleur étouffante, enfermement.",

    symbolic_meaning="ESPRIT vs MATIÈRE. Air = souffle vital, esprit, liberté, transcendance. "
                     "Pas d'air = matière, contrainte, corps, réalité. "
                     "L'air dans la voix = sincérité, vulnérabilité, émotion contenue.",
    narrative_use="Scènes célestes, spirituelles, flashbacks. Respiration avant un moment important.",
    archetype="Le souffle (air) vs la pierre (pas d'air).",

    detection_method="Mesure de l'énergie dans les bandes 8-16kHz.",
    synthesis_method="Bruit blanc filtré passe-haut + excitation de résonances.",
    cogniarc_skill="attention — l'air capte l'attention comme un détail saillant",
))

# ─── 2.3 SPATIAL ───

_register(ParameterMap(
    parameter=AudioParameter.REVERB_TIME,
    display_name="Reverb Time / Espace",
    unit="secondes (RT60)",
    range_human="0.2 s (petite pièce) à 8 s (cathédrale)",

    what_it_does="Temps que met le son à décroître de 60dB après la source. "
                 "Crée l'impression d'ESPACE. Courte = petite pièce. Longue = grand espace.",
    emotional_effect="Courte → intimité, présence, urgence, claustrophobie. "
                     "Longue → solennité, transcendance, vide, isolement, nostalgie. "
                     "Très longue → infini, divin, terreur cosmique.",
    physical_sensation="Courte = pression, proximité physique. "
                       "Longue = vertige, sensation de vide immense.",

    symbolic_meaning="L'ESPACE autour du son. Courte = ici, présent, concret. "
                     "Longue = ailleurs, passé, mémoire, infini. "
                     "Reverb qui augmente = un personnage qui entre dans un espace plus grand (physique ou mental). "
                     "Reverb qui disparaît = espace qui se referme, retour à la réalité.",
    narrative_use="Cathédrale = divin. Couloir = solitude. Chambre = intimité. "
                  "Caverne = mystère. Plein air = liberté.",
    archetype="Le temple (longue) vs la chambre (courte).",

    detection_method="Analyse de la décroissance spectrale après un transitoire.",
    synthesis_method="Convolution reverb + algorithmique (Room, Hall, Plate, Spring).",
    cogniarc_skill="spatial_inference — reverb_time = taille de la région",
))

_register(ParameterMap(
    parameter=AudioParameter.ECHO_FEEDBACK,
    display_name="Feedback d'Écho / Répétitions",
    unit="%",
    range_human="0% (pas d'écho) à 99% (auto-oscillation)",

    what_it_does="Nombre de répétitions de l'écho. Plus le feedback est haut, "
                 "plus le son se répète avant de disparaître.",
    emotional_effect="Peu de répétitions → écho naturel, distance mesurée. "
                     "Beaucoup de répétitions → obsession, piège temporel, folie. "
                     "Auto-oscillation (100%) → infini, perte de contrôle.",
    physical_sensation="Répétitions = sensation de poursuite, de ne pas pouvoir s'échapper.",

    symbolic_meaning="LA RÉPÉTITION. Peu = réverbération naturelle, distance. "
                     "Beaucoup = obsession, cycle, boucle temporelle, destin inéluctable. "
                     "Qui augmente = quelque chose qui RATTRAPE (le passé qui revient). "
                     "Qui diminue = quelque chose qui S'ÉLOIGNE définitivement.",
    narrative_use="Répétitions croissantes = folie qui s'installe. "
                  "Écho infini = malédiction, boucle temporelle.",
    archetype="Le destin qui se répète (écho) vs l'instant unique (pas d'écho).",

    detection_method="Comptage des répétitions après un pulse. Ratio d'amplitude inter-répétitions.",
    synthesis_method="Delay line + feedback loop. Paramètre 'feedback' du delay.",
    cogniarc_skill="temporal_inference — le feedback = le motif OSCILLATING ou CONSTANT",
))

# ─── 2.4 MODULATION ───

_register(ParameterMap(
    parameter=AudioParameter.VIBRATO_RATE,
    display_name="Vibrato / Tremblement",
    unit="Hz",
    range_human="0.5 Hz (lent) à 8 Hz (rapide)",

    what_it_does="Oscillation cyclique de la hauteur. "
                 "Lent = bercement, hésitation. Rapide = tremblement, excitation, peur.",
    emotional_effect="Lent (0.5-2 Hz) → calme, bercement, sensualité, hésitation. "
                     "Moyen (3-5 Hz) → émotion, passion, tension. "
                     "Rapide (>6 Hz) → anxiété, peur, froid, pathologique.",
    physical_sensation="Lent = balancement du corps. Rapide = chair de poule, tremblement.",

    symbolic_meaning="L'ÉMOTION QUI VIBRE. Lent = vie qui pulse calmement. "
                     "Rapide = vie qui tremble (peur, excitation, fièvre). "
                     "Vibrato qui accélère = peur qui monte. Vibrato qui ralentit = apaisement.",
    narrative_use="Voix avec vibrato lent = chaleur, humanité. "
                  "Vibrato rapide et froid = peur, maladie, possession.",
    archetype="Le battement de cœur (vibrato lent) vs le tremblement de peur (vibrato rapide).",

    detection_method="Détection de modulation de fréquence fondamentale (F0).",
    synthesis_method="LFO (Low Frequency Oscillator) sur la hauteur. "
                     "Paramètres : rate + depth.",
    cogniarc_skill="temporal_inference — CONSTANT (vibrato stable) vs ACCELERATING (qui s'emballe)",
))

_register(ParameterMap(
    parameter=AudioParameter.TREMOLO_RATE,
    display_name="Trémolo / Palpitation",
    unit="Hz",
    range_human="0.5 Hz à 20 Hz",

    what_it_does="Oscillation cyclique du VOLUME (pas de la hauteur). "
                 "Contrairement au vibrato, le trémolo coupe et rétablit le son.",
    emotional_effect="Lent → battement, pulsation, respiration. "
                     "Rapide → scintillement, instabilité, machine, effet stroboscopique.",
    physical_sensation="Trémolo lent = pulsation cardiaque ressentie. "
                       "Trémolo rapide = scintillement visuel, vibration.",

    symbolic_meaning="L'ALTERNANCE PRÉSENCE/ABSENCE. Trémolo = être et ne pas être. "
                     "Lent = respiration, vie cyclique. Rapide = clignotement, machine, instabilité. "
                     "Trémolo qui ralentit → mort qui approche (cœur qui faiblit). "
                     "Trémolo qui accélère → panique, course contre la montre.",
    narrative_use="Trémolo lent = cœur qui bat, tension qui monte. "
                  "Trémolo rapide = machine, alarme, urgence.",
    archetype="Le cœur qui bat (lent) vs l'alarme (rapide).",

    detection_method="Enveloppe AM (Amplitude Modulation). Détection de crêtes cycliques.",
    synthesis_method="LFO sur l'amplitude. Paramètres : rate + depth.",
    cogniarc_skill="temporal_inference — OSCILLATING pattern",
))

# ─── 2.5 PSYCHOACOUSTIQUE (ILLUSIONS) ───

_register(ParameterMap(
    parameter=AudioParameter.DOPPLER_SHIFT,
    display_name="Effet Doppler",
    unit="semi-tons de glissement",
    range_human="0 à ±3 semi-tons selon la vitesse",

    what_it_does="Variation de hauteur due au mouvement relatif source/auditeur. "
                 "Qui s'approche = son plus aigu. Qui s'éloigne = son plus grave. "
                 "Classique : train qui passe (eeeeeeeyyOOOOOuuuuuuum).",
    emotional_effect="Approche → anticipation, excitation, menace qui arrive. "
                     "Passage → climax, moment de bascule. "
                     "Éloignement → soulagement, tristesse, disparition.",
    physical_sensation="Sensation physique de mouvement dans l'espace. "
                       "Le son semble traverser le corps au moment du passage.",

    symbolic_meaning="LE MOUVEMENT, LE PASSAGE. Doppler = quelque chose qui VIENT puis REPART. "
                     "C'est le son du DESTIN qui passe : arrive, traverse, s'en va. "
                     "La courbe doppler elle-même raconte : tension (approche) → climax (passage) → résolution (départ).",
    narrative_use="Train, voiture, vaisseau, projectile. Tout objet en mouvement. "
                  "Aussi : révélation qui arrive puis ses conséquences.",
    archetype="Le train (destin qui passe). La comète (apparition → disparition).",

    detection_method="Glissando descendant après un pic de fréquence. "
                     "Détection de la courbe caractéristique f(t) = f0 / (1 - v·cos(θ)/c).",
    synthesis_method="Filtre + pitch shifter modulé par une courbe 1/x. "
                     "Simulation : approche → pitch monte → passage → pitch descend.",
    cogniarc_skill="temporal_inference + spatial_inference — le Doppler combine mouvement DANS le temps ET l'espace",
))

_register(ParameterMap(
    parameter=AudioParameter.SHEPARD_TONE,
    display_name="Tonalité de Shepard / Illusion infinie",
    unit="N/A (illusion)",
    range_human="Perception de hauteur infiniment croissante/décroissante",

    what_it_does="Illusion auditive d'une hauteur qui monte (ou descend) sans jamais arrêter. "
                 "Superposition de sinusoïdes espacées d'octaves, avec enveloppe spectrale en cloche "
                 "qui se déplace : les notes hautes s'éteignent pendant que les basses réapparaissent.",
    emotional_effect="Shepard ascendant → escalade infinie, espoir sans fin, impossible, obsession. "
                     "Shepard descendant → chute infinie, désespoir, vertige, sentiment d'impuissance. "
                     "Les deux → vertige existentiel.",
    physical_sensation="Vertige, désorientation, sensation de mouvement perpétuel sans progression réelle.",

    symbolic_meaning="L'ILLUSION DU PROGRÈS INFINI. Shepard = le syndrome de Sisyphe SONORE. "
                     "Tu as l'impression d'avancer, de monter, mais tu n'arrives JAMAIS au sommet. "
                     "C'est le bruit du capitalisme, de la course au progrès, de l'obsession. "
                     "Qui descend = dépression, spirale descendante, sans fond.",
    narrative_use="Montage qui stresse, obsession, boucle temporelle, "
                  "spirale descendante, rêve sans fin.",
    archetype="Sisyphe (ascension infinie) vs la spirale de l'enfer (descente infinie).",

    detection_method="Analyse spectrogramme : raies parallèles équidistantes (log scale) "
                     "avec enveloppe qui se déplace.",
    synthesis_method="Superposition de 4-6 sinusoïdes espacées d'octaves. "
                     "Enveloppe gaussienne qui glisse logarithmiquement.",
    cogniarc_skill="temporal_inference — le LOOP pattern, le mouvement perpétuel",
))

_register(ParameterMap(
    parameter=AudioParameter.MISSING_FUNDAMENTAL,
    display_name="Fondamental Manquant",
    unit="Hz (fondamental reconstruit)",
    range_human="Le cerveau RECONSTRUIT la fondamentale à partir des harmoniques",

    what_it_does="Phénomène où le cerveau PERÇOIT une fréquence fondamentale même si elle est absente "
                 "du signal, parce que ses harmoniques sont présentes. "
                 "Le cerveau calcule le PGCD des harmoniques pour retrouver la fondamentale.",
    emotional_effect="Stabilité rassurante : même incomplet, le cerveau COMPLÈTE. "
                     "C'est la preuve que la perception n'est pas passive mais CONSTRUCTIVE.",
    physical_sensation="Sensation de profondeur alors que la basse est absente. "
                       "Le corps 'sent' une fréquence qui n'existe pas physiquement.",

    symbolic_meaning="LA RECONSTRUCTION. Le tout est plus que la somme des parties. "
                     "Le cerveau COMPLÈTE ce qui manque. "
                     "Symbole de l'espoir, de la complétion, du sens qu'on donne à l'incomplétude. "
                     "Aussi : la présence invisible (Dieu, l'âme, le vide quantique).",
    narrative_use="Téléphone, haut-parleurs lo-fi : le cerveau complète les basses. "
                  "Métaphore de la complétion, de trouver le sens même quand il manque.",
    archetype="L'invisible qui façonne le visible. L'âme dans le corps.",

    detection_method="Analyse harmonique : si les harmoniques sont alignées (n×f, n×2f, n×3f...) "
                     "mais que la fondamentale n×1f est absente → fondamental manquant.",
    synthesis_method="Jouer uniquement les harmoniques 2, 3, 4, 5 sans la fondamentale. "
                     "Le cerveau reconstruit automatiquement.",
    cogniarc_skill="symbolic_inference — le cerveau infère le symbole manquant",
))

_register(ParameterMap(
    parameter=AudioParameter.BINAURAL_BEAT,
    display_name="Battements Binauraux",
    unit="Hz (différence de fréquence)",
    range_human="0.5 Hz à 30 Hz (correspond aux ondes cérébrales)",

    what_it_does="Deux fréquences LÉGÈREMENT différentes jouées dans chaque oreille. "
                 "Le cerveau perçoit une troisième fréquence = la DIFFÉRENCE entre les deux. "
                 "f_binaural = |f_gauche - f_droite|",
    emotional_effect="Delta (0.5-4 Hz) → sommeil profond, régénération. "
                     "Theta (4-8 Hz) → méditation, créativité, rêve éveillé. "
                     "Alpha (8-12 Hz) → relaxation, calme, concentration légère. "
                     "Beta (12-30 Hz) → concentration active, alerte, anxiété si trop haut. "
                     "Gamma (>30 Hz) → performance cognitive, insight.",
    physical_sensation="Sensation de pulsation dans la tête. "
                       "Le corps se synchronise progressivement à la fréquence dominante.",

    symbolic_meaning="LA SYNCHRONISATION. Deux légèrement décalés créent un TROISIÈME espace. "
                     "Symbole de la RÉSOLUTION harmonieuse de la différence. "
                     "Deux qui ne sont pas tout à fait d'accord mais dont la friction crée quelque chose de neuf. "
                     "Aussi : la conscience elle-même (le cerveau qui s'écoute).",
    narrative_use="Méditation, introspection, voyage mental, guérison, connexion.",
    archetype="La friction créatrice (deux mondes qui en créent un troisième).",

    detection_method="Pas de détection objective (phénomène psychoacoustique pur). "
                     "Simulation : deux tonalités pures avec Δf connu.",
    synthesis_method="Deux sinusoïdes avec fréquences f et f+Δ, une dans chaque canal stéréo. "
                     "Port de casque OBLIGATOIRE (l'illusion ne marche pas en haut-parleurs).",
    cogniarc_skill="temporal_inference — le pattern OSCILLATING créé par l'interaction de deux flux",
))

_register(ParameterMap(
    parameter=AudioParameter.PRECEDENCE_EFFECT,
    display_name="Effet de Précédence (Haas)",
    unit="ms",
    range_human="0 ms (même son) à 40 ms (deux sons distincts)",

    what_it_does="Quand deux oreilles entendent le même son à des temps légèrement différents (<40ms), "
                 "le cerveau LOCALISE le son du côté de la PREMIÈRE arrivée. "
                 "Le deuxième son est 'absorbé' dans la perception du premier.",
    emotional_effect="Stabilité spatiale : on sait d'où vient un son même entouré de réflexions. "
                     "Perte de l'effet = désorientation, confusion spatiale.",
    physical_sensation="Capacité à localiser une source les yeux fermés. "
                       "Base de la localisation auditive humaine.",

    symbolic_meaning="LA PREMIÈRE IMPRESSION. Ce qui arrive en premier définit la perception. "
                     "Les suivants sont absorbés dans le premier. "
                     "Symbole du PRÉJUGÉ : la première info écrase les suivantes. "
                     "Aussi : le leadership, l'initiative (le premier qui parle gagne).",
    narrative_use="Localisation d'une menace dans le noir. "
                  "Qui parle en premier dans une conversation = dominant.",
    archetype="Le premier pas (celui qui fait le premier, même imperceptiblement, gagne).",

    detection_method="Différence interaurale de temps (ITD) < 40ms = effet Haas. "
                     "Détection par cross-corrélation entre les canaux gauche et droit.",
    synthesis_method="Délai <40ms sur un canal + level compensatoire.",
    cogniarc_skill="attention — ce qui arrive en premier capture l'attention",
))

_register(ParameterMap(
    parameter=AudioParameter.AUDITORY_STREAMING,
    display_name="Streaming Auditif / Ségrégation",
    unit="N/A (phénomène perceptif)",
    range_human="Capacité à suivre UNE voix dans une foule (cocktail party)",

    what_it_does="Capacité du cerveau à SÉPARER plusieurs sources sonores simultanées "
                 "en flux distincts. Basée sur : hauteur, timbre, localisation, rythme. "
                 "Une voix dans une foule. Un violon dans un orchestre.",
    emotional_effect="Streaming réussi → contrôle, compréhension, orientation. "
                     "Streaming qui échoue (+ de 3 flux simultanés) → surcharge cognitive, stress, incompréhension.",
    physical_sensation="Le 'cocktail party effect' : on peut suivre UNE voix dans une pièce bruyante. "
                       "Épuisant sur la durée.",

    symbolic_meaning="L'ATTENTION SÉLECTIVE. La capacité à ISOLER UN SIGNAL du bruit. "
                     "Symbole de la discrimination, du discernement, de la focalisation. "
                     "L'incapacité à streamer = confusion, perte de repères, folie. "
                     "Le streaming parfait (entendre TOUT simultanément) = omniscience, extase mystique.",
    narrative_use="Scène de foule où le héros entend UN appel. "
                  "Téléphone qui sonne sous le bruit. "
                  "Perte de streaming = trouble mental (schizophrénie, surcharge sensorielle).",
    archetype="Le chamane qui entend une voix dans le chaos (streaming) vs la folie (perte de streaming).",

    detection_method="Analyse de scène auditive (CASA). "
                     "Séparation de sources par localisation + hauteur + timbre.",
    synthesis_method="Multiples sources avec des signatures spectrales et spatiales distinctes.",
    cogniarc_skill="attention — le focus de l'attention auditive est l'équivalent du crosshair visuel",
))


# ════════════════════════════════════════════════
#  3. CARTES THÉMATIQUES
# ════════════════════════════════════════════════

def carte_par_emotion() -> dict[str, list[str]]:
    """Inverse: d'une émotion → quels paramètres sonores la créent?"""
    return {
        "joie": ["brightness+", "vibrato lent", "attack rapide", "gain moyen+", "reverb courte"],
        "tristesse": ["brightness-", "reverb longue", "attack lent", "gain faible", "echo feedback+"],
        "peur": ["vibrato rapide", "tremolo+", "distortion+", "high freq soudaines", "silence+attaque brutale"],
        "colère": ["gain++", "distortion++", "brightness++", "attack très rapide", "compression forte"],
        "sérénité": ["warmth+", "reverb naturelle", "vibrato lent", "attack lent", "air+"],
        "mystère": ["reverb longue", "brightness-", "echo feedback+", "fréquences basses continues", "silence"],
        "urgence": ["tremolo rapide", "gain+", "attack très rapide", "fréquences mid aiguës", "rythme irrégulier"],
        "nostalgie": ["reverb longue", "warmth+", "attack lent", "gain faible", "air sur les aigus"],
        "amour": ["warmth++", "vibrato lent", "attack moyen", "gain moyen", "harmoniques riches"],
        "horreur": ["infrasons", "distortion+", "silence+attaque soudaine", "reverb très longue", "dissonance"],
    }


def carte_par_archetype() -> dict[str, list[str]]:
    """Inverse: d'un archétype → quels paramètres le représentent?"""
    return {
        "le héros": ["brightness+", "gain+", "attack ferme", "harmoniques riches", "reverb héroïque"],
        "le vilain": ["distortion+", "gain variable", "fréquences basses", "echo feedback+", "dissonance"],
        "la nature": ["warmth+", "air+", "gain modéré", "reverb naturelle", "fréquences irrégulières"],
        "la machine": ["tremolo régulier", "fréquences pures", "attack très précis", "pas de reverb", "bitcrush+"],
        "l'esprit": ["air++", "reverb très longue", "vibrato lent", "shepard tone", "binaural"],
        "la mort": ["silence", "infrasons", "fundamental manquant", "echo feedback qui décroît", "gain qui diminue"],
        "l'amour": ["warmth++", "vibrato lent", "attack doux", "harmoniques paires", "chorus+"],
        "le cosmos": ["reverb infinie", "shepard tone", "air++", "fréquences très basses", "fréquences très hautes"],
        "le temps": ["echo+", "tick régulier", "shepard tone", "doppler perpétuel"],
        "le destin": ["doppler", "gain crescendo", "fréquence unique qui approche", "silence puis impact"],
    }


# ════════════════════════════════════════════════
#  4. UTILITAIRE
# ════════════════════════════════════════════════

def afficher_carte(param: AudioParameter) -> str:
    """Affiche la carte complète d'un paramètre."""
    pm = SOUND_CARTOGRAPHY.get(param)
    if not pm:
        return f"Paramètre inconnu: {param}"
    
    lines = [
        f"\n{'═' * 60}",
        f"  🔊 {pm.display_name.upper()}  ({pm.parameter.value})",
        f"  Unité: {pm.unit} | Plage: {pm.range_human}",
        f"{'═' * 60}",
        f"\n  ▸ EFFET: {pm.what_it_does}",
        f"  ▸ ÉMOTION: {pm.emotional_effect}",
        f"  ▸ SENSATION: {pm.physical_sensation}",
        f"\n  ▸ SYMBOLE: {pm.symbolic_meaning}",
        f"  ▸ NARRATIF: {pm.narrative_use}",
        f"  ▸ ARCHÉTYPE: {pm.archetype}",
        f"\n  ▸ DÉTECTION: {pm.detection_method}",
        f"  ▸ SYNTHÈSE: {pm.synthesis_method}",
        f"  ▸ SKILL: {pm.cogniarc_skill}",
        f"{'═' * 60}",
    ]
    return "\n".join(lines)


def analyser_emotion(parametres: dict[str, float]) -> list[str]:
    """Analyse un ensemble de paramètres → quelles émotions sont suggérées."""
    scores: dict[str, float] = {}
    emo_map = carte_par_emotion()
    
    for emotion, signatures in emo_map.items():
        score = 0.0
        for sig in signatures:
            # Extraction du paramètre et de la direction (+/-)
            parts = sig.split()
            if len(parts) >= 2:
                direction = parts[-1]
                name = " ".join(parts[:-1])
                if name in parametres and direction in ("+", "-", "++", "--"):
                    val = parametres[name]
                    if direction == "+":
                        score += val if val > 0.5 else 0
                    elif direction == "++":
                        score += val * 2 if val > 0.7 else 0
                    elif direction == "-":
                        score += (1 - val) if val < 0.5 else 0
                    elif direction == "--":
                        score += (1 - val) * 2 if val < 0.3 else 0
        scores[emotion] = score
    
    # Top 3 émotions
    sorted_emos = sorted(scores.items(), key=lambda x: -x[1])
    return [e for e, s in sorted_emos[:3] if s > 0]


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        param_name = sys.argv[1].lower()
        for p in AudioParameter:
            if p.value == param_name:
                print(afficher_carte(p))
                break
        else:
            print(f"Paramètre '{param_name}' inconnu. Liste disponible:")
            for p in AudioParameter:
                print(f"  - {p.value}")
    print("Cartographie Sonore — 20+ paramètres audio catalogués")
    print(f"{'═' * 60}")
    for p in AudioParameter:
        pm = SOUND_CARTOGRAPHY.get(p)
        if pm:
            print(f"  🔊 {pm.display_name:25s} → {pm.symbolic_meaning[:60]}...")
    print(f"\nNombre de paramètres cartographiés: {len(SOUND_CARTOGRAPHY)}")
    print(f"Nombre d'émotions: {len(carte_par_emotion())}")
    print(f"Nombre d'archétypes: {len(carte_par_archetype())}")
    print(f"\nUsage: python3 audio_cartography.py <param_name>")
    print("Paramètres disponibles:")
    for p in sorted(SOUND_CARTOGRAPHY.keys(), key=lambda x: x.value):
        print(f"  - {p.value}")
