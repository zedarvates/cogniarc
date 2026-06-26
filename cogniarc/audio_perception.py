#!/usr/bin/env python3
"""
Audio Perception — comprendre les jeux par le son.

Quand un jeu a du son, le son n'est PAS aléatoire. Il encode :
    - Les événements (collision, succès, échec)
    - Les transitions (ouverture de porte, activation)
    - Les états (mode danger, zone spéciale)
    - Le rythme (quand agir, quand attendre)

La vision voit des PIXELS qui changent.
L'audio entend des ÉVÉNEMENTS qui se produisent.

Un "ding" = succès. Un "buzz" = échec. Un "whoosh" = mouvement.
Le pattern temporel des sons révèle la structure du jeu.

Ce module analyse :
    1. Le SPECTRE audio (fréquences, harmoniques)
    2. Le RYTHME (tempo, motifs temporels)
    3. Les ÉVÉNEMENTS (transitoires, attaques)
    4. La CORRÉLATION audio-vidéo (quel son avec quel changement visuel)

Même sans jeu audio réel, ce module définit le SQUELETTE
de ce que serait une perception auditive pour ARC-AGI-3.
"""

from __future__ import annotations

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


# ══════════════════════════════════════════════════════════════
#  1.  TYPES DE SONS (ONTOLOGIE AUDITIVE)
# ══════════════════════════════════════════════════════════════


class SoundType(Enum):
    """Types de sons qu'un jeu ARC peut produire.

    Chaque type encode une INFORMATION sur l'état du jeu.
    """
    # ── Événements de base ──
    CLICK = "click"                   # Action/utilisation (bouton, levier)
    STEP = "step"                     # Déplacement (pas, glissement)
    IMPACT = "impact"                 # Collision (mur, objet)
    PICKUP = "pickup"                 # Ramassage d'objet
    
    # ── Résultats ──
    SUCCESS = "success"               # Niveau réussi, objectif atteint
    FAILURE = "failure"               # Échec, mort, game over
    ALERT = "alert"                   # Danger, activation ennemie
    
    # ── États ──
    AMBIENT = "ambient"               # Son continu (vent, moteur)
    WARNING = "warning"               # Timer, compte à rebours
    POWERUP = "powerup"               # Changement d'état spécial
    
    # ── Transitions ──
    WHOOSH = "whoosh"                 # Mouvement rapide
    DOOR = "door"                     # Ouverture/fermeture
    SWITCH = "switch"                 # Changement de mode
    
    # ── Patterns ──
    RHYTHM = "rhythm"                 # Pattern rythmique
    MELODY = "melody"                 # Mélodie (souvent = succès)
    ALARM = "alarm"                   # Signal répété urgent


@dataclass
class AudioEvent:
    """Un événement sonore détecté.

    Attributes:
        sound_type: Type de son
        timestamp: Quand le son s'est produit
        duration: Durée du son en secondes
        frequency: Fréquence dominante (Hz)
        amplitude: Volume (0-1)
        pattern: Motif temporel (bip unique, répété, continu)
    """
    sound_type: SoundType
    timestamp: float = 0.0
    duration: float = 0.0
    frequency: float = 0.0
    amplitude: float = 0.0
    pattern: str = ""  # "single", "repeat", "continuous", "crescendo"

    def describe(self) -> str:
        """Description lisible du son."""
        if self.amplitude < 0.3:
            volume = "faible"
        elif self.amplitude < 0.7:
            volume = "moyen"
        else:
            volume = "fort"

        freq_desc = ""
        if self.frequency > 0:
            if self.frequency < 200:
                freq_desc = "grave"
            elif self.frequency < 1000:
                freq_desc = "médium"
            else:
                freq_desc = "aigu"

        return f"[{self.sound_type.value}] {volume} {freq_desc} ({self.duration:.1f}s)"


# ══════════════════════════════════════════════════════════════
#  2.  SYMBOLES AUDITIFS
# ══════════════════════════════════════════════════════════════


class AudioSymbol(Enum):
    """Symboles audio — ce que le son RÉVÈLE sur le jeu.

    La vision voit des pixels qui changent de couleur.
    L'audio entend des succès, des échecs, des transitions.
    Ces symboles sont INVISIBLES dans la grille seule.
    """
    # Confirmations
    ACTION_APPLIED = "action_applied"         # "click" = l'action a marché
    ACTION_FAILED = "action_failed"           # "buzz" = action non valide
    LEVEL_COMPLETE = "level_complete"         # "jingle" = niveau réussi
    
    # Mouvements
    COLLISION = "collision"                   # "thud" = bloqué par mur
    MOVEMENT = "movement"                     # "step" = déplacement réussi
    TELEPORT = "teleport"                     # "whoosh" = déplacement rapide
    
    # États
    DANGER = "danger"                         # "alarm" = zone dangereuse
    POWERED = "powered"                       # "hum" = activation
    COUNTDOWN = "countdown"                   # "tick" = temps limité
    
    # Relations causales
    CAUSE_EFFECT = "cause_effect"             # Son après action = causalité
    SIMULTANEITY = "simultaneity"             # Deux sons = événement composé
    SEQUENCE = "sequence"                     # Pattern rythmique = séquence


@dataclass
class AudioCue:
    """Un indice audio qui aide à comprendre le jeu.

    Lien entre un son entendu et ce qu'il révèle sur le jeu.
    """
    symbol: AudioSymbol
    event: AudioEvent
    confidence: float = 0.5


# ══════════════════════════════════════════════════════════════
#  3.  ANALYSEUR AUDITIF
# ══════════════════════════════════════════════════════════════


class AudioPerception:
    """Perception auditive — analyser le son pour comprendre le jeu.

    Même sans micro / sans jeu audio réel, ce module définit :
        - Quels SONS peuvent exister dans un jeu ARC
        - Ce que chaque son RÉVÈLE sur l'état du jeu
        - Comment CORRÉLER sons et changements visuels
        - Comment les SONS améliorent la compréhension temporelle

    Quand un jeu aura du son, ce module sera prêt à l'analyser.
    """

    def __init__(self):
        self.events: list[AudioEvent] = []
        self.cues: list[AudioCue] = []
        self._sequence: list[tuple[float, SoundType]] = []

    # ── Simulation d'analyse audio (sans vrai micro) ──

    def analyze_waveform(self, samples: np.ndarray, sample_rate: int = 44100,
                         timestamps: Optional[list[float]] = None) -> list[AudioEvent]:
        """Analyse une forme d'onde audio et détecte les événements.

        Args:
            samples: Échantillons audio (float array, -1 à 1)
            sample_rate: Taux d'échantillonnage en Hz
            timestamps: Timestamps optionnels pour chaque échantillon

        Returns:
            Liste des événements audio détectés
        """
        self.events = []
        
        # Détection de transitoires (attaques soudaines)
        envelope = np.abs(samples)
        threshold = np.mean(envelope) + 2 * np.std(envelope)
        above_threshold = envelope > threshold
        
        # Trouver les groupes de samples au-dessus du seuil
        if np.any(above_threshold):
            # Simplification : analyse fréquentielle basique
            fft = np.fft.rfft(samples)
            freqs = np.fft.rfftfreq(len(samples), 1 / sample_rate)
            magnitude = np.abs(fft)
            
            if len(magnitude) > 0 and np.max(magnitude) > 0:
                dominant_freq = freqs[np.argmax(magnitude)]
                max_amp = float(np.max(np.abs(samples)))
                
                # Classifier le type de son par fréquence
                if dominant_freq < 200:
                    s_type = SoundType.IMPACT
                elif dominant_freq < 500:
                    s_type = SoundType.STEP
                elif dominant_freq < 2000:
                    if max_amp > 0.8:
                        s_type = SoundType.ALERT
                    else:
                        s_type = SoundType.CLICK
                else:
                    if max_amp > 0.7:
                        s_type = SoundType.SUCCESS
                    else:
                        s_type = SoundType.PICKUP
                
                event = AudioEvent(
                    sound_type=s_type,
                    frequency=float(dominant_freq),
                    amplitude=float(max_amp),
                    duration=float(len(samples) / sample_rate),
                )
                self.events.append(event)
                self._sequence.append((timestamps[0] if timestamps else 0, s_type))
        
        return self.events

    # ── Analyse rythmique ──

    def detect_rhythm(self, events: Optional[list[AudioEvent]] = None) -> dict:
        """Détecte les patterns rythmiques dans une séquence d'événements.

        Le rythme révèle la STRUCTURE TEMPORELLE du jeu :
            - Bip régulier = timer
            - Accélération = urgence
            - Pattern répété = séquence d'actions
        """
        events = events or self.events
        if len(events) < 2:
            return {"pattern": "unknown", "tempo": 0}

        # Calculer les intervalles entre événements
        intervals = []
        for i in range(1, min(len(events), 10)):
            dt = events[i].timestamp - events[i - 1].timestamp
            if dt > 0:
                intervals.append(dt)

        if not intervals:
            return {"pattern": "single_event", "tempo": 0}

        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)

        if std_interval / max(mean_interval, 0.001) < 0.2:
            # Intervalles réguliers = rythme constant
            return {
                "pattern": "regular",
                "tempo": 60.0 / mean_interval if mean_interval > 0 else 0,
                "interval": mean_interval,
                "regularity": 1.0 - std_interval / max(mean_interval, 0.001),
            }
        elif len(events) >= 3:
            # Vérifier l'accélération
            if intervals[0] > intervals[-1] * 1.5:
                return {"pattern": "accelerating", "tempo_desc": "urgence croissante"}
            elif intervals[0] * 1.5 < intervals[-1]:
                return {"pattern": "decelerating", "tempo_desc": "ralentissement"}

        return {"pattern": "irregular", "tempo": 60.0 / np.mean(intervals) if intervals else 0}

    # ── Corrélation audio-vidéo ──

    def correlate_with_delta(self, event: AudioEvent,
                             delta_changed: np.ndarray) -> AudioCue:
        """Corrèle un son avec un changement visuel.

        Si un son se produit EN MÊME TEMPS qu'un changement visuel,
        ils sont probablement liés par une relation de cause à effet.

        Args:
            event: L'événement audio
            delta_changed: Masque des pixels qui ont changé

        Returns:
            Un indice audio avec confiance
        """
        pixels_changed = np.sum(delta_changed) if delta_changed.size > 0 else 0

        if event.sound_type == SoundType.CLICK and pixels_changed > 0:
            return AudioCue(
                symbol=AudioSymbol.ACTION_APPLIED,
                event=event,
                confidence=0.9,
            )
        elif event.sound_type == SoundType.IMPACT and pixels_changed == 0:
            return AudioCue(
                symbol=AudioSymbol.COLLISION,
                event=event,
                confidence=0.85,
            )
        elif event.sound_type == SoundType.SUCCESS:
            return AudioCue(
                symbol=AudioSymbol.LEVEL_COMPLETE,
                event=event,
                confidence=0.95,
            )
        elif event.sound_type == SoundType.ALERT:
            return AudioCue(
                symbol=AudioSymbol.DANGER,
                event=event,
                confidence=0.8,
            )
        else:
            return AudioCue(
                symbol=AudioSymbol.CAUSE_EFFECT,
                event=event,
                confidence=0.5,
            )

    # ── Rapport ──

    def get_insights(self) -> list[str]:
        """Retourne les insights audio sous forme de texte lisible.

         Ces INSIGHTS sont des informations que la VISION SEULE
         ne peut pas donner.
        """
        insights = []

        for cue in self.cues:
            if cue.confidence > 0.7:
                if cue.symbol == AudioSymbol.ACTION_APPLIED:
                    insights.append(f"✅ Action confirmée par son (confiance {cue.confidence:.0%})")
                elif cue.symbol == AudioSymbol.COLLISION:
                    insights.append(f"🧱 Collision détectée à l'audio — action bloquée")
                elif cue.symbol == AudioSymbol.LEVEL_COMPLETE:
                    insights.append(f"🏆 Niveau réussi ! (signal audio de succès)")
                elif cue.symbol == AudioSymbol.DANGER:
                    insights.append(f"⚠️  Alerte sonore — zone dangereuse")

        if not insights:
            insights.append("🔇 Aucun signal audio interprété")

        return insights


# ══════════════════════════════════════════════════════════════
#  4.  DÉMO CONCEPTUELLE
# ══════════════════════════════════════════════════════════════


def demo():
    """Démo conceptuelle : ce que le son révèle que la vision ne voit pas."""
    print("🔊 Audio Perception — comprendre les jeux par le son")
    print("=" * 55)
    print()

    print("🎯 Ce que le son révèle, que la VISION SEULE ne voit pas :")
    print()
    print("  Vision : des pixels changent de (0,0,0) à (255,0,0)")
    print("  Audio :  'BOOM' → collision avec un mur")
    print()
    print("  Vision : la grille reste identique")
    print("  Audio :  'DING DING DING' → timer, urgence croissante")
    print()
    print("  Vision : un pixel apparaît à (5,3)")
    print("  Audio :  'POP' → un objet a été créé")
    print()

    # Scénario : simulation d'audio
    print("1️⃣  Simulation d'un niveau avec audio")
    print()

    # Simuler 3 événements audio
    events = [
        AudioEvent(SoundType.CLICK, timestamp=0.1, duration=0.05, 
                   frequency=800, amplitude=0.6, pattern="single"),
        AudioEvent(SoundType.SUCCESS, timestamp=2.5, duration=0.5,
                   frequency=2000, amplitude=0.9, pattern="crescendo"),
        AudioEvent(SoundType.ALERT, timestamp=5.0, duration=0.3,
                   frequency=400, amplitude=0.8, pattern="repeat"),
    ]

    ap = AudioPerception()
    ap.events = events

    for ev in events:
        print(f"   🔊 {ev.describe()}")
    print()

    # Détection de rythme
    rhythm = ap.detect_rhythm()
    print(f"2️⃣  Analyse rythmique : {rhythm['pattern']}")
    print()

    # Corrélation avec des changements visuels simulés
    print("3️⃣  Corrélation audio → insights :")
    for ev in events:
        # Simuler un delta visuel
        delta = np.ones((10, 10), dtype=bool) if ev.sound_type != SoundType.ALERT else np.zeros((10, 10), dtype=bool)
        cue = ap.correlate_with_delta(ev, delta)
        ap.cues.append(cue)
        print(f"   🔗 {ev.sound_type.value:<10} → {cue.symbol.value:<20} "
              f"(confiance: {cue.confidence:.0%})")
    print()

    # Résumé
    print("4️⃣  Insights audio :")
    for insight in ap.get_insights():
        print(f"   {insight}")
    print()

    print("🧠 Ce que la vision PERD sans l'audio :")
    print("   - Savoir si une action a réussi AVANT de voir le résultat")
    print("   - Détecter une collision même si le visuel ne change pas")
    print("   - Comprendre le rythme du jeu (quand agir vite)")
    print("   - Distinguer un événement important d'un bruit de fond")
    print()
    print("🔮 Vision + Audio = perception multi-modale")
    print("   Comme un humain : yeux + oreilles = compréhension complète")


if __name__ == "__main__":
    demo()
