"""
Phase States of Matter — Complete phase diagrams, transitions, gas dynamics.
Covers solid, liquid, gas, and plasma with reversible transitions.
Designed for small LLM approximate reasoning about matter states.
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# ============================================================
# 1. PHASE TYPES & PROPERTIES
# ============================================================

class Phase(Enum):
    SOLID = "solid"             # Crystalline/amorphous — défini, peu compressible
    LIQUID = "liquid"           # Défini, peu compressible, prend la forme du contenant
    GAS = "gas"                 # Expansible, compressible, homogène
    PLASMA = "plasma"           # Ionisé, conducteur, hautes températures
    SUPERCRITICAL = "supercritical"  # Au-delà du point critique — gaz+ liquide


# ============================================================
# 2. SUBSTANCE DATABASE
# ============================================================

@dataclass
class SubstanceData:
    """Complete physical properties of a substance"""
    name: str
    molar_mass: float              # g/mol
    density_solid: float           # kg/m³ at 20°C
    density_liquid: float          # kg/m³ at melting point
    melting_point_k: float         # K (0°C = 273.15K)
    boiling_point_k: float         # K at 1 atm
    critical_temp_k: float         # K — above this, supercritical
    critical_pressure_pa: float    # Pa
    triple_point_temp_k: float     # K — all 3 phases coexist
    triple_point_pressure_pa: float
    latent_heat_fusion: float      # J/kg (solid ↔ liquid)
    latent_heat_vaporization: float # J/kg (liquid ↔ gas)
    latent_heat_sublimation: float # J/kg (solid ↔ gas)
    specific_heat_solid: float     # J/(kg·K)
    specific_heat_liquid: float    # J/(kg·K)
    specific_heat_gas: float       # J/(kg·K) at constant pressure
    plasma_threshold_k: float      # Temperature for ionization
    thermal_conductivity: float    # W/(m·K)
    thermal_expansion: float       # 1/K (coefficient)
    color_solid: str = "#888888"
    color_liquid: str = "#4488ff"
    color_gas: str = "#ffffff"


# Substance encyclopaedia
SUBSTANCES = {
    "water": SubstanceData(
        name="Water", molar_mass=18.015, density_solid=917, density_liquid=997,
        melting_point_k=273.15, boiling_point_k=373.15,
        critical_temp_k=647.1, critical_pressure_pa=22.064e6,
        triple_point_temp_k=273.16, triple_point_pressure_pa=611.73,
        latent_heat_fusion=334000, latent_heat_vaporization=2260000,
        latent_heat_sublimation=2835000,
        specific_heat_solid=2093, specific_heat_liquid=4184, specific_heat_gas=1996,
        plasma_threshold_k=5000, thermal_conductivity=0.598, thermal_expansion=2.07e-4,
        color_solid="#ccddff", color_liquid="#3388ff", color_gas="#eeeeff"
    ),
    "iron": SubstanceData(
        name="Iron", molar_mass=55.845, density_solid=7874, density_liquid=6980,
        melting_point_k=1811, boiling_point_k=3134,
        critical_temp_k=8350, critical_pressure_pa=200e6,
        triple_point_temp_k=1811, triple_point_pressure_pa=1e5,
        latent_heat_fusion=247000, latent_heat_vaporization=6340000,
        latent_heat_sublimation=6587000,
        specific_heat_solid=449, specific_heat_liquid=820, specific_heat_gas=1050,
        plasma_threshold_k=10000, thermal_conductivity=80, thermal_expansion=1.2e-5,
        color_solid="#888888", color_liquid="#ff6600", color_gas="#ff2200"
    ),
    "copper": SubstanceData(
        name="Copper", molar_mass=63.546, density_solid=8960, density_liquid=8020,
        melting_point_k=1357, boiling_point_k=2840,
        critical_temp_k=7600, critical_pressure_pa=180e6,
        triple_point_temp_k=1357, triple_point_pressure_pa=0.5,
        latent_heat_fusion=205000, latent_heat_vaporization=4730000,
        latent_heat_sublimation=4935000,
        specific_heat_solid=385, specific_heat_liquid=495, specific_heat_gas=1040,
        plasma_threshold_k=12000, thermal_conductivity=401, thermal_expansion=1.67e-5,
        color_solid="#cc8833", color_liquid="#ffaa44", color_gas="#ff4400"
    ),
    "wood": SubstanceData(
        name="Wood (Oak)", molar_mass=100, density_solid=700, density_liquid=900,
        melting_point_k=523, boiling_point_k=623,
        critical_temp_k=2500, critical_pressure_pa=5e6,
        triple_point_temp_k=500, triple_point_pressure_pa=1e5,
        latent_heat_fusion=200000, latent_heat_vaporization=2000000,
        latent_heat_sublimation=2200000,
        specific_heat_solid=2400, specific_heat_liquid=3200, specific_heat_gas=1500,
        plasma_threshold_k=3000, thermal_conductivity=0.17, thermal_expansion=5e-5,
        color_solid="#8B4513", color_liquid="#663300", color_gas="#444400"
    ),
    "gold": SubstanceData(
        name="Gold", molar_mass=196.97, density_solid=19320, density_liquid=17300,
        melting_point_k=1337, boiling_point_k=3243,
        critical_temp_k=7250, critical_pressure_pa=170e6,
        triple_point_temp_k=1337, triple_point_pressure_pa=0.01,
        latent_heat_fusion=63400, latent_heat_vaporization=1700000,
        latent_heat_sublimation=1763400,
        specific_heat_solid=129, specific_heat_liquid=157, specific_heat_gas=950,
        plasma_threshold_k=15000, thermal_conductivity=318, thermal_expansion=1.42e-5,
        color_solid="#ffcc00", color_liquid="#ffdd44", color_gas="#ff8800"
    ),
    "air": SubstanceData(
        name="Air", molar_mass=28.97, density_solid=0, density_liquid=875,
        melting_point_k=60, boiling_point_k=79,
        critical_temp_k=132.5, critical_pressure_pa=3.77e6,
        triple_point_temp_k=59, triple_point_pressure_pa=7.8e4,
        latent_heat_fusion=25000, latent_heat_vaporization=200000,
        latent_heat_sublimation=225000,
        specific_heat_solid=2000, specific_heat_liquid=2000, specific_heat_gas=1005,
        plasma_threshold_k=8000, thermal_conductivity=0.026, thermal_expansion=3.43e-3,
        color_solid="#8888ff", color_liquid="#aaaaff", color_gas="#ffffff"
    ),
    "ethanol": SubstanceData(
        name="Ethanol", molar_mass=46.07, density_solid=820, density_liquid=789,
        melting_point_k=159, boiling_point_k=351,
        critical_temp_k=514, critical_pressure_pa=6.3e6,
        triple_point_temp_k=158, triple_point_pressure_pa=4.3e-9,
        latent_heat_fusion=108000, latent_heat_vaporization=846000,
        latent_heat_sublimation=954000,
        specific_heat_solid=2100, specific_heat_liquid=2440, specific_heat_gas=1430,
        plasma_threshold_k=6000, thermal_conductivity=0.171, thermal_expansion=1.12e-3,
        color_solid="#ddddff", color_liquid="#aaddff", color_gas="#ffffff"
    ),
}


# ============================================================
# 3. PHASE DIAGRAM — Predict state from P and T
# ============================================================

class PhaseDiagram:
    """
    Compute phase boundaries and predict state at (P, T).
    Uses Clausius-Clapeyron for vaporization/sublimation boundaries.
    """
    
    def __init__(self, substance: SubstanceData):
        self.data = substance
        self.R = 8.314  # J/(mol·K)
    
    def phase_at(self, pressure_pa: float, temp_k: float) -> Phase:
        """Determine phase at given pressure and temperature"""
        d = self.data
        
        # Above critical point → supercritical
        if temp_k >= d.critical_temp_k and pressure_pa >= d.critical_pressure_pa:
            return Phase.SUPERCRITICAL
        
        # Plasma threshold
        if temp_k >= d.plasma_threshold_k:
            return Phase.PLASMA
        
        # Below triple point temp
        if temp_k <= d.triple_point_temp_k:
            if pressure_pa <= d.triple_point_pressure_pa:
                return Phase.GAS  # Sublimation
            return Phase.SOLID
        
        # Between melting and boiling
        if temp_k >= d.boiling_point_k:
            if pressure_pa >= 1e5:  # Above 1atm → maybe still boiling
                boiling_p = self.boiling_pressure_at(temp_k)
                if pressure_pa >= boiling_p:
                    return Phase.LIQUID
                return Phase.GAS
            return Phase.GAS
        
        if temp_k >= d.melting_point_k:
            # Liquid region (possibly gas if pressure is very low)
            boiling_p = self.boiling_pressure_at(temp_k) if temp_k < d.critical_temp_k else 1e10
            if pressure_pa >= boiling_p:
                return Phase.LIQUID
            return Phase.GAS
        
        return Phase.SOLID
    
    def boiling_pressure_at(self, temp_k: float) -> float:
        """
        Pressure at which boiling occurs for given temperature.
        Clausius-Clapeyron: ln(P₂/P₁) = (ΔH_vap/R)(1/T₁ - 1/T₂)
        """
        d = self.data
        if temp_k < d.triple_point_temp_k:
            return d.triple_point_pressure_pa
        if temp_k >= d.critical_temp_k:
            return d.critical_pressure_pa
        
        # P_atm = 101325 Pa at T_boiling
        delta_h = d.latent_heat_vaporization / (d.molar_mass / 1000)  # J/mol
        p_atm = 101325
        ratio = (delta_h / self.R) * (1 / d.boiling_point_k - 1 / temp_k)
        return p_atm * math.exp(ratio)
    
    def melting_pressure_at(self, temp_k: float) -> float:
        """
        Pressure required for melting at given temperature (simplified linear).
        """
        d = self.data
        slope = 0.1 * d.density_liquid / d.density_solid  # Approximate steepness
        return (temp_k - d.melting_point_k) * slope * 1000 + 101325
    
    def describe_phase_at(self, pressure_pa: float, temp_k: float) -> dict:
        """Full description of the phase state"""
        phase = self.phase_at(pressure_pa, temp_k)
        temp_c = temp_k - 273.15
        
        descriptions = {
            Phase.SOLID: f"🧊 Solide à {temp_c:.0f}°C",
            Phase.LIQUID: f"💧 Liquide à {temp_c:.0f}°C",
            Phase.GAS: f"💨 Gaz à {temp_c:.0f}°C",
            Phase.PLASMA: f"⚡ Plasma à {temp_c:.0f}°C (ionisé!)",
            Phase.SUPERCRITICAL: f"🌀 Supercritique — ni gaz ni liquide, P={pressure_pa/1e6:.1f}MPa, T={temp_c:.0f}°C"
        }
        
        return {
            "phase": phase.value,
            "temp_k": round(temp_k, 1),
            "temp_c": round(temp_c, 1),
            "pressure_pa": round(pressure_pa, 0),
            "description": descriptions.get(phase, f"❓ Phase inconnue à {temp_c:.0f}°C"),
            "phase_name": phase.name,
            "near_transition": self._near_transition(pressure_pa, temp_k)
        }
    
    def _near_transition(self, pressure_pa: float, temp_k: float) -> Optional[str]:
        """Is the substance near a phase transition?"""
        d = self.data
        epsilon = 5  # K
        
        if abs(temp_k - d.melting_point_k) < epsilon:
            return "↕️ Proche du point de fusion/solidification"
        if abs(temp_k - d.boiling_point_k) < epsilon:
            return "↕️ Proche du point d'ébullition/condensation"
        if abs(temp_k - d.triple_point_temp_k) < epsilon and \
           abs(pressure_pa - d.triple_point_pressure_pa) < 50000:
            return "📍 Au point triple — les 3 phases coexistent!"
        return None


# ============================================================
# 4. PHASE TRANSITION SYSTEM
# ============================================================

class PhaseTransitionEngine:
    """
    Simulates phase changes with energy balance.
    Converts thermal energy ↔ latent heat ↔ temperature change.
    """
    
    def __init__(self, substance: SubstanceData):
        self.data = substance
        self.phase: Phase = Phase.SOLID
        self.temp_k: float = substance.melting_point_k + 200  # Default: above melt
        self.mass_kg: float = 1.0
        self.pressure_pa: float = 101325  # 1 atm default
        self.thermal_energy: float = 0.0  # J
        self.history: List[str] = []
        
        # Initialize thermal energy at current temp
        self._init_energy()
    
    def _init_energy(self):
        """Start with energy corresponding to current temp and phase"""
        d = self.data
        E = 0.0
        temp = self.temp_k
        
        # Heat from 0K to melting point (solid)
        E = self.mass_kg * d.specific_heat_solid * min(temp, d.melting_point_k)
        
        if temp > d.melting_point_k:
            # Latent heat of fusion
            E += self.mass_kg * d.latent_heat_fusion
            # Heat in liquid phase
            E += self.mass_kg * d.specific_heat_liquid * (temp - d.melting_point_k)
        
        if temp > d.boiling_point_k:
            # Latent heat of vaporization
            E += self.mass_kg * d.latent_heat_vaporization
            # Heat in gas phase
            E += self.mass_kg * d.specific_heat_gas * (temp - d.boiling_point_k)
        
        if temp > d.plasma_threshold_k:
            E += self.mass_kg * 1e6  # Approximate ionization energy
        
        self.thermal_energy = E
        
        # Determine initial phase
        self.phase = self._phase_from_temp(temp)
    
    def _phase_from_temp(self, temp_k: float) -> Phase:
        d = self.data
        if temp_k >= d.plasma_threshold_k:
            return Phase.PLASMA
        if temp_k >= d.boiling_point_k:
            return Phase.GAS
        if temp_k >= d.melting_point_k:
            return Phase.LIQUID
        return Phase.SOLID
    
    def add_heat(self, joules: float):
        """Add heat energy, compute phase changes"""
        self.thermal_energy += joules
        self._update_state()
    
    def remove_heat(self, joules: float):
        """Remove heat energy, compute phase changes"""
        self.thermal_energy = max(0, self.thermal_energy - joules)
        self._update_state()
    
    def set_temperature(self, temp_k: float):
        """Set temperature directly (recompute energy)"""
        old = self.temp_k
        self.temp_k = temp_k
        self._init_energy()
        if abs(old - temp_k) > 5:
            phase_before = self.phase
            self.phase = self._phase_from_temp(temp_k)
            if phase_before != self.phase:
                self.history.append(
                    f"T passe de {old-273.15:.0f}°C à {temp_k-273.15:.0f}°C: "
                    f"{phase_before.value} → {self.phase.value}"
                )
    
    def _update_state(self):
        """Recompute temperature and phase from thermal energy"""
        d = self.data
        E = self.thermal_energy
        m = self.mass_kg
        
        # Track transitions for logging
        prev_phase = self.phase
        
        # Energy thresholds for each phase + transition
        E_solid_max = m * d.specific_heat_solid * d.melting_point_k
        E_fusion = m * d.latent_heat_fusion
        E_liquid_max = m * d.specific_heat_liquid * (d.boiling_point_k - d.melting_point_k)
        E_vaporization = m * d.latent_heat_vaporization
        
        if E <= E_solid_max:
            self.phase = Phase.SOLID
            self.temp_k = E / (m * d.specific_heat_solid)
        elif E <= E_solid_max + E_fusion:
            self.phase = Phase.SOLID  # Melting in progress
            self.temp_k = d.melting_point_k
            if prev_phase != Phase.SOLID:
                self.phase = Phase.LIQUID  # Actually transitioning
        elif E <= E_solid_max + E_fusion + E_liquid_max:
            self.phase = Phase.LIQUID
            liquid_E = E - E_solid_max - E_fusion
            self.temp_k = d.melting_point_k + liquid_E / (m * d.specific_heat_liquid)
        elif E <= E_solid_max + E_fusion + E_liquid_max + E_vaporization:
            self.phase = Phase.LIQUID  # Boiling in progress
            self.temp_k = d.boiling_point_k
        else:
            self.phase = Phase.GAS
            gas_E = E - E_solid_max - E_fusion - E_liquid_max - E_vaporization
            self.temp_k = d.boiling_point_k + gas_E / (m * d.specific_heat_gas)
            if self.temp_k >= d.plasma_threshold_k:
                self.phase = Phase.PLASMA
        
        if prev_phase != self.phase:
            self.history.append(
                f"Transition: {prev_phase.value} → {self.phase.value} "
                f"à {self.temp_k-273.15:.0f}°C (E={E:.0f}J)"
            )
    
    def describe(self) -> str:
        d = self.data
        phase_icons = {Phase.SOLID: "🧊", Phase.LIQUID: "💧",
                       Phase.GAS: "💨", Phase.PLASMA: "⚡",
                       Phase.SUPERCRITICAL: "🌀"}
        icon = phase_icons.get(self.phase, "❓")
        temp_c = self.temp_k - 273.15
        
        # Density at current phase
        if self.phase == Phase.SOLID:
            density = d.density_solid
        elif self.phase == Phase.LIQUID:
            density = d.density_liquid
        elif self.phase == Phase.GAS:
            density = self.pressure_pa * (d.molar_mass / 1000) / (8.314 * self.temp_k)
        else:
            density = 0.1
        
        volume = self.mass_kg / max(density, 0.01)
        
        return (
            f"{icon} {d.name} — {self.phase.value.upper()}\n"
            f"  T = {temp_c:.0f}°C  |  P = {self.pressure_pa/1e5:.2f} bar\n"
            f"  E_thermique = {self.thermal_energy/1000:.1f} kJ\n"
            f"  Densité = {density:.0f} kg/m³  |  Volume = {volume:.6f} m³\n"
            f"  Phases traversées: {len(self.history)} transitions"
        )
    
    def get_total_enthalpy(self) -> float:
        """Total heat content relative to 0K"""
        return self.thermal_energy


# ============================================================
# 5. GAS DYNAMICS — Ideal gas law + adiabatic processes
# ============================================================

class GasDynamics:
    """Ideal gas law and thermodynamic processes for gases"""
    
    R = 8.314  # J/(mol·K)
    
    @staticmethod
    def ideal_gas_law(pressure_pa: float = None, volume_m3: float = None,
                      n_mol: float = None, temp_k: float = None) -> dict:
        """PV = nRT — compute missing variable"""
        R = GasDynamics.R
        known = sum(1 for v in [pressure_pa, volume_m3, n_mol, temp_k] if v is not None)
        
        if known < 3:
            return {"error": f"Besoin de 3 variables sur 4 (connues: {known})"}
        
        if pressure_pa is None:
            pressure_pa = n_mol * R * temp_k / volume_m3
        elif volume_m3 is None:
            volume_m3 = n_mol * R * temp_k / pressure_pa
        elif n_mol is None:
            n_mol = pressure_pa * volume_m3 / (R * temp_k)
        elif temp_k is None:
            temp_k = pressure_pa * volume_m3 / (n_mol * R)
        
        return {
            "pressure_pa": round(pressure_pa, 0),
            "pressure_bar": round(pressure_pa / 1e5, 2),
            "volume_m3": round(volume_m3, 6),
            "n_mol": round(n_mol, 2),
            "temp_k": round(temp_k, 1),
            "temp_c": round(temp_k - 273.15, 1),
            "formula": "PV = nRT"
        }
    
    @staticmethod
    def adiabatic_compression(v1: float, v2: float, t1: float, 
                              gamma: float = 1.4) -> dict:
        """T₂ = T₁(V₁/V₂)^(γ-1) for adiabatic process"""
        if v2 <= 0:
            return {"error": "Volume final doit être > 0"}
        ratio = v1 / v2
        t2 = t1 * ratio ** (gamma - 1)
        work = GasDynamics.R * (t2 - t1) / (gamma - 1)  # J/mol
        
        return {
            "t2_k": round(t2, 1),
            "t2_c": round(t2 - 273.15, 1),
            "delta_T": round(t2 - t1, 1),
            "compression_ratio": round(ratio, 1),
            "work_per_mol_J": round(work, 0),
            "delta_T_caused_by": f"compression adiabatique ×{ratio:.1f}"
        }
    
    @staticmethod
    def gas_density(molar_mass: float, pressure_pa: float, temp_k: float) -> float:
        """ρ = PM/RT"""
        return pressure_pa * (molar_mass / 1000) / (GasDynamics.R * temp_k)


# ============================================================
# 6. LLM INTERFACE
# ============================================================

def analyze_substance(substance_name: str, pressure_pa: float = 101325,
                       temp_k: float = 293.15, mass_kg: float = 1.0) -> str:
    """Full analysis of a substance's state and behavior"""
    
    sub = SUBSTANCES.get(substance_name.lower())
    if not sub:
        available = ", ".join(SUBSTANCES.keys())
        return f"Substance '{substance_name}' inconnue. Disponibles: {available}"
    
    lines = [f"{'='*60}"]
    lines.append(f"  ANALYSE DE PHASE: {sub.name}")
    lines.append(f"{'='*60}")
    
    # Phase diagram
    diagram = PhaseDiagram(sub)
    phase_info = diagram.describe_phase_at(pressure_pa, temp_k)
    
    lines.append(f"\n📍 État actuel ({phase_info['temp_c']:.0f}°C, {pressure_pa/1e5:.2f}bar):")
    lines.append(f"  {phase_info['description']}")
    if phase_info['near_transition']:
        lines.append(f"  {phase_info['near_transition']}")
    
    # Properties
    lines.append(f"\n📊 Propriétés:")
    lines.append(f"  Masse molaire: {sub.molar_mass:.2f} g/mol")
    lines.append(f"  Masse: {mass_kg} kg")
    lines.append(f"  Point fusion: {sub.melting_point_k-273.15:.0f}°C")
    lines.append(f"  Point ébullition: {sub.boiling_point_k-273.15:.0f}°C")
    lines.append(f"  Point triple: {sub.triple_point_temp_k-273.15:.0f}°C @ {sub.triple_point_pressure_pa/100:.1f}mbar")
    lines.append(f"  Point critique: {sub.critical_temp_k-273.15:.0f}°C @ {sub.critical_pressure_pa/1e6:.1f}MPa")
    lines.append(f"  Seuil plasma: >{sub.plasma_threshold_k-273.15:.0f}°C")
    
    # Latent heats
    lines.append(f"\n🔥 Chaleurs latentes:")
    lines.append(f"  Fusion: {sub.latent_heat_fusion/1000:.0f} kJ/kg")
    lines.append(f"  Vaporisation: {sub.latent_heat_vaporization/1000:.0f} kJ/kg")
    lines.append(f"  Sublimation: {sub.latent_heat_sublimation/1000:.0f} kJ/kg")
    
    # Energy to change phase
    lines.append(f"\n🔋 Énergie nécessaire pour {mass_kg}kg:")
    lines.append(f"  Solide→Liquide: {mass_kg * sub.latent_heat_fusion/1000:.1f} kJ")
    lines.append(f"  Liquide→Gaz: {mass_kg * sub.latent_heat_vaporization/1000:.1f} kJ")
    lines.append(f"  Solide→Gaz: {mass_kg * sub.latent_heat_sublimation/1000:.1f} kJ")
    
    # Transition simulator
    engine = PhaseTransitionEngine(sub)
    engine.mass_kg = mass_kg
    engine.pressure_pa = pressure_pa
    engine.set_temperature(temp_k)
    
    lines.append(f"\n⏳ Historique des transitions de {sub.name}:")
    
    # Simulate heating from -50 to max
    engine.set_temperature(223.15)  # -50°C
    for t_c in range(-50, int(sub.plasma_threshold_k - 273.15), 100):
        engine.set_temperature(t_c + 273.15)
    
    if engine.history:
        for h in engine.history[-5:]:
            lines.append(f"  {h}")
    
    # Attempt phase transitions
    lines.append(f"\n🔄 Prédictions de transition:")
    temp_c = temp_k - 273.15
    
    if temp_k < sub.melting_point_k:
        delta = sub.melting_point_k - temp_k
        energy = mass_kg * sub.specific_heat_solid * delta
        lines.append(f"  Pour fondre: +{delta:.0f}°C ou +{energy/1000:.0f} kJ")
        lines.append(f"  Densité: {sub.density_solid} kg/m³ (solide)")
        lines.append(f"  Subit la sublimation si P < {sub.triple_point_pressure_pa/100:.0f} mbar")
    
    elif temp_k < sub.boiling_point_k:
        delta_m = temp_k - sub.melting_point_k
        delta_b = sub.boiling_point_k - temp_k
        energy = mass_kg * sub.specific_heat_liquid * delta_b + mass_kg * sub.latent_heat_vaporization
        lines.append(f"  Liquide: +{delta_b:.0f}°C jusqu'à ébullition")
        lines.append(f"  Pour gazéifier complet: +{energy/1000:.1f} kJ")
        lines.append(f"  Se solidifie si -{delta_m:.0f}°C (libère {mass_kg*sub.latent_heat_fusion/1000:.1f} kJ)")
    
    elif temp_k < sub.plasma_threshold_k:
        lines.append(f"  Gaz: se refroidit par détente adiabatique")
        lines.append(f"  Se condense à {sub.boiling_point_k-273.15:.0f}°C")
        lines.append(f"  Vers plasma au-delà de {sub.plasma_threshold_k-273.15:.0f}°C")
    
    else:
        lines.append(f"  ⚡ Plasma — état ionisé, conducteur électrique")
        lines.append(f"  Température extrême: {temp_c:.0f}°C")
    
    return "\n".join(lines)


# ============================================================
# 7. DEMO
# ============================================================

def demo():
    substances = ["water", "iron", "ethanol"]
    
    for sub_name in substances:
        print(f"\n{'='*60}")
        print(f"{'='*60}")
        
        # Room temperature
        print(analyze_substance(sub_name, pressure_pa=101325, temp_k=293.15, mass_kg=1.0))
        
        # High temperature
        print(f"\n  --- Même substance à 3000°C ---")
        print(f"  {PhaseDiagram(SUBSTANCES[sub_name]).describe_phase_at(101325, 3273.15)['description']}")
    
    # Gas dynamics demo
    print(f"\n{'='*60}")
    print("  LOI DES GAZ PARFAITS")
    print(f"{'='*60}")
    gas = GasDynamics.ideal_gas_law(pressure_pa=101325, n_mol=1.0, temp_k=293.15)
    print(f"  1 mole de gaz à 20°C et 1 atm → V = {gas['volume_m3']:.3f} m³ ({gas['volume_m3']*1000:.1f} L)")
    
    ad = GasDynamics.adiabatic_compression(24.5, 1.0, 293.15)
    print(f"  Compression adiabatique ×24.5: ΔT = {ad['delta_T']:.0f}°C")
    print(f"  Travail: {ad['work_per_mol_J']:.0f} J/mol")


if __name__ == "__main__":
    demo()
