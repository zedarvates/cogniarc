"""
Advanced Physics Extensions — Elasticity, Resonance, Hydraulics, Waves, Chaos.
8 domain-specific physics engines for LLM approximate reasoning.
Each is a standalone expert module with rules + computation.
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# ============================================================
# 1. ELASTICITY & PLASTICITY — Deformation, rupture, fatigue
# ============================================================

class DeformationRegime(Enum):
    ELASTIC = "elastic"            # Reversible — Hooke's law
    PLASTIC = "plastic"            # Permanent deformation
    RUPTURE = "rupture"            # Material fails
    FATIGUE = "fatigue"            # Weakening from cyclic loading
    CREEP = "creep"                # Slow deformation under constant load


MATERIAL_STRENGTH = {
    "acier_doux":     {"E": 210e9, "sigma_y": 250e6, "sigma_u": 400e6, "epsilon_rupture": 0.20, "fatigue_limit": 200e6},
    "acier_haute_res": {"E": 210e9, "sigma_y": 800e6, "sigma_u": 1000e6, "epsilon_rupture": 0.08, "fatigue_limit": 400e6},
    "alu":            {"E": 70e9,  "sigma_y": 100e6, "sigma_u": 200e6, "epsilon_rupture": 0.15, "fatigue_limit": 70e6},
    "bois_chene":     {"E": 12e9,  "sigma_y": 50e6,  "sigma_u": 90e6,  "epsilon_rupture": 0.01, "fatigue_limit": 30e6},
    "bois_pin":       {"E": 9e9,   "sigma_y": 30e6,  "sigma_u": 60e6,  "epsilon_rupture": 0.008, "fatigue_limit": 20e6},
    "beton":          {"E": 30e9,  "sigma_y": 25e6,  "sigma_u": 30e6,  "epsilon_rupture": 0.003, "fatigue_limit": 10e6},
    "verre":          {"E": 70e9,  "sigma_y": 50e6,  "sigma_u": 50e6,  "epsilon_rupture": 0.001, "fatigue_limit": 20e6},
    "caoutchouc":     {"E": 0.01e9,"sigma_y": 15e6,  "sigma_u": 25e6,  "epsilon_rupture": 5.0,   "fatigue_limit": 5e6},
    "nylon":          {"E": 3e9,   "sigma_y": 50e6,  "sigma_u": 80e6,  "epsilon_rupture": 0.50,  "fatigue_limit": 25e6},
    "os_humain":      {"E": 15e9,  "sigma_y": 120e6, "sigma_u": 170e6, "epsilon_rupture": 0.02,  "fatigue_limit": 50e6},
}


@dataclass
class BeamAnalysis:
    """Analyze a beam under load"""
    material: str
    length: float              # m
    width: float               # m
    height: float              # m
    load: float                # N
    load_position: float = 0.5  # Fraction of length (0.5 = center)
    support_type: str = "simply_supported"  # "simply_supported", "cantilever", "fixed_both"
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def moment_of_inertia(self) -> float:
        """I = bh³/12 for rectangular beam"""
        return (self.width * self.height**3) / 12
    
    @property
    def section_modulus(self) -> float:
        """Z = bh²/6"""
        return (self.width * self.height**2) / 6
    
    def max_bending_moment(self) -> float:
        """Maximum bending moment based on support type"""
        L = self.length
        P = self.load
        a = self.load_position  # fraction
        
        if self.support_type == "cantilever":
            return P * L * a  # Moment at fixed end
        elif self.support_type == "simply_supported":
            if abs(a - 0.5) < 0.01:
                return P * L / 4  # Center load
            return P * L * a * (1 - a)  # Off-center
        elif self.support_type == "fixed_both":
            return P * L / 8
        return P * L / 4
    
    def max_deflection(self) -> float:
        """Maximum beam deflection"""
        E = MATERIAL_STRENGTH[self.material]["E"]
        I = self.moment_of_inertia
        L = self.length
        P = self.load
        a = self.load_position
        
        if self.support_type == "cantilever":
            return (P * (L * a)**3) / (3 * E * I)
        elif self.support_type == "simply_supported":
            if abs(a - 0.5) < 0.01:
                return (P * L**3) / (48 * E * I)
            b = L * (1 - a)
            return (P * L * a * b * (L + b) * math.sqrt(3 * a * (L + b))) / (27 * E * I * L) if a * b > 0 else 0
        elif self.support_type == "fixed_both":
            return (P * L**3) / (192 * E * I)
        return 0
    
    def max_stress(self) -> float:
        """σ_max = M_max / Z"""
        Z = self.section_modulus
        M = self.max_bending_moment()
        return M / max(Z, 1e-10)
    
    def safety_factor(self) -> float:
        """Safety factor relative to yield"""
        mat = MATERIAL_STRENGTH[self.material]
        return mat["sigma_y"] / max(self.max_stress(), 1.0)
    
    def regime(self) -> DeformationRegime:
        mat = MATERIAL_STRENGTH[self.material]
        stress = self.max_stress()
        if stress > mat["sigma_u"]:
            return DeformationRegime.RUPTURE
        elif stress > mat["sigma_y"]:
            return DeformationRegime.PLASTIC
        else:
            return DeformationRegime.ELASTIC
    
    def fatigue_cycles_to_failure(self, stress_amplitude: float = None) -> float:
        """Approximate S-N curve: N = (σ_f / σ_a)^k"""
        mat = MATERIAL_STRENGTH[self.material]
        if stress_amplitude is None:
            stress_amplitude = self.max_stress()
        
        sigma_f = mat["sigma_u"]
        fatigue_limit = mat["fatigue_limit"]
        
        if stress_amplitude <= fatigue_limit:
            return float('inf')  # Infinite life
        
        # Basquin's law: σ_a = σ_f' (2N)^b
        k = 8.0  # Typical for metals
        return (sigma_f / stress_amplitude) ** k / 2
    
    def analyze(self) -> dict:
        mat = MATERIAL_STRENGTH[self.material]
        stress = self.max_stress()
        deflection = self.max_deflection()
        sf = self.safety_factor()
        regime = self.regime()
        
        # Load capacity
        max_load = self.load * sf
        
        # Visual verdict
        if regime == DeformationRegime.RUPTURE:
            verdict = "⚠️ RUPTURE — la poutre CÈDE"
        elif regime == DeformationRegime.PLASTIC:
            verdict = "⚠️ DÉFORMATION PERMANENTE — la poutre se tord"
        elif sf < 2.0:
            verdict = f"⚡ ÉLASTIQUE mais proche de la limite (FS={sf:.1f})"
        elif sf < 5.0:
            verdict = f"✅ ÉLASTIQUE — sécurité standard (FS={sf:.1f})"
        else:
            verdict = f"✅ TRÈS SÛR — large marge (FS={sf:.1f})"
        
        return {
            "material": self.material,
            "E_GPa": round(mat["E"] / 1e9, 1),
            "sigma_y_MPa": round(mat["sigma_y"] / 1e6, 0),
            "stress_MPa": round(stress / 1e6, 1),
            "deflection_mm": round(deflection * 1000, 2),
            "safety_factor": round(sf, 1),
            "regime": regime.value,
            "verdict": verdict,
            "max_safe_load_N": round(max_load, 0),
            "fatigue_cycles": round(self.fatigue_cycles_to_failure(), 0) if self.fatigue_cycles_to_failure() < 1e9 else "illimité"
        }


# ============================================================
# 2. RESONANCE — Natural frequency, forced oscillation
# ============================================================

@dataclass
class Oscillator:
    """Mass-spring-damper system"""
    mass: float              # kg
    stiffness: float         # N/m
    damping: float = 0.0     # N·s/m
    initial_displacement: float = 0.0
    
    @property
    def natural_frequency(self) -> float:
        """ω_n = √(k/m) rad/s"""
        return math.sqrt(self.stiffness / max(self.mass, 1e-10))
    
    @property
    def natural_frequency_hz(self) -> float:
        return self.natural_frequency / (2 * math.pi)
    
    @property
    def damping_ratio(self) -> float:
        """ζ = c / (2√(km))"""
        critical = 2 * math.sqrt(self.stiffness * self.mass)
        return self.damping / max(critical, 1e-10)
    
    @property
    def damped_frequency(self) -> float:
        """ω_d = ω_n √(1-ζ²)"""
        zeta = self.damping_ratio
        if zeta >= 1:
            return 0  # Overdamped, no oscillation
        return self.natural_frequency * math.sqrt(1 - zeta**2)
    
    def amplitude_at_frequency(self, driving_freq: float, driving_force: float) -> float:
        """Frequency response: X = F₀ / (k √((1-r²)² + (2ζr)²)) where r = ω/ω_n"""
        omega = driving_freq * 2 * math.pi
        omega_n = self.natural_frequency
        zeta = self.damping_ratio
        
        if omega_n < 1e-10:
            return 0
        
        r = omega / omega_n
        denominator = self.stiffness * math.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
        return driving_force / max(denominator, 1e-10)
    
    def resonance_amplification(self, driving_freq: float) -> float:
        """Q factor at given frequency"""
        omega = driving_freq * 2 * math.pi
        omega_n = self.natural_frequency
        zeta = self.damping_ratio
        
        if omega_n < 1e-10:
            return 1.0
        
        r = omega / omega_n
        return 1.0 / max(math.sqrt((1 - r**2)**2 + (2 * zeta * r)**2), 1e-10)
    
    def is_near_resonance(self, driving_freq: float, tolerance: float = 0.1) -> bool:
        return abs(driving_freq - self.natural_frequency_hz) / max(self.natural_frequency_hz, 1e-10) < tolerance
    
    def classify(self) -> str:
        zeta = self.damping_ratio
        if zeta < 0.1:
            return f"sous-amorti (oscille {self.damped_frequency:.1f} rad/s)"
        elif zeta < 1.0:
            return f"amorti (retour à l'équilibre sans oscillation)"
        elif zeta < 1.1:
            return f"critiquement amorti (retour le plus rapide)"
        else:
            return f"sur-amorti (retour très lent)"
    
    def analyze(self, driving_freq: float = None, driving_force: float = 100) -> dict:
        omega = self.natural_frequency
        freq_hz = self.natural_frequency_hz
        zeta = self.damping_ratio
        
        result = {
            "natural_freq_hz": round(freq_hz, 2),
            "natural_freq_rad_s": round(omega, 2),
            "period_s": round(1 / max(freq_hz, 1e-10), 3),
            "damping_ratio": round(zeta, 3),
            "class": self.classify(),
            "damped_freq_rad_s": round(self.damped_frequency, 2) if self.damped_frequency > 0 else 0,
        }
        
        if driving_freq is not None:
            result["driving_freq_hz"] = round(driving_freq, 2)
            result["is_near_resonance"] = self.is_near_resonance(driving_freq)
            result["amplification"] = round(self.resonance_amplification(driving_freq), 1)
            result["amplitude_m"] = round(self.amplitude_at_frequency(driving_freq, driving_force), 4)
            
            if self.is_near_resonance(driving_freq):
                if zeta < 0.05:
                    result["danger"] = "⚠️ RÉSONANCE CATASTROPHIQUE — amplitudes explosives (Tacoma!)"
                else:
                    result["danger"] = "⚡ Proche de la résonance — amplitudes amplifiées"
            else:
                result["danger"] = "✅ Hors résonance"
        
        return result


# ============================================================
# 3. HYDRAULICS — Pascal, force multiplication
# ============================================================

@dataclass
class HydraulicSystem:
    """Pascal's principle: F₁/A₁ = F₂/A₂"""
    piston1_area: float       # m² (master cylinder)
    piston2_area: float       # m² (slave cylinder)
    input_force: float = 100  # N
    fluid_density: float = 1000  # kg/m³
    height_difference: float = 0  # m (positive if output is higher)
    
    @property
    def pressure(self) -> float:
        """P = F₁/A₁ (Pa)"""
        return self.input_force / max(self.piston1_area, 1e-10)
    
    @property
    def output_force(self) -> float:
        """F₂ = P × A₂ - ρghA₂ (minus hydrostatic)"""
        hydrostatic = self.fluid_density * 9.81 * self.height_difference
        return self.pressure * self.piston2_area - hydrostatic * self.piston2_area
    
    @property
    def mechanical_advantage(self) -> float:
        """MA = A₂/A₁"""
        return self.piston2_area / max(self.piston1_area, 1e-10)
    
    @property
    def displacement_ratio(self) -> float:
        """d₁/d₂ = A₂/A₁ (conservation of volume)"""
        return self.mechanical_advantage
    
    def output_displacement(self, input_displacement: float) -> float:
        """V₁ = A₁d₁ = A₂d₂ = V₂ → d₂ = d₁ × A₁/A₂"""
        return input_displacement / max(self.mechanical_advantage, 1e-10)
    
    def analyze(self) -> dict:
        ma = self.mechanical_advantage
        return {
            "pressure_Pa": round(self.pressure, 0),
            "pressure_bar": round(self.pressure / 1e5, 3),
            "output_force_N": round(self.output_force, 0),
            "mechanical_advantage": round(ma, 1),
            "type": "multiplicateur de force" if ma > 1 else "multiplicateur de vitesse/déplacement" if ma < 1 else "transmission 1:1",
            "example_10cm_input": f"déplace {self.output_displacement(0.1)*1000:.1f}mm en sortie"
        }


# ============================================================
# 4. WAVES — Propagation, reflection, interference
# ============================================================

class WaveType(Enum):
    SOUND = "sound"
    WATER = "water"
    SEISMIC_P = "seismic_p"
    SEISMIC_S = "seismic_s"
    STRING = "string"
    LIGHT = "light"


@dataclass
class Wave:
    """Generic wave with propagation, reflection, interference"""
    wave_type: WaveType
    frequency: float           # Hz
    amplitude: float           # m (or Pa for sound)
    medium: str = "air"        # Affects speed
    
    @property
    def speed(self) -> float:
        """Wave speed in medium (m/s)"""
        speeds = {
            WaveType.SOUND: {"air": 343, "eau": 1480, "acier": 5100, "bois": 3500},
            WaveType.WATER: {"mer": 1.5, "tsunami": 200},
            WaveType.SEISMIC_P: {"roche": 6000, "sol": 2000},
            WaveType.STRING: {"acier": 200, "nylon": 100},
            WaveType.LIGHT: {"air": 3e8, "eau": 2.25e8, "verre": 2e8},
        }
        medium_speeds = speeds.get(self.wave_type, {})
        return medium_speeds.get(self.medium, 343)
    
    @property
    def wavelength(self) -> float:
        """λ = v/f"""
        return self.speed / max(self.frequency, 1e-10)
    
    @property 
    def wavenumber(self) -> float:
        """k = 2π/λ"""
        return 2 * math.pi / max(self.wavelength, 1e-10)
    
    @property
    def angular_frequency(self) -> float:
        return 2 * math.pi * self.frequency
    
    @property
    def intensity(self) -> float:
        """I ∝ A² (simplified)"""
        return self.amplitude**2
    
    def pressure_amplitude(self) -> float:
        """Sound pressure: p = vρωA (simplified)"""
        if self.wave_type == WaveType.SOUND:
            rho = 1.225  # air density
            return self.speed * rho * self.angular_frequency * self.amplitude
        return 0
    
    def db_spl(self) -> float:
        """Sound pressure level in dB"""
        p = self.pressure_amplitude()
        if p < 1e-10:
            return 0
        return 20 * math.log10(p / 2e-5)  # Ref: 20µPa
    
    def reflect(self, medium2: str) -> float:
        """Reflection coefficient: R = (Z₂-Z₁)/(Z₂+Z₁)"""
        speeds = {
            WaveType.SOUND: {"air": 343, "eau": 1480, "acier": 5100},
        }
        medium_speeds = speeds.get(self.wave_type, {})
        v1 = medium_speeds.get(self.medium, 343)
        v2 = medium_speeds.get(medium2, 343)
        
        rho1 = 1.225 if self.medium == "air" else 1000 if self.medium == "eau" else 7800
        rho2 = 1.225 if medium2 == "air" else 1000 if medium2 == "eau" else 7800
        
        Z1 = rho1 * v1
        Z2 = rho2 * v2
        
        R = (Z2 - Z1) / (Z2 + Z1)
        return round(R, 3)
    
    def interfere(self, other: "Wave", distance: float, phase_diff: float = 0) -> float:
        """Superposition: A_total = √(A₁²+A₂²+2A₁A₂cos(Δφ))"""
        delta_phi = 2 * math.pi * distance / max(self.wavelength, 1e-10) + phase_diff
        return math.sqrt(
            self.amplitude**2 + other.amplitude**2 + 
            2 * self.amplitude * other.amplitude * math.cos(delta_phi)
        )
    
    def doppler_shift(self, source_speed: float, observer_speed: float) -> float:
        """f' = f × (v ± v_o)/(v ∓ v_s)"""
        v = self.speed
        num = v + observer_speed
        den = v - source_speed
        return self.frequency * num / max(den, 1e-10)
    
    def analyze(self) -> dict:
        return {
            "type": self.wave_type.value,
            "frequency_Hz": round(self.frequency, 1),
            "wavelength_m": round(self.wavelength, 2),
            "speed_ms": round(self.speed, 0),
            "amplitude": round(self.amplitude, 4),
            "db_spl": round(self.db_spl(), 1) if self.wave_type == WaveType.SOUND else None,
            "period_ms": round(1000 / max(self.frequency, 1e-10), 1)
        }


# ============================================================
# 5. CHAOS — Sensitivity to initial conditions
# ============================================================

class ChaosAnalyzer:
    """Analyze chaotic systems: diverging trajectories, Lyapunov exponents"""
    
    @staticmethod
    def logistic_map(x0: float, r: float, n_iter: int = 50) -> List[float]:
        """x_{n+1} = r · x_n · (1 - x_n)"""
        trajectory = [x0]
        x = x0
        for _ in range(n_iter):
            x = r * x * (1 - x)
            trajectory.append(x)
        return trajectory
    
    @staticmethod
    def double_pendulum(theta1: float, theta2: float, 
                        omega1: float, omega2: float,
                        l1: float = 1.0, l2: float = 1.0,
                        m1: float = 1.0, m2: float = 1.0,
                        dt: float = 0.01, steps: int = 200) -> List[Tuple[float, float]]:
        """Double pendulum trajectory (chaotic for large angles)"""
        g = 9.81
        trajectory = [(theta1, theta2)]
        
        for _ in range(steps):
            delta = theta2 - theta1
            den1 = (m1 + m2) * l1 - m2 * l1 * math.cos(delta)**2
            den2 = (l2 / l1) * den1
            
            a1 = (m2 * l1 * omega1**2 * math.sin(delta) * math.cos(delta) +
                  m2 * g * math.sin(theta2) * math.cos(delta) +
                  m2 * l2 * omega2**2 * math.sin(delta) -
                  (m1 + m2) * g * math.sin(theta1)) / max(den1, 1e-10)
            
            a2 = (-m2 * l2 * omega2**2 * math.sin(delta) * math.cos(delta) +
                  (m1 + m2) * (g * math.sin(theta1) * math.cos(delta) -
                   l1 * omega1**2 * math.sin(delta) -
                   g * math.sin(theta2))) / max(den2, 1e-10)
            
            omega1 += a1 * dt
            omega2 += a2 * dt
            theta1 += omega1 * dt
            theta2 += omega2 * dt
            
            trajectory.append((theta1, theta2))
        
        return trajectory
    
    @staticmethod
    def lyapunov_estimate(trajectory_a: List[float], trajectory_b: List[float]) -> float:
        """Estimate largest Lyapunov exponent from divergence"""
        n = min(len(trajectory_a), len(trajectory_b))
        if n < 2:
            return 0
        
        d0 = abs(trajectory_a[0] - trajectory_b[0]) + 1e-15
        separations = []
        
        for i in range(n):
            d = abs(trajectory_a[i] - trajectory_b[i])
            separations.append(math.log(d / d0))
        
        # Linear fit of log(separation) vs time
        if len(separations) > 10:
            t = np.arange(len(separations))
            slope = np.polyfit(t[10:], separations[10:], 1)[0]
            return slope
        
        return 0
    
    @staticmethod
    def sensitivity_analysis(x0: float, r: float, epsilon: float = 0.0001) -> dict:
        """How much does the outcome change with a tiny perturbation?"""
        traj_a = ChaosAnalyzer.logistic_map(x0, r)
        traj_b = ChaosAnalyzer.logistic_map(x0 + epsilon, r)
        
        # Divergence after N steps
        final_diff = abs(traj_a[-1] - traj_b[-1])
        lyap = ChaosAnalyzer.lyapunov_estimate(traj_a, traj_b)
        
        chaotic = lyap > 0.01
        
        return {
            "initial_offset": epsilon,
            "final_divergence": round(final_diff, 4),
            "lyapunov_estimate": round(lyap, 4),
            "is_chaotic": chaotic,
            "prediction_horizon": round(math.log(1.0 / epsilon) / max(lyap, 1e-10), 1) if chaotic else "illimité",
            "verdict": "🦋 CHAOTIQUE — une pichenette change tout" if chaotic else "✅ PRÉDICTIBLE — trajectoire stable"
        }


# ============================================================
# 6. OPTIMIZATION — Least action, brachistochrone
# ============================================================

class OptimizationEngine:
    """Optimal paths and least action principles"""
    
    @staticmethod
    def brachistochrone_time(start: Tuple[float, float], end: Tuple[float, float], 
                             gravity: float = 9.81) -> float:
        """
        Time to slide down a cycloid (fastest path).
        For a cycloid, T = π√(R/g) where R is the generating circle radius.
        Simplified approximation for straight vs circular vs cycloid.
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        L = math.sqrt(dx**2 + dy**2)
        
        # Straight line time (on frictionless surface)
        theta = math.atan2(abs(dy), abs(dx))
        t_straight = math.sqrt(2 * L / (gravity * math.sin(max(theta, 0.01))))
        
        # Cycloid is about 1.2-1.3× faster than straight line
        t_cycloid = t_straight * 0.8
        
        return t_cycloid
    
    @staticmethod
    def trajectory_optimization(launch_angle_deg: float, initial_speed: float,
                                target_distance: float, gravity: float = 9.81) -> dict:
        """Find the optimal launch angle to reach a target"""
        theta = math.radians(launch_angle_deg)
        v0 = initial_speed
        g = gravity
        
        # Range: R = v₀² sin(2θ) / g
        R_max = v0**2 / g  # At 45°
        R = v0**2 * math.sin(2 * theta) / g
        
        # Time of flight
        t_flight = 2 * v0 * math.sin(theta) / g
        
        # Max height
        h_max = (v0 * math.sin(theta))**2 / (2 * g)
        
        # Optimal angle for given distance
        if target_distance < R_max:
            opt_angle_rad = 0.5 * math.asin(target_distance * g / v0**2)
            opt_angle_deg = math.degrees(opt_angle_rad)
            requires_optimal = abs(launch_angle_deg - opt_angle_deg) > 5
        else:
            opt_angle_deg = 45
            R = R_max
            requires_optimal = True
        
        return {
            "range_m": round(R, 1),
            "max_range_m": round(R_max, 1),
            "height_max_m": round(h_max, 1),
            "flight_time_s": round(t_flight, 2),
            "reaches_target": target_distance <= R,
            "optimal_angle_deg": round(opt_angle_deg, 1),
            "needs_angle_adjustment": requires_optimal
        }
    
    @staticmethod
    def friction_optimal_speed(curve_radius: float, friction_coeff: float,
                                gravity: float = 9.81) -> float:
        """Maximum speed in a turn without slipping: v_max = √(μgr)"""
        return math.sqrt(friction_coeff * gravity * curve_radius)


# ============================================================
# 7. THERMODYNAMICS — Cycles, efficiency, entropy
# ============================================================

class ThermoCycle(Enum):
    CARNOT = "carnot"
    OTTO = "otto"
    DIESEL = "diesel"
    RANKINE = "rankine"


class ThermodynamicsEngine:
    """Heat engines and thermodynamic cycles"""
    
    @staticmethod
    def carnot_efficiency(T_hot: float, T_cold: float) -> float:
        """η = 1 - T_cold/T_hot (Kelvin) — theoretical maximum"""
        if T_hot <= 0:
            return 0
        return 1 - T_cold / T_hot
    
    @staticmethod
    def otto_efficiency(compression_ratio: float, gamma: float = 1.4) -> float:
        """η = 1 - 1/r^(γ-1)"""
        if compression_ratio <= 1:
            return 0
        return 1 - 1 / (compression_ratio ** (gamma - 1))
    
    @staticmethod
    def diesel_efficiency(compression_ratio: float, cutoff_ratio: float, 
                          gamma: float = 1.4) -> float:
        """η = 1 - (1/r^(γ-1)) × (ρ^γ-1)/(γ(ρ-1))"""
        if compression_ratio <= 1:
            return 0
        term1 = 1 / (compression_ratio ** (gamma - 1))
        term2 = (cutoff_ratio**gamma - 1) / (gamma * (cutoff_ratio - 1)) if cutoff_ratio > 1 else 1
        return 1 - term1 * term2
    
    @staticmethod
    def heat_required(mass: float, specific_heat: float, delta_T: float,
                      latent_heat: float = 0, phase_change: bool = False) -> float:
        """Q = mcΔT (+ mL if phase change)"""
        Q = mass * specific_heat * abs(delta_T)
        if phase_change:
            Q += mass * latent_heat
        return Q
    
    @staticmethod
    def entropy_change(Q: float, T: float) -> float:
        """ΔS = Q/T (reversible)"""
        if T <= 0:
            return 0
        return Q / T
    
    @staticmethod
    def analyze_engine(cycle: ThermoCycle, params: dict) -> dict:
        if cycle == ThermoCycle.CARNOT:
            eta = ThermodynamicsEngine.carnot_efficiency(
                params.get("T_hot", 800) + 273,
                params.get("T_cold", 300) + 273
            )
        elif cycle == ThermoCycle.OTTO:
            eta = ThermodynamicsEngine.otto_efficiency(params.get("r", 10))
        elif cycle == ThermoCycle.DIESEL:
            eta = ThermodynamicsEngine.diesel_efficiency(
                params.get("r", 18), params.get("cutoff", 2.5)
            )
        else:
            eta = 0.3
        
        return {
            "cycle": cycle.value,
            "efficiency": round(eta * 100, 1),
            "max_theoretical_W": round(eta * params.get("Q_in", 100000)),
            "waste_heat_W": round((1 - eta) * params.get("Q_in", 100000)),
            "comparison": f"{eta*100:.1f}% — {'excellent' if eta > 0.5 else 'bon' if eta > 0.35 else 'standard' if eta > 0.2 else 'faible'}"
        }


# ============================================================
# 8. STICK-SLIP FRICTION — Alternating adhesion/sliding
# ============================================================

class StickSlipAnalyzer:
    """Analyze stick-slip friction: creaking doors, violin strings, earthquakes"""
    
    @staticmethod
    def stick_slip_cycle(static_friction: float, dynamic_friction: float,
                         spring_constant: float, pull_speed: float,
                         mass: float = 1.0, n_cycles: int = 5) -> List[dict]:
        """
        Model stick-slip: mass pulled by spring at constant speed.
        Sticks until F_spring = μ_s·N, then slips until F_spring < μ_k·N.
        """
        events = []
        x_spring = 0.0  # Spring extension
        x_mass = 0.0    # Mass position
        sliding = False
        
        N = mass * 9.81  # Normal force
        F_stick_max = static_friction * N
        F_slip = dynamic_friction * N
        
        while len(events) < n_cycles * 2:
            dt = 0.001
            
            if not sliding:
                # Stick phase: spring extends
                x_spring += pull_speed * dt
                F_spring = spring_constant * (x_spring - x_mass)
                
                if F_spring >= F_stick_max:
                    sliding = True
                    events.append({
                        "phase": "stick_end",
                        "time": round(len(events) * 0.05, 3),
                        "spring_force": round(F_spring, 1),
                        "displacement": round(x_spring - x_mass, 4)
                    })
            else:
                # Slip phase: mass accelerates
                F_spring = spring_constant * (x_spring - x_mass)
                F_net = F_spring - F_slip
                
                if F_net <= 0 and sliding:
                    sliding = False
                    x_mass = x_spring  # Snap to spring position
                    events.append({
                        "phase": "slip_end",
                        "time": round(len(events) * 0.05, 3),
                        "spring_force": round(F_spring, 1),
                        "total_slip": round(x_spring - x_mass, 4)
                    })
                else:
                    a = F_net / mass
                    x_mass += 0  # Simplified
            
            x_spring += pull_speed * dt
        
        return events
    
    @staticmethod
    def critical_pull_speed(static_friction: float, dynamic_friction: float,
                            spring_constant: float, mass: float = 1.0) -> float:
        """Below this speed, stick-slip occurs. Above it, smooth sliding."""
        delta_mu = static_friction - dynamic_friction
        N = mass * 9.81
        return (delta_mu * N) / math.sqrt(spring_constant * mass)
    
    @staticmethod
    def analyze(static_friction: float, dynamic_friction: float,
                spring_constant: float, pull_speed: float,
                mass: float = 1.0) -> dict:
        v_crit = StickSlipAnalyzer.critical_pull_speed(
            static_friction, dynamic_friction, spring_constant, mass
        )
        
        if pull_speed < v_crit * 0.1:
            behavior = "stick-slip intense — mouvements saccadés violents"
        elif pull_speed < v_crit:
            behavior = "stick-slip modéré — alternance collé/glissé audible"
        elif pull_speed < v_crit * 2:
            behavior = "transition — quasi-continu avec micro-saccades"
        else:
            behavior = "glissement continu — mouvement fluide"
        
        mu_ratio = static_friction / max(dynamic_friction, 1e-10)
        
        return {
            "critical_speed_ms": round(v_crit, 4),
            "pull_speed_ms": round(pull_speed, 4),
            "mu_ratio": round(mu_ratio, 1),
            "behavior": behavior,
            "will_squeak": pull_speed < v_crit,
            "example": "porte qui grince" if pull_speed < v_crit and mu_ratio > 1.5 else 
                       "violon (archet)" if pull_speed < v_crit else
                       "glissement fluide"
        }


# ============================================================
# 9. UNIFIED DEMO — All extensions
# ============================================================

def demo_all():
    sep = "=" * 60
    
    # 1. Elasticity
    print(f"{sep}\n  1. ÉLASTICITÉ — Poutre sous charge\n{sep}")
    beam = BeamAnalysis("bois_chene", length=3.0, width=0.1, height=0.2, load=2000, support_type="simply_supported")
    result = beam.analyze()
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # 2. Resonance
    print(f"\n{sep}\n  2. RÉSONANCE — Pont de Tacoma\n{sep}")
    bridge = Oscillator(mass=50000, stiffness=1e6, damping=500, initial_displacement=0.5)
    result = bridge.analyze(driving_freq=bridge.natural_frequency_hz, driving_force=10000)
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # 3. Hydraulics
    print(f"\n{sep}\n  3. HYDRAULIQUE — Vérin\n{sep}")
    jack = HydraulicSystem(piston1_area=0.0001, piston2_area=0.01, input_force=200)
    result = jack.analyze()
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # 4. Waves
    print(f"\n{sep}\n  4. ONDES — Son dans l'air\n{sep}")
    w1 = Wave(WaveType.SOUND, frequency=440, amplitude=0.001, medium="air")
    result = w1.analyze()
    result["reflection_eau"] = w1.reflect("eau")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # 5. Chaos
    print(f"\n{sep}\n  5. CHAOS — Effet papillon\n{sep}")
    result = ChaosAnalyzer.sensitivity_analysis(0.5, 3.9, 0.0001)
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # 6. Optimization
    print(f"\n{sep}\n  6. OPTIMISATION — Tir balistique\n{sep}")
    result = OptimizationEngine.trajectory_optimization(30, 50, 200)
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # 7. Thermodynamics
    print(f"\n{sep}\n  7. THERMODYNAMIQUE — Moteur\n{sep}")
    result = ThermodynamicsEngine.analyze_engine(ThermoCycle.OTTO, {"r": 10, "Q_in": 100000})
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    # 8. Stick-Slip
    print(f"\n{sep}\n  8. STICK-SLIP — Porte qui grince\n{sep}")
    result = StickSlipAnalyzer.analyze(0.8, 0.4, 500, 0.01, 2.0)
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    demo_all()
