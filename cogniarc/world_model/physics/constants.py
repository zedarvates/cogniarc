"""Shared constants for World Model Tool."""
# Numerical
EPSILON = 1e-12

# Physical
G = 6.67430e-11   # Gravitational constant
G_EARTH = 9.81    # Standard gravity (m/s²)
G_MOON = 1.62
G_MARS = 3.72

# Materials (common)
MATERIAL_STRENGTH = {
    "acier":  {"E": 210e9, "sigma_y": 400e6, "density": 7800},
    "beton":  {"E": 30e9,  "sigma_y": 25e6,  "density": 2400},
    "bois":   {"E": 12e9,  "sigma_y": 50e6,  "density": 700},
    "alu":    {"E": 70e9,  "sigma_y": 200e6, "density": 2700},
}

# Liquids (common)
LIQUID_PROPS = {
    "water":   {"density": 1000, "viscosity": 0.001},
    "oil":     {"density": 900,  "viscosity": 0.8},
    "honey":   {"density": 1400, "viscosity": 10.0},
    "mercury": {"density": 13500,"viscosity": 0.0015},
}
