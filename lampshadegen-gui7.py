#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Procedural Lampshade Generator - Refactored
===========================================

This script procedurally generates 3D printable lampshades based on a wide
variety of artistic styles, mathematical formulas, and natural patterns.

This refactored version focuses on:
- **Clarity & Readability:** Extensive comments and a more organized structure.
- **Extensibility:** A modular design that makes adding new styles easier.
- **Robustness:** Better parameter management and error handling.
- **Performance:** Optimizations for complex calculations.

Core Concepts:
- **Styles:** The primary shape is defined by a `LampshadeStyle`.
- **Modifiers:** Optional transformations that add complexity (e.g., ribs, swirls).
- **Parameters:** A central `LampshadeParameters` dataclass controls everything.
- **E27 Socket:** A standard E27 socket is guaranteed for usability.
- **Manifold Output:** Generates 3D printable STL files with sealed perforations.

Author: AI Assistant (Refactored by Gemini)
License: MIT
"""

# --- Standard Library Imports ---
import argparse
import math
import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename
from tkinter import messagebox
import tempfile


# --- Third-Party Imports ---
import numpy as np
from stl import mesh

# Attempt to import optional libraries for the previewer
try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt # Import pyplot for colormaps
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not found. Preview functionality will be disabled.")
try:
    from scipy.signal import convolve2d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not found. Some styles (e.g., Reaction-Diffusion) will be disabled.")


# --- Global Configuration ---
OUTPUT_DIR = "generated_lampshades"
# Default perforation chance, can be overridden by command-line args.
DEFAULT_PERFORATION_CHANCE = 0.13
# Default minimum height from the ground to the bottom of the lampshade
DEFAULT_MIN_FLOOR_HEIGHT = 5.0
# Default wall thickness for the sides of the lampshade
DEFAULT_WALL_THICKNESS = 2.0


# --- Default Generation Boundary Constraints (in mm) ---
# These values are used to bound the initial random generation of the
# lampshade's core dimensions (height, radii), not for post-scaling.
DEFAULT_MIN_X_SIZE = 50
DEFAULT_MIN_Y_SIZE = 50
DEFAULT_MIN_Z_SIZE = 100
DEFAULT_MAX_X_SIZE = 220
DEFAULT_MAX_Y_SIZE = 220
DEFAULT_MAX_Z_SIZE = 250

# --- Enumerations for Controlled Selection ---

class Complexity(Enum):
    """Defines the polygon complexity level for the generated mesh."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    RANDOM = "random"

class LampshadeStyle(Enum):
    """
    Defines the foundational artistic style for the lampshade.
    Each style corresponds to a specific generation algorithm. I've curated this
    list from your originals, removing some that were conceptually overlapping
    or less likely to produce good results, and keeping the strong ones.
    """
    # --- Core & Geometric ---
    CLASSIC_CURVE = "classic_curve"
    HEXAGONAL_MESH = "hexagonal_mesh"
    WAVE_DEFORMATION = "wave_deformation"
    PLEATED_FABRIC = "pleated_fabric"
    FOLDED_PAPER = "folded_paper"
    BARREL_CACTUS_RIBS = "barrel_cactus_ribs"

    # --- Organic & Natural ---
    ORGANIC_VORONOI = "organic_voronoi"
    WOVEN_BASKET = "woven_basket"
    MUSHROOM_GILL = "mushroom_gill"
    PINECONE_SCALES = "pinecone_scales"
    LEAF_VENATION = "leaf_venation"
    NAUTILUS_SHELL = "nautilus_shell"
    WORLEY_NOISE_CELLS = "worley_noise_cells"

    # --- Algorithmic & Fractal ---
    FRACTAL_SURFACE = "fractal_surface"
    GYROID_SURFACE = "gyroid_surface"
    ATTRACTOR_FLOW = "attractor_flow" # Clifford/De Jong attractors
    PHYLLOTAXIS_SURFACE = "phyllotaxis_surface" # Sunflower spiral in 3D
    REACTION_DIFFUSION = "reaction_diffusion" # Turing patterns
    SUPERFORMULA_2D = "superformula_2d"
    FIBONACCI_SPIRAL = "fibonacci_spiral"
    HARMONOGRAPH_SPIRAL = "harmonograph_spiral"

    # --- Experimental & Complex ---
    # These are ambitious but can produce stunning results.
    CRYSTAL_GEODE = "crystal_geode"
    CAVE_STALACTITES = "cave_stalactites"
    DRIPPING_LIQUID = "dripping_liquid"

    # --- Advanced Mathematical & Physics-Inspired ---
    # New styles based on crystallography, minimal surfaces, and wave physics.
    SCHWARZ_PRIMITIVE_TPMS = "schwarz_primitive_tpms"  # Triply Periodic Minimal Surface (P-surface)
    QUASICRYSTAL_ICOSAHEDRAL = "quasicrystal_icosahedral"  # Aperiodic 5-fold symmetry patterns
    LISSAJOUS_CYLINDRICAL_KNOT = "lissajous_cylindrical_knot"  # 3D Lissajous curves on cylinder
    CHLADNI_CYLINDRICAL_RESONANCE = "chladni_cylindrical_resonance"  # Vibrating cylinder nodal patterns


class PerforationPattern(Enum):
    """
    Defines how perforations (holes) are distributed across the surface.
    This provides artistic control over the light-filtering patterns.
    """
    RANDOM = "random"
    VERTICAL_STRIPE = "vertical_stripe"
    SPIRAL_STRIPE = "spiral_stripe"
    CHECKERBOARD = "checkerboard"
    HEX_GRID = "hex_grid"
    POLKA_DOT = "polka_dot"
    WAVE_VERTICAL = "wave_vertical"
    PERLIN_NOISE = "perlin_noise"
    RAIN_DROPS = "rain_drops"
    ZEBRA_STRIPES = "zebra_stripes"
    STARFIELD = "starfield"

class FittingType(Enum):
    """Defines standard light bulb fitting sizes."""
    NONE = 0.0 # For solid objects like vases
    E27 = 27.0 # Standard screw-in
    E14 = 14.0 # Small screw-in
    B22 = 22.0 # Bayonet cap

# --- Global Feature Toggles ---
# Set these to False to disable a feature from being randomly selected during generation.
MODIFIER_TOGGLES = {
    "enable_swirl_effect": True,
    "enable_taper_profile": True,
    "enable_mid_height_modulation": True,
    "enable_sinusoidal_shear": True,
    "enable_squash_stretch": True,
    "enable_vertical_drift": True,
    "enable_asymmetric_base": True,
    "enable_twist_effect": True,
}

# --- New Feature Configurations ---
DEFAULT_FITTING_TYPE = FittingType.E27
DEFAULT_PERFORATION_SIZE = 1 # Size in grid units (1x1, 2x2, etc.)
ENABLE_STYLE_BLENDING = True
STYLE_BLEND_FACTOR = 0.5 # 0.0 = 100% Style 1, 1.0 = 100% Style 2


@dataclass
class LampshadeParameters:
    """
    A single, structured configuration object for generating a lampshade.
    Using a dataclass provides type safety, default values, and a clean
    way to manage the large number of parameters involved.
    """
    # --- Core Dimensions (in millimeters) ---
    height: float = field(default_factory=lambda: random.uniform(DEFAULT_MIN_Z_SIZE, DEFAULT_MAX_Z_SIZE))
    top_radius: float = field(default_factory=lambda: random.uniform(DEFAULT_MIN_X_SIZE / 2, (DEFAULT_MAX_X_SIZE / 2) * 0.9))
    bottom_radius: float = field(default_factory=lambda: random.uniform(DEFAULT_MIN_Y_SIZE / 2, DEFAULT_MAX_Y_SIZE / 2))
    wall_thickness: float = field(default_factory=lambda: DEFAULT_WALL_THICKNESS)
    min_floor_height: float = DEFAULT_MIN_FLOOR_HEIGHT

    # --- E27 Socket (Standard Fitting) ---
    # Fixed value for a standard E27 light bulb fitting. Non-optional.
    socket_hole_radius: float = 27.0 / 2.0
    fitting_type: FittingType = DEFAULT_FITTING_TYPE

    # --- Mesh Resolution ---
    # Higher values create a smoother model but increase computation time.
    # These are set by the 'complexity' argument later.
    radial_segments: int = 128
    height_segments: int = 96

    # --- Style Selection & Intensity ---
    style: LampshadeStyle = field(default_factory=lambda: random.choice(list(LampshadeStyle)))
    style2: LampshadeStyle = field(default_factory=lambda: random.choice(list(LampshadeStyle)))
    style1_intensity: float = field(default_factory=lambda: random.uniform(0.7, 1.5))
    style2_intensity: float = field(default_factory=lambda: random.uniform(0.7, 1.5))
    blend_factor: float = STYLE_BLEND_FACTOR
    enable_blending: bool = ENABLE_STYLE_BLENDING
    perforation_pattern: PerforationPattern = PerforationPattern.RANDOM
    perforation_size: int = DEFAULT_PERFORATION_SIZE
    perforation_chance: float = DEFAULT_PERFORATION_CHANCE

    # --- Style-Specific Parameters ---
    # These parameters only affect the model if their corresponding style is selected.
    # Using a factory ensures a new random value is generated for each instance.
    curve_factor: float = field(default_factory=lambda: random.uniform(-1.0, 1.0))
    wave_amplitude: float = field(default_factory=lambda: random.uniform(2, 10))
    wave_frequency: int = field(default_factory=lambda: random.randint(2, 6))
    voronoi_cells: int = field(default_factory=lambda: random.randint(20, 60))
    voronoi_depth: float = field(default_factory=lambda: random.uniform(2, 8))
    fractal_octaves: int = field(default_factory=lambda: random.randint(2, 5))
    fractal_amplitude: float = field(default_factory=lambda: random.uniform(1, 6))
    hex_scale: float = field(default_factory=lambda: random.uniform(15, 30))
    hex_depth: float = field(default_factory=lambda: random.uniform(1, 4))
    pleat_frequency: int = field(default_factory=lambda: random.randint(12, 36))
    pleat_amplitude: float = field(default_factory=lambda: random.uniform(2, 8))
    weave_frequency_v: int = field(default_factory=lambda: random.randint(8, 20))
    weave_frequency_h: int = field(default_factory=lambda: random.randint(6, 16))
    weave_amplitude: float = field(default_factory=lambda: random.uniform(2, 6))
    drip_count: int = field(default_factory=lambda: random.randint(5, 15))
    drip_length_factor: float = field(default_factory=lambda: random.uniform(0.1, 0.4))
    drip_amplitude: float = field(default_factory=lambda: random.uniform(3, 10))
    attractor_a: float = field(default_factory=lambda: random.uniform(-2.0, 2.0))
    attractor_b: float = field(default_factory=lambda: random.uniform(-2.0, 2.0))
    attractor_c: float = field(default_factory=lambda: random.uniform(-2.0, 2.0))
    attractor_d: float = field(default_factory=lambda: random.uniform(-2.0, 2.0))
    attractor_influence: float = field(default_factory=lambda: random.uniform(5, 15))
    attractor_point_count: int = field(default_factory=lambda: random.randint(2000, 5000))
    superformula_m: float = field(default_factory=lambda: random.uniform(2, 12))
    superformula_n1: float = field(default_factory=lambda: random.uniform(0.5, 5))
    superformula_n2: float = field(default_factory=lambda: random.uniform(0.5, 5))
    superformula_n3: float = field(default_factory=lambda: random.uniform(1, 10))
    phylo_point_count: int = field(default_factory=lambda: random.randint(200, 800))
    phylo_divergence_angle: float = field(default_factory=lambda: random.uniform(137.4, 137.6)) # Golden angle
    phylo_point_radius: float = field(default_factory=lambda: random.uniform(4, 10))
    phylo_influence: float = field(default_factory=lambda: random.uniform(-10, 10))
    fold_frequency: int = field(default_factory=lambda: random.randint(4, 12))
    fold_amplitude: float = field(default_factory=lambda: random.uniform(5, 15))
    gill_frequency: int = field(default_factory=lambda: random.randint(20, 60))
    gill_depth: float = field(default_factory=lambda: random.uniform(4, 12))
    pinecone_scale_rows: int = field(default_factory=lambda: random.randint(5, 12))
    pinecone_scale_cols: int = field(default_factory=lambda: random.randint(8, 16))
    pinecone_scale_curl: float = field(default_factory=lambda: random.uniform(1, 5))
    geode_opening_z_norm: float = field(default_factory=lambda: random.uniform(0.3, 0.7))
    geode_opening_size: float = field(default_factory=lambda: random.uniform(0.2, 0.6))
    geode_crystal_count: int = field(default_factory=lambda: random.randint(10, 30))
    geode_crystal_size: float = field(default_factory=lambda: random.uniform(3, 10))
    geode_inner_depth: float = field(default_factory=lambda: random.uniform(10, 25))
    vein_depth: float = field(default_factory=lambda: random.uniform(1, 4))
    vein_primary_freq: int = field(default_factory=lambda: random.randint(3, 7))
    vein_secondary_freq: int = field(default_factory=lambda: random.randint(8, 20))
    nautilus_growth_factor: float = field(default_factory=lambda: random.uniform(0.1, 0.3))
    nautilus_chamber_count: int = field(default_factory=lambda: random.randint(3, 8))
    stalactite_count: int = field(default_factory=lambda: random.randint(10, 40))
    stalactite_length_max: float = field(default_factory=lambda: random.uniform(10, 50))
    gyroid_frequency: float = field(default_factory=lambda: random.uniform(5, 20))
    gyroid_amplitude: float = field(default_factory=lambda: random.uniform(2, 8))
    rd_feed_rate: float = field(default_factory=lambda: random.uniform(0.03, 0.07))
    rd_kill_rate: float = field(default_factory=lambda: random.uniform(0.05, 0.08))
    rd_iterations: int = field(default_factory=lambda: random.randint(20, 50))
    rd_amplitude: float = field(default_factory=lambda: random.uniform(-5, 5))
    cactus_rib_count: int = field(default_factory=lambda: random.randint(8, 24))
    cactus_rib_amp: float = field(default_factory=lambda: random.uniform(4, 10))
    worley_cell_count: int = field(default_factory=lambda: random.randint(20, 80))
    worley_amplitude: float = field(default_factory=lambda: random.uniform(2, 8))
    fibonacci_amplitude: float = field(default_factory=lambda: random.uniform(2, 8))
    fibonacci_density: int = field(default_factory=lambda: random.randint(3, 8))
    hg_freq1: float = field(default_factory=lambda: random.uniform(1, 4))
    hg_freq2: float = field(default_factory=lambda: random.uniform(1, 4))
    hg_phase1: float = field(default_factory=lambda: random.uniform(0, np.pi))
    hg_phase2: float = field(default_factory=lambda: random.uniform(0, np.pi))
    hg_decay: float = field(default_factory=lambda: random.uniform(0.01, 0.1))
    hg_amplitude: float = field(default_factory=lambda: random.uniform(5, 15))

    # --- Schwarz Primitive TPMS Parameters ---
    # Level set: cos(x) + cos(y) + cos(z) = c, creates cubic lattice of interconnected chambers
    tpms_p_frequency: float = field(default_factory=lambda: random.uniform(8, 25))  # Controls cell density
    tpms_p_threshold: float = field(default_factory=lambda: random.uniform(-0.3, 0.3))  # Level set constant c
    tpms_p_amplitude: float = field(default_factory=lambda: random.uniform(3, 10))  # Displacement strength

    # --- Quasicrystal Icosahedral Parameters ---
    # Based on 6D→3D projection (cut-and-project method), uses golden ratio φ=1.618
    quasi_wave_count: int = field(default_factory=lambda: random.randint(5, 7))  # Number of wave directions (5-7 for quasiperiodic)
    quasi_frequency: float = field(default_factory=lambda: random.uniform(10, 30))  # Base frequency
    quasi_amplitude: float = field(default_factory=lambda: random.uniform(2, 8))  # Displacement amplitude
    quasi_phase_seed: int = field(default_factory=lambda: random.randint(0, 1000))  # Seed for phase offsets

    # --- Lissajous Cylindrical Knot Parameters ---
    # r(t) = A·sin(at+φ₁), mapped to cylindrical coordinates for mathematical knots
    liss_freq_radial: int = field(default_factory=lambda: random.randint(2, 7))  # Radial frequency a
    liss_freq_vertical: int = field(default_factory=lambda: random.randint(3, 9))  # Vertical frequency b
    liss_freq_angular: int = field(default_factory=lambda: random.randint(1, 5))  # Angular frequency c
    liss_phase1: float = field(default_factory=lambda: random.uniform(0, np.pi))  # Phase offset φ₁
    liss_phase2: float = field(default_factory=lambda: random.uniform(0, np.pi))  # Phase offset φ₂
    liss_amplitude: float = field(default_factory=lambda: random.uniform(3, 10))  # Displacement amplitude
    liss_knot_width: float = field(default_factory=lambda: random.uniform(5, 15))  # Width of knot influence

    # --- Chladni Cylindrical Resonance Parameters ---
    # Based on ∇²ψ + k²ψ = 0 on cylindrical surfaces (Bessel functions)
    chladni_mode_m: int = field(default_factory=lambda: random.randint(2, 8))  # Circumferential mode number
    chladni_mode_n: int = field(default_factory=lambda: random.randint(1, 5))  # Axial mode number
    chladni_amplitude: float = field(default_factory=lambda: random.uniform(3, 12))  # Displacement amplitude
    chladni_superposition: int = field(default_factory=lambda: random.randint(1, 3))  # Number of superimposed modes


    # --- Perforation Pattern Parameters ---
    perf_pattern_freq: int = field(default_factory=lambda: random.randint(2, 8))
    perf_pattern_scale: float = field(default_factory=lambda: random.uniform(20, 60))
    perf_pattern_density: float = field(default_factory=lambda: random.uniform(0.1, 0.7))

    # --- Modifier Toggles & Parameters ---
    # The default factory now checks the global toggle before deciding to be True/False.
    enable_swirl_effect: bool = field(default_factory=lambda: random.choice([True, False]) if MODIFIER_TOGGLES.get("enable_swirl_effect") else False)
    swirl_strength: float = field(default_factory=lambda: random.uniform(np.pi / 4, np.pi))

    enable_taper_profile: bool = field(default_factory=lambda: random.choice([True, False]) if MODIFIER_TOGGLES.get("enable_taper_profile") else False)
    taper_factor: float = field(default_factory=lambda: random.uniform(-0.8, 0.8))
    
    enable_mid_height_modulation: bool = field(default_factory=lambda: random.choice([True, False]) if MODIFIER_TOGGLES.get("enable_mid_height_modulation") else False)
    mid_mod_amplitude: float = field(default_factory=lambda: random.uniform(-20, 20))

    enable_sinusoidal_shear: bool = field(default_factory=lambda: random.choice([True, False]) if MODIFIER_TOGGLES.get("enable_sinusoidal_shear") else False)
    shear_amplitude: float = field(default_factory=lambda: random.uniform(5, 20))
    shear_frequency: int = field(default_factory=lambda: random.randint(1, 5))
    
    enable_squash_stretch: bool = field(default_factory=lambda: random.choice([True, False]) if MODIFIER_TOGGLES.get("enable_squash_stretch") else False)
    squash_stretch_amplitude: float = field(default_factory=lambda: random.uniform(-20, 20))
    squash_stretch_frequency: int = field(default_factory=lambda: random.randint(1, 4))
    
    enable_vertical_drift: bool = field(default_factory=lambda: random.choice([True, False]) if MODIFIER_TOGGLES.get("enable_vertical_drift") else False)
    vertical_drift_amplitude: float = field(default_factory=lambda: random.uniform(2, 8))
    vertical_drift_frequency: int = field(default_factory=lambda: random.randint(2, 6))
    
    enable_asymmetric_base: bool = field(default_factory=lambda: random.choice([True, False]) if MODIFIER_TOGGLES.get("enable_asymmetric_base") else False)
    asymmetric_base_offset: float = field(default_factory=lambda: random.uniform(-np.pi / 8, np.pi / 8))

    enable_twist_effect: bool = field(default_factory=lambda: random.choice([True, False]) if MODIFIER_TOGGLES.get("enable_twist_effect") else False)
    twist_angle_deg: float = field(default_factory=lambda: random.uniform(30, 180))

    # --- Pre-calculated Data Structures ---
    # These are initialized here and populated by the `__post_init__` method.
    # This avoids recalculating complex data for every vertex.
    voronoi_seed_points: List[np.ndarray] = field(default_factory=list, init=False)
    attractor_points: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    phylo_points: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    geode_crystals: List[np.ndarray] = field(default_factory=list, init=False)
    stalactites: List[List[float]] = field(default_factory=list, init=False)
    rd_pattern: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    worley_points: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    modifier_chain: List[str] = field(default_factory=list, init=False)
    # Pre-calculated for new styles
    lissajous_points: np.ndarray = field(default_factory=lambda: np.array([]), init=False)
    quasi_phases: np.ndarray = field(default_factory=lambda: np.array([]), init=False)


    def __post_init__(self):
        """
        This method is automatically called after the dataclass is initialized.
        It's used for validation and to pre-generate data needed by certain styles.
        This is a powerful feature of dataclasses for setup logic.
        """
        # 1. Set socket radius based on fitting type
        self.socket_hole_radius = self.fitting_type.value / 2.0

        # 2. Validate Dimensions: Ensure the lampshade is physically possible.
        min_bottom_radius = self.socket_hole_radius + self.wall_thickness + 5.0 # 5mm buffer
        if self.fitting_type != FittingType.NONE and self.bottom_radius < min_bottom_radius:
            print(f"Warning: Bottom radius is too small. Adjusting from {self.bottom_radius:.1f} to {min_bottom_radius:.1f}mm.")
            self.bottom_radius = min_bottom_radius
            
        # 3. Handle Style Blending
        if self.enable_blending:
            # Ensure style2 is different from style1
            while self.style2 == self.style:
                self.style2 = random.choice(list(LampshadeStyle))
        else:
            self.style2 = self.style # If not blending, style2 is the same as style1

        # 5. Pre-generate data for computationally expensive styles.
        # This prevents recalculating the same complex data for every single vertex.
        # It's a key optimization.
        if self.style == LampshadeStyle.ORGANIC_VORONOI or self.style2 == LampshadeStyle.ORGANIC_VORONOI: self._generate_voronoi_seeds()
        if self.style == LampshadeStyle.ATTRACTOR_FLOW or self.style2 == LampshadeStyle.ATTRACTOR_FLOW: self._generate_attractor_points()
        if self.style == LampshadeStyle.PHYLLOTAXIS_SURFACE or self.style2 == LampshadeStyle.PHYLLOTAXIS_SURFACE: self._generate_phylo_points()
        if self.style == LampshadeStyle.CRYSTAL_GEODE or self.style2 == LampshadeStyle.CRYSTAL_GEODE: self._generate_geode_crystals()
        if self.style == LampshadeStyle.CAVE_STALACTITES or self.style2 == LampshadeStyle.CAVE_STALACTITES: self._generate_stalactites()
        if self.style == LampshadeStyle.REACTION_DIFFUSION or self.style2 == LampshadeStyle.REACTION_DIFFUSION: self._generate_rd_pattern()
        if self.style == LampshadeStyle.WORLEY_NOISE_CELLS or self.style2 == LampshadeStyle.WORLEY_NOISE_CELLS: self._generate_worley_points()
        # Pre-generation for new styles
        if self.style == LampshadeStyle.LISSAJOUS_CYLINDRICAL_KNOT or self.style2 == LampshadeStyle.LISSAJOUS_CYLINDRICAL_KNOT: self._generate_lissajous_points()
        if self.style == LampshadeStyle.QUASICRYSTAL_ICOSAHEDRAL or self.style2 == LampshadeStyle.QUASICRYSTAL_ICOSAHEDRAL: self._generate_quasi_phases()

    def build_modifier_chain(self):
        """Builds the modifier chain based on the current 'enable_' flags."""
        self.modifier_chain = [mod for mod, is_enabled in self.__dict__.items() if mod.startswith("enable_") and is_enabled]
        random.shuffle(self.modifier_chain)

    def get_config_string(self) -> str:
        """Returns a formatted string of the lampshade's configuration."""
        config_lines = ["--- Lampshade Generation Configuration ---"]
        if self.enable_blending:
            config_lines.append(f"  Style 1: {self.style.value} (Intensity: {self.style1_intensity:.2f})")
            config_lines.append(f"  Style 2: {self.style2.value} (Intensity: {self.style2_intensity:.2f})")
            config_lines.append(f"  Blend Factor: {self.blend_factor:.2f}")
        else:
            config_lines.append(f"  Style: {self.style.value}")
            config_lines.append(f"  Style Intensity: {self.style1_intensity:.2f}")
        config_lines.append(f"  Dimensions (H x TopR x BotR): {self.height:.1f} x {self.top_radius:.1f} x {self.bottom_radius:.1f} mm")
        config_lines.append(f"  Wall Thickness: {self.wall_thickness:.2f} mm")
        config_lines.append(f"  Resolution (Radial x Height): {self.radial_segments} x {self.height_segments}")
        config_lines.append("\n--- Modifier Chain (Randomized Order) ---")
        if self.modifier_chain:
            for mod in self.modifier_chain:
                config_lines.append(f"  - {mod[7:].replace('_', ' ').title()}")
        else:
            config_lines.append("  None")
        config_lines.append("-" * 40)
        return "\n".join(config_lines)

    # --- Data Generation Methods for __post_init__ ---
    def _generate_voronoi_seeds(self):
        """Generates random 3D points on the surface of the base shape."""
        for _ in range(self.voronoi_cells):
            angle = random.uniform(0, 2 * math.pi)
            z_norm = random.uniform(0, 1)
            radius = self.top_radius + (self.bottom_radius - self.top_radius) * z_norm
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = z_norm * self.height
            self.voronoi_seed_points.append(np.array([x, y, z]))

    def _generate_attractor_points(self):
        """Generates a point cloud based on a Clifford attractor algorithm."""
        points = []
        x, y = 0.1, 0.1
        a, b, c, d = self.attractor_a, self.attractor_b, self.attractor_c, self.attractor_d
        for _ in range(self.attractor_point_count):
            xn = np.sin(a * y) + c * np.cos(a * x)
            yn = np.sin(b * x) + d * np.cos(b * y)
            x, y = xn, yn
            points.append([x, y])
        points_np = np.array(points)
        points_np -= points_np.min(axis=0)
        points_np /= points_np.max(axis=0)
        points_np = (points_np * 2) - 1
        max_radius = max(self.top_radius, self.bottom_radius)
        points_np[:, 0] *= max_radius
        points_np[:, 1] *= self.height
        self.attractor_points = np.insert(points_np, 2, points_np[:, 1], axis=1) # Use Y for Z
        self.attractor_points[:, 1] = 0 # Flatten to XZ plane

    def _generate_phylo_points(self):
        """Generates points based on Phyllotaxis (sunflower spiral) algorithm."""
        points = []
        angle_rad = np.radians(self.phylo_divergence_angle)
        for n in range(self.phylo_point_count):
            z_norm = n / self.phylo_point_count
            radius = self.top_radius + (self.bottom_radius - self.top_radius) * z_norm
            theta = n * angle_rad
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = z_norm * self.height
            points.append([x, y, z])
        self.phylo_points = np.array(points)

    def _generate_geode_crystals(self):
        """Generates seed points for crystals inside the geode opening."""
        for _ in range(self.geode_crystal_count):
            angle = random.uniform(0, 2 * math.pi)
            z_norm = self.geode_opening_z_norm + random.uniform(-self.geode_opening_size, self.geode_opening_size) * 0.5
            z_norm = np.clip(z_norm, 0.05, 0.95)
            radius = self.top_radius + (self.bottom_radius - self.top_radius) * z_norm - self.wall_thickness - self.geode_inner_depth
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = z_norm * self.height
            self.geode_crystals.append(np.array([x, y, z]))

    def _generate_stalactites(self):
        """Generates starting angles, lengths, and phases for stalactites."""
        for _ in range(self.stalactite_count):
            self.stalactites.append([
                random.uniform(0, 2 * math.pi), # angle
                random.uniform(0.1, self.stalactite_length_max), # length
                random.uniform(0, 2 * np.pi) # phase for waviness
            ])

    def _generate_rd_pattern(self):
        """Pre-calculates a Reaction-Diffusion (Turing) pattern on a 2D grid."""
        if not SCIPY_AVAILABLE:
            print("Skipping Reaction-Diffusion: SciPy not installed.")
            return
        print("Pre-calculating Reaction-Diffusion pattern...")
        size = 64
        A = np.ones((size, size))
        B = np.zeros((size, size))
        r = 10
        A[size//2-r:size//2+r, size//2-r:size//2+r] = 0.5
        B[size//2-r:size//2+r, size//2-r:size//2+r] = 0.25
        lap_kernel = np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]])

        for _ in range(self.rd_iterations):
            lap_A = convolve2d(A, lap_kernel, mode='same', boundary='wrap')
            lap_B = convolve2d(B, lap_kernel, mode='same', boundary='wrap')
            reaction = A * B**2
            A += (1.0 * lap_A - reaction + self.rd_feed_rate * (1 - A))
            B += (0.5 * lap_B + reaction - (self.rd_kill_rate + self.rd_feed_rate) * B)
        self.rd_pattern = B

    def _generate_worley_points(self):
        """Generates random 3D points for Worley noise calculation."""
        points = []
        for _ in range(self.worley_cell_count):
            angle = random.uniform(0, 2 * math.pi)
            z_norm = random.uniform(0, 1)
            radius = self.top_radius + (self.bottom_radius - self.top_radius) * z_norm
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = z_norm * self.height
            points.append([x, y, z])
        self.worley_points = np.array(points)

    def _generate_lissajous_points(self):
        """
        Generates 3D Lissajous curve points mapped to cylindrical coordinates.
        The curve wraps around the lampshade creating mathematical knot patterns.
        """
        points = []
        num_points = 500  # Dense sampling for smooth curves
        for i in range(num_points):
            t = 2 * np.pi * i / num_points * self.liss_freq_angular
            # Lissajous parametric equations mapped to cylinder
            r_offset = self.liss_amplitude * np.sin(self.liss_freq_radial * t + self.liss_phase1)
            z_norm = 0.5 + 0.45 * np.sin(self.liss_freq_vertical * t + self.liss_phase2)
            theta = t
            # Calculate base radius at this height
            base_r = self.top_radius + (self.bottom_radius - self.top_radius) * z_norm
            r = base_r + r_offset
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = z_norm * self.height
            points.append([x, y, z])
        self.lissajous_points = np.array(points)

    def _generate_quasi_phases(self):
        """
        Generates phase offsets for quasicrystal wave interference.
        Uses golden ratio relationships for aperiodic patterns.
        """
        np.random.seed(self.quasi_phase_seed)
        # Generate quasi_wave_count directions with golden-ratio-based angles
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio φ ≈ 1.618
        phases = []
        for i in range(self.quasi_wave_count):
            # Use golden angle increments for aperiodic spacing
            angle = 2 * np.pi * i / phi
            phase = np.random.uniform(0, 2 * np.pi)
            phases.append([angle, phase])
        self.quasi_phases = np.array(phases)


class LampshadeGenerator:
    """
    The main class responsible for taking parameters and generating the 3D mesh.
    This class handles the entire mesh creation process from vertex generation
    to face creation.
    """
    def __init__(self, params: LampshadeParameters, seed: int):
        """
        Initializes the generator with a specific set of parameters.
        Args:
            params: A LampshadeParameters object containing the full configuration.
            seed: The random seed used for this generation, for reproducibility.
        """
        self.params = params
        self.seed = seed
        # This will store the list of all [x, y, z] vertex coordinates as a NumPy array
        # for performance.
        self.vertices = np.array([])
        self.perforation_grid = np.array([])

    def generate(self) -> mesh.Mesh:
        """
        The main public method to generate the lampshade mesh.
        Returns:
            A `stl.mesh.Mesh` object representing the final lampshade.
        """
        print(self.params.get_config_string())
        self._generate_vertex_grid()
        faces = self._create_faces()

        # Create the mesh object from the vertices and faces.
        stl_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = self.vertices[f[j]]

        # **RELIABLE FLOOR FIX:** Translate the entire mesh up after generation.
        # This ensures the bottom is perfectly flat at Z=0 before being moved.
        stl_mesh.z += self.params.min_floor_height

        stl_mesh.update_normals()
        print(f"Generated lampshade with {len(self.vertices)} vertices and {len(faces)} faces.")
        return stl_mesh

    def _get_base_radius_at_z(self, z_norm: float) -> float:
        """
        Calculates the base radius of the lampshade at a given normalized height.
        This defines the fundamental profile curve of the lampshade before any
        artistic styles or modifiers are applied.
        Args:
            z_norm: The normalized height (0.0 at the bottom, 1.0 at the top).
        Returns:
            The calculated radius in millimeters.
        """
        p = self.params

        # Default to a smoothstep curve for a gentle transition.
        t = z_norm * z_norm * (3.0 - 2.0 * z_norm)
        base_radius = p.bottom_radius * (1 - t) + p.top_radius * t

        # --- Apply Profile-Modifying Styles ---
        if p.style == LampshadeStyle.CLASSIC_CURVE:
            linear_radius = p.bottom_radius * (1 - z_norm) + p.top_radius * z_norm
            # MODIFICATION: Use style1_intensity for the primary style effect
            curve_amount = p.curve_factor * p.style1_intensity * z_norm * (1 - z_norm) * (p.bottom_radius + p.top_radius)
            base_radius = linear_radius + curve_amount
        elif p.style == LampshadeStyle.NAUTILUS_SHELL:
            t = z_norm * p.nautilus_chamber_count
            # MODIFICATION: Use style1_intensity for the primary style effect
            base_radius = p.top_radius * np.exp(p.nautilus_growth_factor * p.style1_intensity * t)

        # --- Apply Profile Modifiers in Randomized Order ---
        for mod in p.modifier_chain:
            if mod == "enable_taper_profile":
                taper = 1.0 + p.taper_factor * (z_norm - 0.5)
                base_radius *= taper
            elif mod == "enable_mid_height_modulation":
                mid_height_offset = p.mid_mod_amplitude * np.exp(-((z_norm - 0.5) / 0.2)**2)
                base_radius += mid_height_offset
            elif mod == "enable_squash_stretch":
                base_radius += p.squash_stretch_amplitude * np.sin(z_norm * np.pi * p.squash_stretch_frequency)

        return base_radius

    # MODIFICATION: Added 'intensity' parameter to apply it per-style
    def _calculate_style_effect(self, style, r, theta, z_norm, x, y, z, intensity):
        """Calculates the radial displacement for a single style."""
        p = self.params
        vertex_pos = np.array([x, y, z])
        displacement = 0.0

        if style == LampshadeStyle.WAVE_DEFORMATION:
            displacement = p.wave_amplitude * np.sin(theta * p.wave_frequency) * np.sin(z_norm * np.pi)
        elif style == LampshadeStyle.ORGANIC_VORONOI and p.voronoi_seed_points:
            distances = np.linalg.norm(np.array(p.voronoi_seed_points) - vertex_pos, axis=1)
            min_dist = np.min(distances)
            displacement = -p.voronoi_depth * np.exp(-(min_dist / (p.hex_scale / 4))**2)
        # ... Add all other style calculations here, returning the displacement ...
        elif style == LampshadeStyle.FRACTAL_SURFACE:
            displacement = sum(p.fractal_amplitude * 0.5**i * np.sin(z * (2**i) * np.pi / p.height) * np.cos(theta * (2**i) * 5) for i in range(p.fractal_octaves))
        elif style == LampshadeStyle.GYROID_SURFACE:
            gx, gy, gz = (coord / p.gyroid_frequency for coord in (x, y, z))
            gyroid_val = np.sin(gx)*np.cos(gy) + np.sin(gy)*np.cos(gz) + np.sin(gz)*np.cos(gx)
            displacement = p.gyroid_amplitude * gyroid_val
        elif style == LampshadeStyle.WORLEY_NOISE_CELLS and p.worley_points.size > 0:
            distances = np.linalg.norm(p.worley_points - vertex_pos, axis=1)
            distances.sort()
            worley_val = distances[1] - distances[0]
            displacement = p.worley_amplitude * worley_val / 20.0
        elif style == LampshadeStyle.HEXAGONAL_MESH:
            hex_x = (x / p.hex_scale) * 2/3
            hex_y = (-x/3 + np.sqrt(3)/3 * y) / p.hex_scale
            hex_z = (-x/3 - np.sqrt(3)/3 * y) / p.hex_scale
            hex_val = np.sin(hex_x * np.pi) * np.sin(hex_y * np.pi) * np.sin(hex_z * np.pi)
            displacement = p.hex_depth * hex_val
        elif style == LampshadeStyle.PLEATED_FABRIC:
            displacement = p.pleat_amplitude * np.sin(theta * p.pleat_frequency)
        elif style == LampshadeStyle.FOLDED_PAPER:
            displacement = p.fold_amplitude * (np.abs(np.sin(theta * p.fold_frequency / 2)) - 0.5)
        elif style == LampshadeStyle.BARREL_CACTUS_RIBS:
            displacement = p.cactus_rib_amp * np.cos(theta * p.cactus_rib_count)
        elif style == LampshadeStyle.WOVEN_BASKET:
            displacement = p.weave_amplitude * np.sin(theta * p.weave_frequency_h) * np.cos(z_norm * np.pi * p.weave_frequency_v)
        elif style == LampshadeStyle.MUSHROOM_GILL:
            displacement = -p.gill_depth * np.sin(z_norm * np.pi) * np.sin(theta * p.gill_frequency)
        elif style == LampshadeStyle.PINECONE_SCALES:
             row_height = p.height / p.pinecone_scale_rows
             col_width = 2 * np.pi / p.radial_segments
             theta_offset = (int(z / row_height) % 2) * col_width / 2
             scale_center_theta = np.floor((theta + theta_offset) / col_width) * col_width
             dist_theta = abs(theta - scale_center_theta)
             if dist_theta < col_width / 2:
                 displacement = p.pinecone_scale_curl * (z % row_height / row_height)
        elif style == LampshadeStyle.LEAF_VENATION:
            vein1 = p.vein_depth * np.sin(theta * p.vein_primary_freq)**8
            vein2 = p.vein_depth/2 * np.sin(z_norm * np.pi * p.vein_secondary_freq)**4
            displacement = -(vein1 + vein2)
        elif style == LampshadeStyle.ATTRACTOR_FLOW and p.attractor_points.size > 0:
            distances = np.linalg.norm(p.attractor_points - vertex_pos, axis=1)
            min_dist = np.min(distances)
            displacement = -p.attractor_influence * np.exp(-(min_dist / (p.bottom_radius * 0.1))**2)
        elif style == LampshadeStyle.PHYLLOTAXIS_SURFACE and p.phylo_points.size > 0:
            distances = np.linalg.norm(p.phylo_points - vertex_pos, axis=1)
            min_dist = np.min(distances)
            displacement = p.phylo_influence * np.exp(-(min_dist / p.phylo_point_radius)**2)
        elif style == LampshadeStyle.REACTION_DIFFUSION and p.rd_pattern.size > 0:
            u = int((theta / (2 * np.pi)) * p.rd_pattern.shape[1]) % p.rd_pattern.shape[1]
            v = int(z_norm * p.rd_pattern.shape[0]) % p.rd_pattern.shape[0]
            displacement = p.rd_amplitude * p.rd_pattern[v, u]
        elif style == LampshadeStyle.SUPERFORMULA_2D:
            m, n1, n2, n3 = p.superformula_m, p.superformula_n1, p.superformula_n2, p.superformula_n3
            t1 = np.abs(np.cos(m * theta / 4))**n2
            t2 = np.abs(np.sin(m * theta / 4))**n3
            sf_r = (t1 + t2)**(-1 / n1)
            displacement = r * sf_r - r # Return the change in radius
        elif style == LampshadeStyle.CRYSTAL_GEODE and p.geode_crystals:
            z_dist = abs(z_norm - p.geode_opening_z_norm)
            geode_carve = p.geode_inner_depth * np.exp(-(z_dist / p.geode_opening_size)**2)
            distances = np.linalg.norm(p.geode_crystals - vertex_pos, axis=1)
            min_dist = np.min(distances)
            crystal_effect = p.geode_crystal_size * np.exp(-(min_dist / (p.geode_crystal_size*0.5))**2)
            displacement = -(geode_carve + crystal_effect)
        elif style == LampshadeStyle.CAVE_STALACTITES and p.stalactites:
            stalactite_effect = 0
            for angle, length, phase in p.stalactites:
                angle_diff = min(abs(theta - angle), 2 * np.pi - abs(theta - angle))
                if angle_diff < (length / p.height) * 2: 
                    z_falloff = np.exp(-(angle_diff * 10)**2)
                    stalactite_shape = length * (1 - z_norm) * z_falloff
                    stalactite_effect = max(stalactite_effect, stalactite_shape)
            displacement = -stalactite_effect
        elif style == LampshadeStyle.DRIPPING_LIQUID:
            z_falloff = np.exp(-(z_norm / p.drip_length_factor)**2)
            drip_shape = (np.sin(theta * p.drip_count) + 1) / 2
            displacement = p.drip_amplitude * drip_shape * z_falloff
        elif style == LampshadeStyle.FIBONACCI_SPIRAL:
             displacement = p.fibonacci_amplitude * np.sin(theta * p.fibonacci_density + z_norm * 10)
        elif style == LampshadeStyle.HARMONOGRAPH_SPIRAL:
            decay = np.exp(-z_norm * p.hg_decay)
            x_h = decay * np.sin(z_norm * p.hg_freq1 * 2 * np.pi + p.hg_phase1)
            y_h = decay * np.sin(z_norm * p.hg_freq2 * 2 * np.pi + p.hg_phase2)
            displacement = x_h * y_h * p.hg_amplitude

        # --- NEW STYLES: Advanced Mathematical & Physics-Inspired ---

        elif style == LampshadeStyle.SCHWARZ_PRIMITIVE_TPMS:
            # Schwarz P (Primitive) minimal surface: cos(x) + cos(y) + cos(z) = c
            # Creates a cubic lattice of interconnected spherical chambers
            # Different from Gyroid - P-surface has cubic symmetry, Gyroid is chiral
            freq = p.tpms_p_frequency
            # Map cylindrical coordinates to Cartesian for TPMS evaluation
            tpms_x = x / freq
            tpms_y = y / freq
            tpms_z = z / freq
            # Schwarz P level set function
            schwarz_p = np.cos(tpms_x) + np.cos(tpms_y) + np.cos(tpms_z)
            # Convert level set value to displacement (saddle surfaces between chambers)
            # Threshold determines the "slice" through the minimal surface
            tpms_val = schwarz_p - p.tpms_p_threshold
            # Smooth step function for printable transitions
            displacement = p.tpms_p_amplitude * np.tanh(tpms_val)

        elif style == LampshadeStyle.QUASICRYSTAL_ICOSAHEDRAL:
            # Quasicrystal patterns based on 6D→3D projection (cut-and-project method)
            # Uses multiple wave directions at golden-ratio-related angles for aperiodicity
            # Creates forbidden 5-fold symmetry patterns (Nobel Prize 2011, Dan Shechtman)
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio φ ≈ 1.618
            quasi_sum = 0.0
            if p.quasi_phases.size > 0:
                for i, (angle, phase) in enumerate(p.quasi_phases):
                    # Project position onto wave direction
                    wave_dir_x = np.cos(angle)
                    wave_dir_y = np.sin(angle)
                    # Include z-dependency scaled by golden ratio for 3D aperiodicity
                    proj = wave_dir_x * x + wave_dir_y * y + (z / p.height) * phi * 50
                    # Sum waves with incommensurate frequencies based on golden ratio
                    freq_mult = phi ** (i % 3)  # Vary frequency by powers of phi
                    quasi_sum += np.cos(proj / p.quasi_frequency * freq_mult + phase)
            # Normalize by wave count and apply amplitude
            displacement = p.quasi_amplitude * quasi_sum / max(1, p.quasi_wave_count)

        elif style == LampshadeStyle.LISSAJOUS_CYLINDRICAL_KNOT:
            # 3D Lissajous curves mapped to cylindrical lampshade surface
            # Creates mathematical knots that wrap around the form
            # Based on parametric equations: r(t) = A·sin(at+φ₁), z(t) = B·sin(bt+φ₂), θ(t) = ct
            if p.lissajous_points.size > 0:
                # Calculate distance to nearest point on the Lissajous curve
                distances = np.linalg.norm(p.lissajous_points - vertex_pos, axis=1)
                min_dist = np.min(distances)
                # Create ridge along the curve using Gaussian falloff
                # Knot_width controls how thick the raised pattern appears
                displacement = p.liss_amplitude * np.exp(-(min_dist / p.liss_knot_width)**2)
            else:
                # Fallback: direct parametric evaluation
                t = theta * p.liss_freq_angular
                r_liss = np.sin(p.liss_freq_radial * t + p.liss_phase1)
                z_liss = np.sin(p.liss_freq_vertical * t + p.liss_phase2)
                # Modulate displacement based on how close we are to the Lissajous curve
                displacement = p.liss_amplitude * r_liss * (0.5 + 0.5 * np.cos((z_norm - 0.5 - 0.45 * z_liss) * np.pi * 4))

        elif style == LampshadeStyle.CHLADNI_CYLINDRICAL_RESONANCE:
            # Chladni patterns on cylindrical surface based on vibrating cylinder eigenmode analysis
            # Solves ∇²ψ + k²ψ = 0 on cylinder using separation of variables
            # Solution: ψ(θ,z) = cos(m·θ) · sin(n·π·z/L) for mode (m,n)
            # Nodal lines (zero displacement) form geometric patterns for light diffusion
            total_displacement = 0.0
            for mode_offset in range(p.chladni_superposition):
                # Each superimposed mode adds complexity
                m = p.chladni_mode_m + mode_offset
                n = p.chladni_mode_n + mode_offset
                # Cylindrical eigenmode: circumferential × axial components
                circumferential = np.cos(m * theta)
                axial = np.sin(n * np.pi * z_norm)
                # Combine with decreasing amplitude for higher modes
                mode_amplitude = 1.0 / (1 + mode_offset * 0.5)
                total_displacement += circumferential * axial * mode_amplitude
            # Normalize and apply amplitude
            displacement = p.chladni_amplitude * total_displacement / p.chladni_superposition

        # MODIFICATION: Apply the passed intensity to the final displacement
        return displacement * intensity

    def _generate_vertex_grid(self):
        """
        Generates the complete grid of vertices for both inner and outer surfaces.
        This method uses a reliable nested loop to ensure correct vertex ordering.
        """
        p = self.params
        vertex_list = [] # Use a standard list for appending

        # Loop through each height segment (the "layers" of the lampshade)
        for z_idx in range(p.height_segments + 1):
            z_norm = z_idx / p.height_segments
            z = z_norm * p.height
            base_r = self._get_base_radius_at_z(z_norm)

            # Loop through each radial segment (the "spokes" of the lampshade)
            for seg_idx in range(p.radial_segments):
                theta = 2 * np.pi * seg_idx / p.radial_segments
                
                # Apply angle-modifying effects
                effective_theta = theta
                for mod in p.modifier_chain:
                    if mod == "enable_swirl_effect":
                        effective_theta += p.swirl_strength * np.exp(-((z - p.height/2) / (p.height * 0.3))**2)
                    elif mod == "enable_asymmetric_base" and z_idx == 0:
                        effective_theta += p.asymmetric_base_offset
                    elif mod == "enable_twist_effect":
                        effective_theta += (z_norm ** 1.5) * np.radians(p.twist_angle_deg)


                # Get base coordinates for style calculations
                base_x = base_r * np.cos(effective_theta)
                base_y = base_r * np.sin(effective_theta)

                # MODIFICATION: Calculate each style effect with its own intensity
                style1_effect = self._calculate_style_effect(p.style, base_r, effective_theta, z_norm, base_x, base_y, z, p.style1_intensity)
                
                blended_effect = style1_effect
                if p.enable_blending:
                    style2_effect = self._calculate_style_effect(p.style2, base_r, effective_theta, z_norm, base_x, base_y, z, p.style2_intensity)
                    blended_effect = (1 - p.blend_factor) * style1_effect + p.blend_factor * style2_effect
                
                outer_r = base_r + blended_effect
                
                outer_r = max(p.socket_hole_radius + 1.0, outer_r) if p.fitting_type != FittingType.NONE else max(0.1, outer_r)

                # Calculate the inner radius
                inner_r = outer_r - p.wall_thickness
                if z_idx == 0: # Ensure the bottom inner radius matches the socket hole
                    inner_r = p.socket_hole_radius if p.fitting_type != FittingType.NONE else 0.0
                inner_r = max(0.0, min(inner_r, outer_r - 0.5)) # Clamp to prevent inversion

                # Calculate final XYZ for both vertices and append to the list
                outer_x_final = outer_r * np.cos(effective_theta)
                outer_y_final = outer_r * np.sin(effective_theta)
                inner_x_final = inner_r * np.cos(effective_theta)
                inner_y_final = inner_r * np.sin(effective_theta)
                
                # Apply Positional Modifiers in Randomized Order
                drifted_z = z
                for mod in p.modifier_chain:
                    if mod == "enable_sinusoidal_shear":
                        shear_offset = p.shear_amplitude * np.sin(z_norm * np.pi * p.shear_frequency)
                        outer_x_final += shear_offset
                        inner_x_final += shear_offset
                    elif mod == "enable_vertical_drift" and z_idx > 0:
                        drifted_z += p.vertical_drift_amplitude * np.sin(theta * p.vertical_drift_frequency)
                        drifted_z = np.clip(drifted_z, 0, p.height)

                vertex_list.append([outer_x_final, outer_y_final, drifted_z])
                vertex_list.append([inner_x_final, inner_y_final, drifted_z])

        # Convert the completed list to a NumPy array
        self.vertices = np.array(vertex_list)

    def _precalculate_perforations(self):
        """
        Creates a grid of perforations using a checkerboard pattern to ensure
        printability by avoiding long horizontal bridges.
        """
        p = self.params
        self.perforation_grid = np.zeros((p.height_segments, p.radial_segments), dtype=int)
        
        if p.perforation_chance <= 0 or p.perforation_size < 1:
            return

        size = p.perforation_size

        # Iterate through the grid in steps of 'size' to define blocks
        for z_idx in range(0, p.height_segments, size):
            for seg_idx in range(0, p.radial_segments, size):
                
                # Use a checkerboard pattern for the blocks
                is_checkerboard_on = ((z_idx // size) + (seg_idx // size)) % 2 == 0
                
                if is_checkerboard_on:
                    # Decide if this entire block should be perforated
                    if self._should_perforate(z_idx, seg_idx):
                        # Mark the entire block as a hole
                        for dz in range(size):
                            for ds in range(size):
                                target_z = z_idx + dz
                                target_s = (seg_idx + ds) % p.radial_segments
                                
                                if target_z < p.height_segments:
                                    self.perforation_grid[target_z, target_s] = 1


    def _create_faces(self) -> List[List[int]]:
        """
        Connects the generated vertices into a list of triangular faces.
        This method now handles sealing the boundaries of arbitrarily shaped
        perforation regions.
        """
        p = self.params
        faces = []
        self._precalculate_perforations() # This now creates blocks of perforations

        # Iterate through each quad of the mesh grid
        for z_idx in range(p.height_segments):
            for seg_idx in range(p.radial_segments):
                # --- Get indices for the current quad's corners ---
                v1_o = (z_idx * p.radial_segments + seg_idx) * 2
                v1_i = v1_o + 1
                v2_o = ((z_idx + 1) * p.radial_segments + seg_idx) * 2
                v2_i = v2_o + 1
                
                # --- Get indices for the quad to the right ---
                next_seg_idx = (seg_idx + 1) % p.radial_segments
                v4_o = (z_idx * p.radial_segments + next_seg_idx) * 2
                v4_i = v4_o + 1
                v3_o = ((z_idx + 1) * p.radial_segments + next_seg_idx) * 2
                v3_i = v3_o + 1
                
                # --- Check perforation status of current quad and its neighbors ---
                is_hole_current = self.perforation_grid[z_idx, seg_idx] == 1
                
                # Check right neighbor (wraps around)
                is_hole_right = self.perforation_grid[z_idx, next_seg_idx] == 1
                
                # Check bottom neighbor (no wrapping, check bounds)
                is_hole_down = False
                if z_idx < p.height_segments - 1:
                    is_hole_down = self.perforation_grid[z_idx + 1, seg_idx] == 1

                # --- Create faces based on perforation status ---
                if not is_hole_current:
                    # This is a solid quad, create the outer and inner walls.
                    faces.extend([
                        [v1_o, v2_o, v3_o], [v1_o, v3_o, v4_o], # Outer wall
                        [v1_i, v4_i, v3_i], [v1_i, v3_i, v2_i], # Inner wall (reversed order)
                    ])
                
                # --- Seal boundaries between solid and hole quads ---
                
                # Check vertical boundary (between current and right quad)
                if is_hole_current != is_hole_right:
                    # There is a boundary. Seal it.
                    # If current is solid and right is hole, wall faces outwards.
                    # If current is hole and right is solid, wall faces inwards.
                    # The order of vertices determines the normal direction.
                    if is_hole_current: # Current is hole, right is solid
                        faces.extend([[v4_o, v3_i, v4_i], [v4_o, v3_o, v3_i]])
                    else: # Current is solid, right is hole
                        faces.extend([[v4_o, v4_i, v3_i], [v4_o, v3_i, v3_o]])

                # Check horizontal boundary (between current and down quad)
                if z_idx < p.height_segments - 1 and is_hole_current != is_hole_down:
                    if is_hole_current: # Current is hole, down is solid
                        faces.extend([[v2_o, v2_i, v3_i], [v2_o, v3_i, v3_o]])
                    else: # Current is solid, down is hole
                        faces.extend([[v2_o, v3_o, v3_i], [v2_o, v3_i, v2_i]])

        # Create the top and bottom surfaces (caps)
        self._create_caps(faces)
        return faces

    def _should_perforate(self, z_idx: int, seg_idx: int) -> bool:
        """Determines if a specific quad should be a hole based on the chosen pattern."""
        p = self.params
        if p.perforation_chance == 0.0: return False # Globally disabled
        if z_idx == 0 or z_idx >= p.height_segments - p.perforation_size: return False # No holes on top/bottom edge

        z_norm = z_idx / p.height_segments
        theta = 2 * np.pi * seg_idx / p.radial_segments
        pattern_val = 1.0 # Default is solid

        # --- Calculate Perforation Pattern ---
        if p.perforation_pattern == PerforationPattern.VERTICAL_STRIPE:
            pattern_val = (np.sin(theta * p.perf_pattern_freq) + 1) / 2
        elif p.perforation_pattern == PerforationPattern.SPIRAL_STRIPE:
            pattern_val = (np.sin(theta * p.perf_pattern_freq + z_norm * np.pi * 4) + 1) / 2
        # ... logic for other patterns ...
        elif p.perforation_pattern == PerforationPattern.PERLIN_NOISE:
            u = (theta / (2 * np.pi)) * p.perf_pattern_scale
            v = z_norm * p.perf_pattern_scale / 2
            noise = perlin_noise_3d(u, v, 0)
            pattern_val = 1 if noise > p.perf_pattern_density else 0

        return random.random() < p.perforation_chance * pattern_val

    def _create_caps(self, faces: List[List[int]]):
        """Creates the top and bottom surfaces to close the mesh."""
        p = self.params
        # Bottom cap (connects outer wall to inner socket hole)
        for seg_idx in range(p.radial_segments):
            next_seg_idx = (seg_idx + 1) % p.radial_segments
            v_o_curr = seg_idx * 2
            v_i_curr = v_o_curr + 1
            v_o_next = next_seg_idx * 2
            v_i_next = v_o_next + 1
            if p.fitting_type == FittingType.NONE:
                # Solid bottom for vase mode
                faces.extend([[v_o_curr, v_o_next, 1], [v_i_curr, v_i_next, 0]])
            else:
                faces.extend([[v_o_curr, v_i_next, v_i_curr], [v_o_curr, v_o_next, v_i_next]])


        # Top cap (connects outer wall to inner wall)
        top_z_offset = p.height_segments * p.radial_segments * 2
        for seg_idx in range(p.radial_segments):
            next_seg_idx = (seg_idx + 1) % p.radial_segments
            v_o_curr = top_z_offset + seg_idx * 2
            v_i_curr = v_o_curr + 1
            v_o_next = top_z_offset + next_seg_idx * 2
            v_i_next = v_o_next + 1
            faces.extend([[v_o_curr, v_i_curr, v_i_next], [v_o_curr, v_i_next, v_o_next]])


# --- Utility Functions ---

def perlin_noise_3d(x, y, z, seed=0):
    """
    A simple 3D Perlin noise implementation. Used for organic styles.
    This creates smooth, natural-looking random patterns.
    """
    # This function is a standard Perlin noise implementation and remains unchanged.
    # It's already quite optimized for what it does.
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
    def lerp(a, b, x): return a + x * (b - a)
    def grad(hash, x, y, z):
        h = hash & 15
        u = x if h < 8 else y
        v = y if h < 4 else z if h in (12, 14) else x
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
    xi, yi, zi = (int(i) & 255 for i in (x, y, z))
    xf, yf, zf = (i - int(i) for i in (x, y, z))
    u, v, w = (fade(i) for i in (xf, yf, zf))
    g000 = grad(p[p[p[xi] + yi] + zi], xf, yf, zf)
    g001 = grad(p[p[p[xi] + yi] + zi + 1], xf, yf, zf - 1)
    g010 = grad(p[p[p[xi] + yi + 1] + zi], xf, yf - 1, zf)
    g011 = grad(p[p[p[xi] + yi + 1] + zi + 1], xf, yf - 1, zf - 1)
    g100 = grad(p[p[p[xi + 1] + yi] + zi], xf - 1, yf, zf)
    g101 = grad(p[p[p[xi + 1] + yi] + zi + 1], xf - 1, yf, zf - 1)
    g110 = grad(p[p[p[xi + 1] + yi + 1] + zi], xf - 1, yf - 1, zf)
    g111 = grad(p[p[p[xi + 1] + yi + 1] + zi + 1], xf - 1, yf - 1, zf - 1)
    x1 = lerp(g000, g100, u)
    x2 = lerp(g010, g110, u)
    x3 = lerp(g001, g101, u)
    x4 = lerp(g011, g111, u)
    y1 = lerp(x1, x2, v)
    y2 = lerp(x3, x4, v)
    return (lerp(y1, y2, w) + 1) / 2

def save_stl(stl_mesh: mesh.Mesh, filename: str):
    """Saves the generated mesh to an STL file in the output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    stl_mesh.save(filepath)
    print(f"\nLampshade saved successfully to: {filepath}")

def generate_filename(params: LampshadeParameters, seed: int) -> str:
    """Generates a descriptive filename based on the chosen style and seed."""
    if params.enable_blending:
        return f"{params.style.value}_X_{params.style2.value}_seed_{seed}.stl"
    return f"{params.style.value}_seed_{seed}.stl"


# --- Main Execution Logic ---

class LampshadeApp(tk.Tk):
    """The main application window for the lampshade generator GUI."""
    def __init__(self):
        super().__init__()
        self.title("Lampshade Generator")
        self.last_generated_mesh = None # To hold the generated mesh in memory
        
        # --- GUI Color Scheme ---
        self.BG_COLOR = '#2E2E2E'
        self.FRAME_BG = '#3C3C3C'
        self.BUTTON_BG = '#4A4A4A'
        self.TEXT_COLOR = '#FFFFFF'
        self.ACCENT_COLOR = '#8A2BE2' # Violet
        self.configure(bg=self.BG_COLOR)
        
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('.', background=self.BG_COLOR, foreground=self.TEXT_COLOR, fieldbackground=self.FRAME_BG, bordercolor=self.ACCENT_COLOR)
        style.configure('TFrame', background=self.BG_COLOR)
        style.configure('TLabel', background=self.FRAME_BG, foreground=self.TEXT_COLOR)
        style.configure('TCheckbutton', background=self.FRAME_BG, foreground=self.TEXT_COLOR)
        style.map('TCheckbutton', background=[('active', self.BG_COLOR)])
        style.configure('TLabelFrame', background=self.FRAME_BG, bordercolor=self.ACCENT_COLOR)
        style.configure('TLabelFrame.Label', background=self.FRAME_BG, foreground=self.ACCENT_COLOR)
        style.configure('TButton', background=self.BUTTON_BG, foreground=self.TEXT_COLOR)
        style.map('TButton', background=[('active', '#483D8B')]) # Dark Slate Blue
        style.configure('TScale', background=self.FRAME_BG)
        # MODIFICATION: Add style for Notebook tabs
        style.configure("TNotebook", background=self.BG_COLOR, borderwidth=0)
        style.configure("TNotebook.Tab", background=self.BUTTON_BG, foreground=self.TEXT_COLOR, padding=[5, 2])
        style.map("TNotebook.Tab", background=[("selected", self.ACCENT_COLOR)], foreground=[("selected", self.TEXT_COLOR)])


        # --- Create parameter variables ---
        self.style_var = tk.StringVar(value=LampshadeStyle.CLASSIC_CURVE.value)
        self.intensity1_var = tk.DoubleVar(value=1.0)
        self.radial_segments_var = tk.IntVar(value=128)
        self.height_segments_var = tk.IntVar(value=96)
        self.fitting_var = tk.StringVar(value=FittingType.E27.name)
        self.wall_thickness_var = tk.DoubleVar(value=DEFAULT_WALL_THICKNESS)
        self.min_floor_height_var = tk.DoubleVar(value=DEFAULT_MIN_FLOOR_HEIGHT)
        
        self.perforation_enabled_var = tk.BooleanVar(value=True)
        self.perforation_size_var = tk.IntVar(value=1)
        self.perforation_pattern_var = tk.StringVar(value=PerforationPattern.RANDOM.value)
        self.perforation_chance_var = tk.DoubleVar(value=DEFAULT_PERFORATION_CHANCE)

        self.style2_var = tk.StringVar(value=LampshadeStyle.WAVE_DEFORMATION.value)
        self.intensity2_var = tk.DoubleVar(value=1.0)
        self.blend_factor_var = tk.DoubleVar(value=STYLE_BLEND_FACTOR)
        self.blend_enabled_var = tk.BooleanVar(value=ENABLE_STYLE_BLENDING)
        
        self.modifiers_enabled_var = tk.BooleanVar(value=True)
        self.modifier_vars = {name: tk.BooleanVar(value=is_enabled) for name, is_enabled in MODIFIER_TOGGLES.items()}
        
        self.swirl_strength_var = tk.DoubleVar(value=np.pi / 2)
        self.taper_factor_var = tk.DoubleVar(value=0.0)
        self.mid_mod_amplitude_var = tk.DoubleVar(value=0.0)
        self.shear_amplitude_var = tk.DoubleVar(value=10.0)
        self.squash_stretch_amplitude_var = tk.DoubleVar(value=0.0)
        self.vertical_drift_amplitude_var = tk.DoubleVar(value=5.0)
        self.asymmetric_base_offset_var = tk.DoubleVar(value=0.0)
        self.twist_angle_var = tk.DoubleVar(value=90.0)

        # Boundary variables
        self.min_x_var = tk.DoubleVar(value=DEFAULT_MIN_X_SIZE)
        self.max_x_var = tk.DoubleVar(value=DEFAULT_MAX_X_SIZE)
        self.min_y_var = tk.DoubleVar(value=DEFAULT_MIN_Y_SIZE)
        self.max_y_var = tk.DoubleVar(value=DEFAULT_MAX_Y_SIZE)
        self.min_z_var = tk.DoubleVar(value=DEFAULT_MIN_Z_SIZE)
        self.max_z_var = tk.DoubleVar(value=DEFAULT_MAX_Z_SIZE)

        # Matplotlib Figure and Canvas
        self.figure = None
        self.ax = None
        self.canvas = None
        if MATPLOTLIB_AVAILABLE:
            self.figure = Figure(figsize=(6, 6), dpi=100)
            self.figure.patch.set_facecolor('black')
            self.ax = self.figure.add_subplot(111, projection='3d')
            self.ax.set_facecolor('black')
            self.ax.xaxis.label.set_color('lime')
            self.ax.yaxis.label.set_color('lime')
            self.ax.zaxis.label.set_color('lime')
            self.ax.tick_params(axis='x', colors='lime')
            self.ax.tick_params(axis='y', colors='lime')
            self.ax.tick_params(axis='z', colors='lime')
            self.ax.spines['bottom'].set_color('lime')
            self.ax.spines['top'].set_color('lime') 
            self.ax.spines['right'].set_color('lime')
            self.ax.spines['left'].set_color('lime')

        # Create and pack widgets
        self._create_widgets()
        self._update_all_slider_labels()

    def _create_slider_with_label(self, parent, text, variable, from_, to, is_int=False):
        """Helper function to create a slider and its value label."""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)
        
        label = ttk.Label(frame, text=text, width=20)
        label.pack(side='left')
        
        slider = ttk.Scale(frame, from_=from_, to=to, orient="horizontal", variable=variable)
        slider.pack(side='left', fill='x', expand=True)
        
        value_label = ttk.Label(frame, text="", width=5)
        value_label.pack(side='left', padx=(5, 0))
        
        callback = lambda *args, var=variable, lbl=value_label, is_int=is_int: self._update_slider_label(var, lbl, is_int)
        variable.trace_add('write', callback)
        
        return value_label, slider

    def _update_slider_label(self, variable, label, is_int=False):
        """Updates the text of a single slider's value label."""
        try:
            value = variable.get()
            format_str = "{:.0f}" if is_int else "{:.2f}"
            label.config(text=format_str.format(value))
        except (tk.TclError, ValueError):
            pass
            
    def _update_all_slider_labels(self):
        """Updates all slider labels to their initial values."""
        self._update_slider_label(self.intensity1_var, self.intensity1_value_label)
        self._update_slider_label(self.intensity2_var, self.intensity2_value_label)
        self._update_slider_label(self.wall_thickness_var, self.wall_thickness_value_label)
        self._update_slider_label(self.min_floor_height_var, self.min_floor_height_value_label)
        self._update_slider_label(self.perforation_size_var, self.perforation_size_value_label, is_int=True)
        self._update_slider_label(self.perforation_chance_var, self.perforation_chance_value_label)
        self._update_slider_label(self.blend_factor_var, self.blend_factor_value_label)
        self._update_slider_label(self.min_x_var, self.min_x_value_label)
        self._update_slider_label(self.max_x_var, self.max_x_value_label)
        self._update_slider_label(self.min_y_var, self.min_y_value_label)
        self._update_slider_label(self.max_y_var, self.max_y_value_label)
        self._update_slider_label(self.min_z_var, self.min_z_value_label)
        self._update_slider_label(self.max_z_var, self.max_z_value_label)
        self._update_slider_label(self.radial_segments_var, self.radial_segments_value_label, is_int=True)
        self._update_slider_label(self.height_segments_var, self.height_segments_value_label, is_int=True)
        
        self._update_slider_label(self.swirl_strength_var, self.swirl_strength_value_label)
        self._update_slider_label(self.taper_factor_var, self.taper_factor_value_label)
        self._update_slider_label(self.mid_mod_amplitude_var, self.mid_mod_amplitude_value_label)
        self._update_slider_label(self.shear_amplitude_var, self.shear_amplitude_value_label)
        self._update_slider_label(self.squash_stretch_amplitude_var, self.squash_stretch_amplitude_value_label)
        self._update_slider_label(self.vertical_drift_amplitude_var, self.vertical_drift_amplitude_value_label)
        self._update_slider_label(self.asymmetric_base_offset_var, self.asymmetric_base_offset_value_label)
        self._update_slider_label(self.twist_angle_var, self.twist_angle_value_label)


    def _create_widgets(self):
        """Creates and lays out all the GUI widgets."""
        # MODIFICATION: Main layout changed to a PanedWindow for resizability
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Left Pane: Control Tabs ---
        left_notebook = ttk.Notebook(main_pane)
        main_pane.add(left_notebook, weight=1)

        # --- Right Pane: Preview and Info Tabs ---
        right_notebook = ttk.Notebook(main_pane)
        main_pane.add(right_notebook, weight=2)

        # --- Create Tab Frames ---
        tab_core_shape = ttk.Frame(left_notebook, padding="10")
        tab_style_texture = ttk.Frame(left_notebook, padding="10")
        tab_preview = ttk.Frame(right_notebook, padding="10")
        tab_modifiers = ttk.Frame(right_notebook, padding="10")
        tab_info = ttk.Frame(right_notebook, padding="10")

        left_notebook.add(tab_core_shape, text="Core Shape")
        left_notebook.add(tab_style_texture, text="Style & Texture")
        
        right_notebook.add(tab_preview, text="3D Preview")
        right_notebook.add(tab_modifiers, text="Modifiers")
        right_notebook.add(tab_info, text="Generation Info")

        # --- Populate "Core Shape" Tab ---
        construction_group = ttk.LabelFrame(tab_core_shape, text="Construction", padding="10")
        construction_group.pack(fill="x", pady=(0, 10), expand=True)
        frame = ttk.Frame(construction_group); frame.pack(fill='x', pady=2)
        ttk.Label(frame, text="Fitting Type:").pack(anchor="w")
        fitting_menu = ttk.Combobox(frame, textvariable=self.fitting_var, values=[f.name for f in FittingType])
        fitting_menu.pack(fill="x", pady=(0, 5))
        frame = ttk.Frame(construction_group); frame.pack(fill='x', pady=2)
        self.wall_thickness_value_label, _ = self._create_slider_with_label(frame, "Wall Thickness (mm):", self.wall_thickness_var, 1.0, 20.0)
        frame = ttk.Frame(construction_group); frame.pack(fill='x', pady=2)
        self.min_floor_height_value_label, _ = self._create_slider_with_label(frame, "Floor Height (mm):", self.min_floor_height_var, 0.0, 50.0)

        bounds_group = ttk.LabelFrame(tab_core_shape, text="Generation Boundaries (mm)", padding="10")
        bounds_group.pack(fill="x", pady=(0, 10), expand=True)
        frame = ttk.Frame(bounds_group); frame.pack(fill='x', pady=2)
        self.min_x_value_label, _ = self._create_slider_with_label(frame, "Min X Size:", self.min_x_var, 20, 300)
        frame = ttk.Frame(bounds_group); frame.pack(fill='x', pady=2)
        self.max_x_value_label, _ = self._create_slider_with_label(frame, "Max X Size:", self.max_x_var, 20, 300)
        frame = ttk.Frame(bounds_group); frame.pack(fill='x', pady=2)
        self.min_y_value_label, _ = self._create_slider_with_label(frame, "Min Y Size:", self.min_y_var, 20, 300)
        frame = ttk.Frame(bounds_group); frame.pack(fill='x', pady=2)
        self.max_y_value_label, _ = self._create_slider_with_label(frame, "Max Y Size:", self.max_y_var, 20, 300)
        frame = ttk.Frame(bounds_group); frame.pack(fill='x', pady=2)
        self.min_z_value_label, _ = self._create_slider_with_label(frame, "Min Z Size:", self.min_z_var, 50, 400)
        frame = ttk.Frame(bounds_group); frame.pack(fill='x', pady=2)
        self.max_z_value_label, _ = self._create_slider_with_label(frame, "Max Z Size:", self.max_z_var, 50, 400)

        complexity_group = ttk.LabelFrame(tab_core_shape, text="Mesh Complexity", padding="10")
        complexity_group.pack(fill="x", pady=(0, 10), expand=True)
        frame = ttk.Frame(complexity_group); frame.pack(fill='x', pady=2)
        self.radial_segments_value_label, _ = self._create_slider_with_label(frame, "Radial Segments:", self.radial_segments_var, 8, 512, is_int=True)
        frame = ttk.Frame(complexity_group); frame.pack(fill='x', pady=2)
        self.height_segments_value_label, _ = self._create_slider_with_label(frame, "Height Segments:", self.height_segments_var, 8, 512, is_int=True)

        # --- Populate "Style & Texture" Tab ---
        style_group = ttk.LabelFrame(tab_style_texture, text="Artistic Style", padding="10")
        style_group.pack(fill="x", pady=(0, 10), expand=True)
        frame = ttk.Frame(style_group); frame.pack(fill='x', pady=2)
        ttk.Label(frame, text="Primary Style:").pack(anchor="w")
        style_menu = ttk.Combobox(frame, textvariable=self.style_var, values=[s.value for s in LampshadeStyle])
        style_menu.pack(fill="x", pady=(0, 5))
        frame = ttk.Frame(style_group); frame.pack(fill='x', pady=2)
        self.intensity1_value_label, _ = self._create_slider_with_label(frame, "Primary Intensity:", self.intensity1_var, 0.1, 10.0)

        self.blend_frame = ttk.LabelFrame(style_group, text="Style Blending", padding="10")
        self.blend_frame.pack(fill="x", pady=(10,0), expand=True)
        blend_check = ttk.Checkbutton(self.blend_frame, text="Enable Style Blending", variable=self.blend_enabled_var, command=self._toggle_blend_controls)
        blend_check.pack(anchor="w")
        frame = ttk.Frame(self.blend_frame); frame.pack(fill='x', pady=2)
        ttk.Label(frame, text="Secondary Style:").pack(anchor="w", pady=(5,0))
        self.style2_menu = ttk.Combobox(frame, textvariable=self.style2_var, values=[s.value for s in LampshadeStyle])
        self.style2_menu.pack(fill="x", pady=(0,5))
        frame = ttk.Frame(self.blend_frame); frame.pack(fill='x', pady=2)
        self.intensity2_value_label, self.intensity2_slider = self._create_slider_with_label(frame, "Secondary Intensity:", self.intensity2_var, 0.1, 10.0)
        frame = ttk.Frame(self.blend_frame); frame.pack(fill='x', pady=2)
        self.blend_factor_value_label, self.blend_factor_slider = self._create_slider_with_label(frame, "Blend Factor:", self.blend_factor_var, 0.0, 1.0)

        perforation_group = ttk.LabelFrame(tab_style_texture, text="Perforation", padding="10")
        perforation_group.pack(fill="x", pady=(0, 10), expand=True)
        ttk.Checkbutton(perforation_group, text="Enable Perforations", variable=self.perforation_enabled_var).pack(anchor="w")
        frame = ttk.Frame(perforation_group); frame.pack(fill='x', pady=2)
        ttk.Label(frame, text="Pattern:").pack(anchor="w")
        perforation_pattern_menu = ttk.Combobox(frame, textvariable=self.perforation_pattern_var, values=[p.value for p in PerforationPattern])
        perforation_pattern_menu.pack(fill="x", pady=(0, 5))
        frame = ttk.Frame(perforation_group); frame.pack(fill='x', pady=2)
        self.perforation_size_value_label, _ = self._create_slider_with_label(frame, "Size:", self.perforation_size_var, 1, 10, is_int=True)
        frame = ttk.Frame(perforation_group); frame.pack(fill='x', pady=2)
        self.perforation_chance_value_label, _ = self._create_slider_with_label(frame, "Chance:", self.perforation_chance_var, 0.0, 1.0)

        # --- Populate "Preview" Tab ---
        if MATPLOTLIB_AVAILABLE:
            self.canvas = FigureCanvasTkAgg(self.figure, master=tab_preview)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        else:
            ttk.Label(tab_preview, text="Matplotlib not found.\nPreview is disabled.").pack()

        # --- Populate "Info" Tab ---
        self.info_text = tk.Text(tab_info, height=10, wrap="word", state="disabled", bg=self.FRAME_BG, fg=self.TEXT_COLOR, relief="flat")
        self.info_text.pack(fill="both", expand=True)

        # --- Populate "Modifiers" Tab ---
        ttk.Checkbutton(tab_modifiers, text="Enable Modifiers", variable=self.modifiers_enabled_var, command=self._toggle_modifier_controls).pack(anchor="w")
        
        self.modifier_checkboxes = []
        self.modifier_sliders = []

        for name, var in self.modifier_vars.items():
            check = ttk.Checkbutton(tab_modifiers, text=name[7:].replace('_', ' ').title(), variable=var)
            check.pack(anchor="w", padx=(10,0))
            self.modifier_checkboxes.append(check)
        
        frame = ttk.Frame(tab_modifiers); frame.pack(fill='x', pady=2)
        self.swirl_strength_value_label, self.swirl_strength_slider = self._create_slider_with_label(frame, "Swirl Strength:", self.swirl_strength_var, 0, np.pi)
        frame = ttk.Frame(tab_modifiers); frame.pack(fill='x', pady=2)
        self.taper_factor_value_label, self.taper_factor_slider = self._create_slider_with_label(frame, "Taper Factor:", self.taper_factor_var, -1.0, 1.0)
        frame = ttk.Frame(tab_modifiers); frame.pack(fill='x', pady=2)
        self.mid_mod_amplitude_value_label, self.mid_mod_amplitude_slider = self._create_slider_with_label(frame, "Mid Amplitude:", self.mid_mod_amplitude_var, -50, 50)
        frame = ttk.Frame(tab_modifiers); frame.pack(fill='x', pady=2)
        self.shear_amplitude_value_label, self.shear_amplitude_slider = self._create_slider_with_label(frame, "Shear Amplitude:", self.shear_amplitude_var, 0, 50)
        frame = ttk.Frame(tab_modifiers); frame.pack(fill='x', pady=2)
        self.squash_stretch_amplitude_value_label, self.squash_stretch_amplitude_slider = self._create_slider_with_label(frame, "Squash/Stretch:", self.squash_stretch_amplitude_var, -50, 50)
        frame = ttk.Frame(tab_modifiers); frame.pack(fill='x', pady=2)
        self.vertical_drift_amplitude_value_label, self.vertical_drift_amplitude_slider = self._create_slider_with_label(frame, "Vertical Drift:", self.vertical_drift_amplitude_var, 0, 20)
        frame = ttk.Frame(tab_modifiers); frame.pack(fill='x', pady=2)
        self.asymmetric_base_offset_value_label, self.asymmetric_base_offset_slider = self._create_slider_with_label(frame, "Base Offset:", self.asymmetric_base_offset_var, -np.pi/4, np.pi/4)
        frame = ttk.Frame(tab_modifiers); frame.pack(fill='x', pady=2)
        self.twist_angle_value_label, self.twist_angle_slider = self._create_slider_with_label(frame, "Twist Angle:", self.twist_angle_var, 0, 360)

        self.modifier_sliders.extend([
            self.swirl_strength_slider, self.taper_factor_slider, self.mid_mod_amplitude_slider,
            self.shear_amplitude_slider, self.squash_stretch_amplitude_slider,
            self.vertical_drift_amplitude_slider, self.asymmetric_base_offset_slider,
            self.twist_angle_slider
        ])

        # --- Action Buttons (at the bottom of the left pane) ---
        button_frame = ttk.Frame(left_notebook)
        button_frame.pack(fill="x", pady=10, side="bottom")
        
        generate_button = ttk.Button(button_frame, text="Generate & Preview", command=self._generate_and_preview)
        generate_button.pack(side="left", expand=True, fill="x", padx=(0,5))
        
        randomize_button = ttk.Button(button_frame, text="Randomize", command=self._randomize_parameters)
        randomize_button.pack(side="left", expand=True, fill="x", padx=5)

        self.save_button = ttk.Button(button_frame, text="Save", command=self._save_lampshade, state="disabled")
        self.save_button.pack(side="left", expand=True, fill="x", padx=(5,0))
        
        self._toggle_blend_controls() # Set initial state
        self._toggle_modifier_controls() # Set initial state

    def _randomize_parameters(self):
        """Randomizes the creative parameters in the GUI."""
        self.style_var.set(random.choice(list(LampshadeStyle)).value)
        self.style2_var.set(random.choice(list(LampshadeStyle)).value)
        self.intensity1_var.set(random.uniform(0.7, 1.5))
        self.intensity2_var.set(random.uniform(0.7, 1.5))
        self.radial_segments_var.set(random.randint(64, 256))
        self.height_segments_var.set(random.randint(48, 192))
        
        self.perforation_enabled_var.set(random.choice([True, False]))
        self.perforation_size_var.set(random.randint(1, 5))
        self.perforation_pattern_var.set(random.choice(list(PerforationPattern)).value)
        self.perforation_chance_var.set(random.uniform(0.05, 0.25))
        
        self.blend_enabled_var.set(random.choice([True, False]))
        self.blend_factor_var.set(random.random())
        
        self.modifiers_enabled_var.set(random.choice([True, False]))
        for var in self.modifier_vars.values():
            var.set(random.choice([True, False]))
            
        self.swirl_strength_var.set(random.uniform(np.pi / 4, np.pi))
        self.taper_factor_var.set(random.uniform(-0.8, 0.8))
        self.mid_mod_amplitude_var.set(random.uniform(-20, 20))
        self.shear_amplitude_var.set(random.uniform(5, 20))
        self.squash_stretch_amplitude_var.set(random.uniform(-20, 20))
        self.vertical_drift_amplitude_var.set(random.uniform(2, 8))
        self.asymmetric_base_offset_var.set(random.uniform(-np.pi / 8, np.pi / 8))
        self.twist_angle_var.set(random.uniform(30, 180))
            
        self._toggle_blend_controls()
        self._toggle_modifier_controls()
        self._update_all_slider_labels()

    def _toggle_blend_controls(self):
        """Enables or disables the secondary style controls based on the checkbox."""
        state = "normal" if self.blend_enabled_var.get() else "disabled"
        self.style2_menu.configure(state=state)
        self.intensity2_slider.configure(state=state)
        self.blend_factor_slider.configure(state=state)

    def _toggle_modifier_controls(self):
        """Enables or disables the individual modifier checkboxes and sliders."""
        state = "normal" if self.modifiers_enabled_var.get() else "disabled"
        for checkbox in self.modifier_checkboxes:
            checkbox.config(state=state)
        for slider in self.modifier_sliders:
            slider.config(state=state)

        if not self.modifiers_enabled_var.get():
            for var in self.modifier_vars.values():
                var.set(False)

    def _show_loading_indicator(self):
        """Displays a 'Generating...' message on the canvas."""
        if not MATPLOTLIB_AVAILABLE or self.ax is None or self.canvas is None:
            return
        self.ax.clear()
        self.ax.text(0.5, 0.5, 0.5, "Generating...\nPlease Wait", 
                     ha='center', va='center', color='lime', fontsize=20)
        self.ax.set_axis_off()
        self.canvas.draw()
        self.update_idletasks()

    def _update_preview_canvas(self, stl_mesh):
        """Draws the provided STL mesh onto the embedded canvas."""
        if not MATPLOTLIB_AVAILABLE or self.ax is None or self.canvas is None:
            return

        # Clear the previous plot
        self.ax.clear()
        self.ax.set_axis_on()

        # Calculate face colors based on Z-coordinate
        z_centers = stl_mesh.vectors[:, :, 2].mean(axis=1)
        # Normalize Z values to range [0, 1] for the colormap
        normalized_z = (z_centers - z_centers.min()) / (z_centers.max() - z_centers.min())
        cmap = plt.get_cmap('viridis')
        face_colors = cmap(normalized_z)

        # Draw the new mesh
        self.ax.add_collection3d(mplot3d.art3d.Poly3DCollection(
            stl_mesh.vectors,
            facecolors=face_colors,
            edgecolors='#00FF00', # Neon Green
            linewidths=0.1,
            alpha=1.0
        ))

        # Scale the plot to fit the mesh with a 10% zoom
        points = stl_mesh.points.flatten()
        max_abs_coord = np.max(np.abs(points)) * 0.9 # Zoom in by 10%
        self.ax.set_xlim(-max_abs_coord, max_abs_coord)
        self.ax.set_ylim(-max_abs_coord, max_abs_coord)
        self.ax.set_zlim(0, stl_mesh.z.max() * 1.0) # Adjust Z limit slightly

        # Set labels and a fixed viewing angle
        self.ax.set_xlabel('X (mm)')
        self.ax.set_ylabel('Y (mm)')
        self.ax.set_zlabel('Z (mm)')
        self.ax.view_init(elev=30., azim=45)

        # Redraw the canvas
        self.canvas.draw()

    def _update_info_zone(self, params, seed, stl_mesh):
        """Updates the info text widget with the latest generation data."""
        config_str = f"Seed: {seed}\n\n"
        config_str += params.get_config_string()
        config_str += f"\n\n--- Mesh Info ---\n"
        config_str += f"  Vertices: {len(stl_mesh.vectors) * 3}\n"
        config_str += f"  Faces: {len(stl_mesh.vectors)}\n"


        self.info_text.config(state="normal") # Enable writing
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, config_str)
        self.info_text.config(state="disabled") # Disable writing

    def _gather_parameters_and_generate(self):
        """Gathers parameters from the GUI, runs the generation, and returns the mesh."""
        params = LampshadeParameters()
        
        # Set parameters from GUI
        params.style = LampshadeStyle(self.style_var.get())
        params.style2 = LampshadeStyle(self.style2_var.get())
        params.style1_intensity = self.intensity1_var.get()
        params.style2_intensity = self.intensity2_var.get()
        params.fitting_type = FittingType[self.fitting_var.get()]
        params.wall_thickness = self.wall_thickness_var.get()
        params.min_floor_height = self.min_floor_height_var.get()

        params.perforation_size = self.perforation_size_var.get()
        params.perforation_pattern = PerforationPattern(self.perforation_pattern_var.get())
        params.perforation_chance = self.perforation_chance_var.get() if self.perforation_enabled_var.get() else 0.0

        params.blend_factor = self.blend_factor_var.get() if self.blend_enabled_var.get() else 0.0
        params.enable_blending = self.blend_enabled_var.get()
        
        if self.modifiers_enabled_var.get():
            for name, var in self.modifier_vars.items():
                setattr(params, name, var.get())
            # MODIFICATION: Get values from all the new modifier sliders
            params.swirl_strength = self.swirl_strength_var.get()
            params.taper_factor = self.taper_factor_var.get()
            params.mid_mod_amplitude = self.mid_mod_amplitude_var.get()
            params.shear_amplitude = self.shear_amplitude_var.get()
            params.squash_stretch_amplitude = self.squash_stretch_amplitude_var.get()
            params.vertical_drift_amplitude = self.vertical_drift_amplitude_var.get()
            params.asymmetric_base_offset = self.asymmetric_base_offset_var.get()
            params.twist_angle_deg = self.twist_angle_var.get()
        else:
            for name in self.modifier_vars:
                setattr(params, name, False)
            params.twist_angle_deg = 0

        
        # Override random dimensions with values from the GUI's boundary settings
        params.height = random.uniform(self.min_z_var.get(), self.max_z_var.get())
        params.top_radius = random.uniform(self.min_x_var.get() / 2, (self.max_x_var.get() / 2) * 0.9)
        params.bottom_radius = random.uniform(self.min_y_var.get() / 2, self.max_y_var.get() / 2)

        params.radial_segments = self.radial_segments_var.get()
        params.height_segments = self.height_segments_var.get()


        # Build the modifier chain AFTER setting the flags from the GUI
        params.build_modifier_chain()
        
        seed = int.from_bytes(os.urandom(4), 'big')
        random.seed(seed)
        np.random.seed(seed)

        generator = LampshadeGenerator(params, seed)
        stl_mesh = generator.generate()
        
        # Update the info zone after generation
        self._update_info_zone(params, seed, stl_mesh)
        
        return stl_mesh

    def _generate_and_preview(self):
        """Generates the lampshade and displays it in the embedded preview."""
        try:
            # Validate boundary inputs
            if self.min_x_var.get() > self.max_x_var.get() or \
               self.min_y_var.get() > self.max_y_var.get() or \
               self.min_z_var.get() > self.max_z_var.get():
                messagebox.showerror("Invalid Boundaries", "Minimum size cannot be greater than maximum size.")
                return

            self._show_loading_indicator()
            print("Generating lampshade...")
            self.last_generated_mesh = self._gather_parameters_and_generate()
            
            if self.last_generated_mesh is None:
                messagebox.showerror("Error", "Failed to generate lampshade.")
                return
            
            # Update the preview canvas
            self._update_preview_canvas(self.last_generated_mesh)

            # Enable the save button
            self.save_button.config(state="normal")

        except Exception as e:
            messagebox.showerror("Generation Error", f"An error occurred during generation:\n{e}")
            self.last_generated_mesh = None
            self.save_button.config(state="disabled")

    def _save_lampshade(self):
        """Saves the last generated lampshade to a user-specified file."""
        if self.last_generated_mesh is None:
            messagebox.showwarning("No Lampshade", "Please generate a lampshade before trying to save.")
            return
            
        filename = asksaveasfilename(
            initialdir=OUTPUT_DIR,
            title="Save Lampshade STL",
            defaultextension=".stl",
            filetypes=[("STL Files", "*.stl")]
        )
        if filename:
            try:
                self.last_generated_mesh.save(filename)
                messagebox.showinfo("Success", f"Lampshade saved successfully to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"An error occurred while saving:\n{e}")


def main():
    """Main function to parse arguments and run the generation process."""
    parser = argparse.ArgumentParser(description="Procedural Lampshade Generator.")
    parser.add_argument('--gui', action='store_true', help="Run the graphical user interface.")
    parser.add_argument('--style', type=str, choices=[s.value for s in LampshadeStyle], help="Specify a single artistic style to generate.")
    parser.add_argument('--seed', type=int, help="Seed for the random number generator for reproducible results.")
    parser.add_argument('--complexity', type=str, choices=[c.value for c in Complexity if c.value != 'random'], default='medium', help="Set the polygon complexity level.")
    parser.add_argument('--batch', type=int, default=1, help="Number of lampshades to generate in a batch.")
    parser.add_argument('--intensity', type=float, help="Set the intensity of the artistic style (e.g., 0.5 for subtle, 2.0 for exaggerated).")
    parser.add_argument('--no-preview', action='store_true', help="Disable the 3D preview after generation (GUI only).")
    parser.add_argument('--no-random-modifiers', action='store_true', help="Disable all random modifiers for a pure style.")
    args = parser.parse_args()

    if args.gui:
        # Launch the Tkinter GUI
        app = LampshadeApp()
        app.mainloop()
    else:
        # Run in command-line mode
        # --- Seeding ---
        base_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), 'big')
        print(f"Using base seed: {base_seed}")

        latest_filename = ""
        for i in range(args.batch):
            # Use a unique seed for each item in the batch for variety
            current_seed = base_seed + i
            random.seed(current_seed)
            np.random.seed(current_seed)

            if args.batch > 1:
                print(f"\n--- Generating Lampshade {i+1}/{args.batch} (Seed: {current_seed}) ---")

            # --- Parameter Initialization ---
            params = LampshadeParameters()

            # Set complexity based on arguments
            complexity_level = args.complexity
            if complexity_level == 'low':
                params.radial_segments, params.height_segments = random.randint(16, 32), random.randint(12, 24)
            elif complexity_level == 'medium':
                params.radial_segments, params.height_segments = random.randint(64, 128), random.randint(48, 96)
            elif complexity_level == 'high':
                params.radial_segments, params.height_segments = random.randint(128, 256), random.randint(96, 192)

            # Override style if specified
            if args.style:
                params.style = LampshadeStyle(args.style)
                
            # Override intensity if specified
            if args.intensity is not None:
                params.style1_intensity = args.intensity

            # Disable modifiers if requested
            if args.no_random_modifiers:
                for attr in dir(params):
                    if attr.startswith("enable_"):
                        setattr(params, attr, False)

            # Build the modifier chain AFTER any potential overrides
            params.build_modifier_chain()

            # --- Generation ---
            generator = LampshadeGenerator(params, current_seed)
            stl_mesh = generator.generate()
            filename = generate_filename(params, current_seed)
            save_stl(stl_mesh, filename)
            latest_filename = filename

        # --- Preview (optional) ---
        if not args.no_preview and args.batch == 1:
            print("\nPreview is only available in GUI mode. Run with --gui flag.")


if __name__ == "__main__":
    main()
