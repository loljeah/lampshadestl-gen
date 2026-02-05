# Lampshade STL Generator

A procedural 3D lampshade generator that creates unique, 3D-printable lampshade designs using mathematical formulas, natural patterns, and artistic styles. Export your creations as STL files ready for 3D printing.

## Features

- **20+ Artistic Styles** - From classic curves to fractal surfaces, organic voronoi patterns to nautilus shells
- **Procedural Generation** - Each lampshade is mathematically generated with endless variations
- **3D-Printable Output** - Manifold STL files with proper wall thickness and sealed geometry
- **Standard Socket Fittings** - Built-in support for E27, E14, and B22 light bulb sockets
- **Customizable Perforations** - 11 perforation patterns for beautiful light diffusion effects
- **8 Shape Modifiers** - Swirl, twist, taper, and more to transform any base style
- **Style Blending** - Combine two styles together for unique hybrid designs
- **Reproducible Seeds** - Recreate any design using its seed number
- **Dual Interface** - Both GUI and command-line modes available
- **Batch Generation** - Generate multiple unique designs in one run

## Installation

### Requirements

- Python 3.7+
- NumPy
- numpy-stl

### Optional Dependencies

- matplotlib (for 3D preview in GUI)
- scipy (for reaction-diffusion patterns)
- tkinter (for GUI mode - usually included with Python)

### Install Dependencies

```bash
pip install numpy numpy-stl

# Optional - for full functionality
pip install matplotlib scipy
```

## Usage

### GUI Mode

Launch the graphical interface for interactive design:

```bash
python lampshadegen-gui6.py --gui
```

The GUI provides:
- Style selection dropdowns with blending controls
- Real-time parameter adjustment
- 3D preview visualization
- Save dialog for STL export
- Batch generation controls

### Command Line Mode

Generate lampshades directly from the terminal:

```bash
# Generate a random lampshade
python lampshadegen-gui6.py

# Generate with a specific style
python lampshadegen-gui6.py --style nautilus_shell

# Generate with a specific seed (reproducible)
python lampshadegen-gui6.py --seed 12345

# High resolution output
python lampshadegen-gui6.py --complexity high

# Batch generate 5 unique designs
python lampshadegen-gui6.py --batch 5

# Combine options
python lampshadegen-gui6.py --style fractal_surface --intensity 1.5 --complexity high --seed 42
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--gui` | Launch graphical interface |
| `--style STYLE` | Use a specific style (see available styles below) |
| `--seed NUMBER` | Random seed for reproducible generation |
| `--complexity LEVEL` | Mesh quality: `low`, `medium`, `high` |
| `--batch COUNT` | Generate multiple lampshades |
| `--intensity FLOAT` | Style effect multiplier (0.5 = subtle, 2.0 = exaggerated) |
| `--no-random-modifiers` | Disable random modifier application |

## Available Styles

### Geometric & Classic
| Style | Description |
|-------|-------------|
| `classic_curve` | Smooth, elegant tapered curve |
| `hexagonal_mesh` | 3D hexagonal tessellation pattern |
| `wave_deformation` | Flowing wave patterns |
| `pleated_fabric` | Fabric-like pleated texture |
| `folded_paper` | Origami-inspired paper folds |
| `barrel_cactus_ribs` | Vertical ribbed cactus pattern |

### Organic & Natural
| Style | Description |
|-------|-------------|
| `organic_voronoi` | Natural cell-like voronoi pattern |
| `woven_basket` | Woven textile texture |
| `mushroom_gill` | Mushroom gill-like ridges |
| `pinecone_scales` | Overlapping pinecone scale pattern |
| `leaf_venation` | Organic leaf vein patterns |
| `nautilus_shell` | Nautilus shell spiral geometry |
| `worley_noise_cells` | Cellular noise-based pattern |

### Algorithmic & Fractal
| Style | Description |
|-------|-------------|
| `fractal_surface` | Multi-octave fractal geometry |
| `gyroid_surface` | Mathematical minimal surface |
| `attractor_flow` | Strange attractor patterns (Clifford/De Jong) |
| `phyllotaxis_surface` | Fibonacci spiral arrangement |
| `reaction_diffusion` | Turing pattern simulation |
| `superformula_2d` | Superformula-based shapes |
| `fibonacci_spiral` | Fibonacci spiral pattern |
| `harmonograph_spiral` | Harmonograph-inspired spirals |

### Experimental
| Style | Description |
|-------|-------------|
| `crystal_geode` | Crystal geode interior texture |
| `cave_stalactites` | Stalactite formations |
| `dripping_liquid` | Liquid drip effects |

## Shape Modifiers

Modifiers transform the base style to create unique variations:

| Modifier | Effect |
|----------|--------|
| Swirl Effect | Rotational twist around vertical axis |
| Taper Profile | Conical tapering from top to bottom |
| Mid-Height Modulation | Bulge or pinch at middle height |
| Sinusoidal Shear | Wave-based shear distortion |
| Squash/Stretch | Vertical compression or extension |
| Vertical Drift | Vertical wave offset pattern |
| Asymmetric Base | Off-center base offset |
| Twist Effect | Helical twist deformation |

## Perforation Patterns

Add perforations for beautiful light diffusion:

| Pattern | Description |
|---------|-------------|
| `random` | Randomly distributed holes |
| `vertical_stripe` | Vertical stripe pattern |
| `spiral_stripe` | Spiral stripe pattern |
| `checkerboard` | Checkerboard grid |
| `hex_grid` | Hexagonal grid pattern |
| `polka_dot` | Polka dot arrangement |
| `wave_vertical` | Vertical wave pattern |
| `perlin_noise` | Perlin noise-based distribution |
| `rain_drops` | Raindrop-like pattern |
| `zebra_stripes` | Zebra stripe pattern |
| `starfield` | Star-like scattered pattern |

## Output

Generated files are saved to the `generated_lampshades/` directory.

### File Naming Convention

```
{style_name}[_X_{secondary_style}]_seed_{seed_number}.stl
```

Examples:
- `nautilus_shell_seed_12345.stl`
- `fractal_surface_X_organic_voronoi_seed_67890.stl`

### Default Dimensions

| Parameter | Default Value |
|-----------|---------------|
| Height | 100-250 mm (randomized) |
| Top Radius | 25-110 mm |
| Bottom Radius | 25-110 mm |
| Wall Thickness | 2.0 mm |
| Floor Height | 5.0 mm |
| Socket | E27 (27mm diameter) |

### Mesh Complexity

| Level | Radial Segments | Height Segments |
|-------|-----------------|-----------------|
| Low | 16-32 | 12-24 |
| Medium | 64-128 | 48-96 |
| High | 128-256 | 96-192 |

## Examples

### Generate a High-Detail Nautilus Shell Lampshade

```bash
python lampshadegen-gui6.py --style nautilus_shell --complexity high --seed 2024
```

### Batch Generate 10 Random Organic Designs

```bash
python lampshadegen-gui6.py --style organic_voronoi --batch 10
```

### Create a Subtle Fractal Design

```bash
python lampshadegen-gui6.py --style fractal_surface --intensity 0.5 --no-random-modifiers
```

### Generate Pure Style Without Modifiers

```bash
python lampshadegen-gui6.py --style gyroid_surface --no-random-modifiers
```

## 3D Printing Tips

- **Wall Thickness**: Default 2.0mm works well for most FDM printers
- **Orientation**: Print with the wider end down for stability
- **Supports**: May be needed depending on style and overhang angles
- **Material**: PLA or PETG recommended; translucent filaments create beautiful effects
- **Infill**: 0% (vase mode) or 10-15% for structural areas

## Socket Compatibility

The generator supports standard light bulb socket fittings:

| Type | Diameter | Common Name |
|------|----------|-------------|
| E27 | 27mm | Standard Edison Screw |
| E14 | 14mm | Small Edison Screw |
| B22 | 22mm | Bayonet Cap |

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to submit issues and pull requests.
