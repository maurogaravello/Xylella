# Xylella

A sophisticated Python-based simulation framework for modeling age-structured, spatially-distributed population dynamics using finite volume methods and nonlocal kernels.

## Overview

**Xylella** implements numerical solutions for complex nonlocal systems, particularly designed for modeling the dynamics of *Xylella fastidiosa* spread in olive tree populations. The program combines:

- **Age-structured population models** with separate juvenile and adult dynamics
- **Spatial heterogeneity** across 2D domains
- **Nonlocal interactions** via convolution kernels for dispersal and aggregation
- **Advanced finite volume methods** (Lax-Friedrichs and upwind schemes)
- **Parallel computing** for efficient simulation and visualization
- **Comprehensive visualization** with contour plots, surface plots, and animations

## Key Features

✨ **Flexible Numerical Methods**
- Lax-Friedrichs scheme for stability with controlled diffusion
- Upwind scheme for monotonicity preservation
- Dimensional splitting for multi-dimensional problems
- Adaptive time-stepping respecting CFL conditions

🔄 **Sophisticated Population Modeling**
- Age-structured populations (juveniles, adults, trees)
- Time-dependent fertility and source terms
- Space and age-dependent mortality rates
- Nonlocal velocity fields driven by weed distributions and tree interactions

⚙️ **Configuration-Driven Architecture**
- Parameter files define all model components (kernels, velocities, initial conditions)
- Easy parameterization for different scenarios
- Support for alternative convolution methods

📊 **Visualization & Analysis**
- Contour plots of population distributions
- 3D surface plots with customizable resolution
- Mass conservation tracking (L¹ norms)
- L∞ norm monitoring for solution stability
- Automated movie generation from plot sequences
- Parallel plot generation for performance

## Quick Start

### Prerequisites

```bash
Python 3.8+
numpy
scipy
matplotlib
```

Install dependencies:
```bash
pip install numpy scipy matplotlib
```

### Basic Usage

1. **Create a simulation directory** with a parameter file:

```bash
mkdir my_simulation
cd my_simulation
# Create parameters.py (see Configuration section)
```

2. **Run the simulation**:

```bash
python main_xylella.py . -s -pn 4
```

This runs the simulation using 4 processors.

3. **Generate plots** from saved results:

```bash
python main_xylella.py . -p -pn 4 -cb
```

This generates plots with color bars using 4 processors.

4. **Create movies** from plot sequences:

```bash
python main_xylella.py . -m
```

## Configuration

Each simulation requires a `parameters.py` file in the simulation directory. Here's a minimal example:

```python
import numpy as np
from scipy import signal

# Domain specification
x1_limits = [0.0, 10.0]
x2_limits = [0.0, 10.0]
a_limits = [0.0, 5.0]

# Grid resolution
N_x1 = 100
N_x2 = 100
N_a = 50

# Temporal parameters
T = 10.0                    # Final simulation time
cfl = 0.5                   # CFL parameter
N_pictures = 20             # Number of snapshots
LF = True                   # Use Lax-Friedrichs (False for upwind)

# Initial conditions
def u0(a, x1, x2):
    """Initial juvenile distribution"""
    return 0.1 * np.exp(-(x1-5.0)**2 - (x2-5.0)**2 - a)

def w0(a, x1, x2):
    """Initial adult distribution"""
    return 0.05 * np.exp(-(x1-5.0)**2 - (x2-5.0)**2 - a)

def z0(x1, x2):
    """Initial tree distribution"""
    return 1.0 * np.exp(-(x1-5.0)**2 - (x2-5.0)**2)

# Mortality rates
def delta_u(a, x1, x2):
    """Juvenile mortality"""
    return 0.1 + 0.05*a

def delta_w(a, x1, x2):
    """Adult mortality"""
    return 0.05 + 0.02*a

def delta_z(x1, x2, w):
    """Tree mortality (depends on adult population)"""
    return 0.01 * w

# Boundary and source terms
def beta(t, x1, x2):
    """Juvenile production (fertility)"""
    return 0.2 * np.exp(-0.1*t)

def s(t, x1, x2):
    """Source term for new trees"""
    return 0.0

def lmbda(x1, x2):
    """Loss parameter for trees"""
    return 1.0

# Weeds (environmental factor)
def weeds(x1, x2):
    """Weed distribution"""
    return 1.0 + 0.5 * np.sin(0.5*x1) * np.cos(0.5*x2)

# Kernels for nonlocal interactions
def eta_mu(x1, x2):
    """Weed convolution kernel"""
    r_sq = x1**2 + x2**2
    return np.exp(-r_sq / 0.5) / (np.pi * 0.5)

def grad_eta(x1, x2):
    """Gradient kernel (returns two components)"""
    r_sq = x1**2 + x2**2
    kernel = np.exp(-r_sq / 1.0) / (np.pi * 1.0)
    return x1 * kernel, x2 * kernel

def eta_z(x1, x2):
    """Tree interaction kernel"""
    r_sq = x1**2 + x2**2
    return np.exp(-r_sq / 2.0) / (np.pi * 2.0)

# Velocity functions
def velocity_juveniles(conv_x, conv_y):
    """Velocity field for juveniles"""
    return conv_x, conv_y

def velocity_adults(conv_x, conv_y):
    """Velocity field for adults"""
    return conv_x, conv_y

# Kernel mesh limits (support of kernels)
x1_eta_mu_limits = [-3.0, 3.0]
x2_eta_mu_limits = [-3.0, 3.0]
x1_grad_eta_limits = [-3.0, 3.0]
x2_grad_eta_limits = [-3.0, 3.0]
x1_eta_z_limits = [-3.0, 3.0]
x2_eta_z_limits = [-3.0, 3.0]

# Optional: visualization parameters
density_limits = [[0, 1], [0, 0.5], [0, 2]]  # [u_range, w_range, z_range]
rcount = 50     # Surface plot rows
ccount = 50     # Surface plot columns

# Optional: convolution method (default is signal.oaconvolve)
# convolution = signal.convolve2d
```

## Command-Line Options

```
usage: main_xylella.py [-h] [-s] [-p] [-pn PROCESSORS_NUMBER] [-m] [-cb] dir_name

positional arguments:
  dir_name                      Directory containing parameters.py

optional arguments:
  -h, --help                    Show help message
  -s, --simulation              Execute the simulation
  -p, --plot                    Generate plots from saved simulation data
  -pn, --processors_number N    Number of processors (default: 1)
  -m, --movie                   Create movies from plot sequences
  -cb, --color_bar              Add color bar to contour plots
```

## Project Structure

```
Python/
├── main_xylella.py               # Main entry point and orchestrator
├── src/
│   ├── SimulationState.py        # Data structures for population states
│   ├── evolution.py              # One-step evolution algorithm
│   ├── finite_volume_methods.py  # Lax-Friedrichs and upwind schemes
│   ├── kernel_meshesgrids.py     # Kernel mesh utilities
│   ├── kernels.py                # Kernel definition and creation
│   ├── meshes.py                 # Spatial and age mesh generation
│   ├── mortality.py              # Mortality parameter management
│   ├── params_scheme.py          # Simulation parameters dataclass
│   ├── plot_functions.py         # Visualization functions
│   └── save_load.py              # I/O operations
├── simulations/                  # Example simulation configurations
│   ├── Aveiro_LF/                # Aveiro scenario with Lax-Friedrichs
│   └── Aveiro_upwind/            # Aveiro scenario with upwind
└── tests/                        # Test configurations
    ├── velocity_down_no_source_LF/
    ├── velocity_down_no_source_upwind/
    ├── velocity_left_no_source_LF/
    └── velocity_left_no_source_upwind/
```

## Output Files

When running a simulation, the following directory structure is created:

```
simulations/my_simulation/Simulation/
├── saving_000.npz               # State snapshots
├── saving_001.npz
├── ...
├── saving_019.npz
├── Mass.npz                     # Mass and norm conservation data
├── log_Simulation.txt           # Detailed simulation log
├── log_Plots.txt                # Plot generation log
├── log_Movie.txt                # Movie creation log
├── plot_contour_juveniles_*.png # 2D contour plots
├── plot_contour_adults_*.png
├── plot_contour_trees_*.png
├── plot_surface_juveniles_*.png # 3D surface plots
├── plot_surface_adults_*.png
├── plot_surface_trees_*.png
├── plot_mass_*.png              # Conservation metrics
└── movie_*.avi                  # Video animations
```

## Mathematical Model

The program solves a system of age-structured, spatially-distributed balance equations:

**Juvenile population:**
$$\frac{\partial u}{\partial t} + \frac{\partial u}{\partial a} + \nabla \cdot (\xi_u u) = -\delta_u u$$

**Adult population:**
$$\frac{\partial w}{\partial t} + \frac{\partial w}{\partial a} + \nabla \cdot (\xi_w w) = -\delta_w w$$

**Tree density:**
$$\frac{\partial z}{\partial t} = -\lambda \delta_z(w_1) z$$

**Boundary conditions:**
- $u(t, 0, x_1, x_2) = \beta(t, x_1, x_2)$ (fertility term)
- $w(t, a_{\max}, x_1, x_2) = u(t, a_J^{\max}, x_1, x_2)$ (maturation)

**Nonlocal velocity fields:**
$$\xi_u = v_u\left(-\text{weeds} \cdot \nabla\eta * \frac{u}{\Delta a}, \ldots\right)$$
$$\xi_w = v_w\left(-(\text{weeds} + z * \eta_z) \cdot \nabla\eta * \frac{w}{\Delta a}, \ldots\right)$$

where $*$ denotes 2D convolution.

## Numerical Methods

### Finite Volume Scheme

The program discretizes conservation laws using:

1. **Lax-Friedrichs scheme** (default):
   - Unconditionally stable in the discrete sense
   - Adds controlled numerical diffusion
   - Good for solutions with shocks or steep gradients

2. **Upwind scheme** (alternative):
   - Preserves monotonicity
   - Lower diffusion than Lax-Friedrichs
   - Sharper gradient resolution

### Dimensional Splitting

The multi-dimensional problem is solved via dimensional splitting:
1. Age evolution: $\partial_a u = -\delta u$
2. $x_1$ evolution: $\partial_t u + \partial_{x_1}(\xi_x u) = 0$
3. $x_2$ evolution: $\partial_t u + \partial_{x_2}(\xi_y u) = 0$

### Time Stepping

Adaptive time-stepping enforces the CFL condition:
$$\Delta t \leq \text{CFL} \cdot \frac{\min(\Delta x_1, \Delta x_2, \Delta a)}{\max(|\xi|, 1)}$$

## Performance

### Computational Complexity
- **Per time step**: O(N² log N) for convolutions + O(N³) for updates
- **Parallel speedup**: Convolutions computed in parallel
- **Memory**: ~400-500 MB for typical configurations (100×100×50 grid)

### Optimization Features
- Multi-threaded convolution operations
- Efficient array views (no data copying for shifted indices)
- Overlap-add convolution method (scipy.signal.oaconvolve)
- Parallel plot generation

## Examples

### Running the Aveiro Scenario

```bash
# Simulate with Lax-Friedrichs scheme
python main_xylella.py ./Python/simulations/Aveiro_LF -s -pn 4

# Generate plots
python main_xylella.py ./Python/simulations/Aveiro_LF -p -pn 4 -cb

# Create movies (requires mencoder)
python main_xylella.py ./Python/simulations/Aveiro_LF -m
```

### Running a Custom Simulation

```bash
# Create simulation directory
mkdir my_scenario
cd my_scenario
cp ../simulations/Aveiro_LF/parameters.py .
# Edit parameters.py as needed

# Run simulation
python ../main_xylella.py . -s -pn 4
```

## Requirements and Dependencies

- **Python**: 3.8+
- **NumPy**: For numerical operations
- **SciPy**: For FFT-based convolution (signal.oaconvolve)
- **Matplotlib**: For plotting and visualization
- **mencoder**: (Optional) For creating AVI movies

Install Python dependencies:
```bash
pip install numpy scipy matplotlib
```

Install mencoder (Ubuntu/Debian):
```bash
sudo apt-get install mencoder
```

## Logging

The program provides detailed logging to console and log files:

### Simulation Log (log_Simulation.txt)
- Start/finish timestamps
- Initial masses and L∞ norms
- Final masses and L∞ norms
- Checkpoint messages
- Total execution time

### Plot Log (log_Plots.txt)
- Plot generation progress
- Processing time per plot
- Total visualization time

### Movie Log (log_Movie.txt)
- Movie creation status
- Mencoder output

## Troubleshooting

### Low Memory Warning
If your grid is too large:
- Reduce `N_x1`, `N_x2`, or `N_a` in parameters.py
- Use fewer snapshots (reduce `N_pictures`)

### Slow Simulation
- Reduce grid resolution
- Decrease `T` (final time)
- Use fewer processors or disable multiprocessing

### Movie Generation Fails
- Ensure mencoder is installed: `mencoder --version`
- Check that plot files exist in Simulation directory

### Numerical Instability
- Decrease CFL parameter (default 0.5)
- Try Lax-Friedrichs scheme for more stability
- Reduce kernel support to avoid spurious interactions

## Documentation

Comprehensive technical documentation is available in:
- **`Xylella_Documentation.tex`**: Complete LaTeX documentation with mathematical details, algorithm descriptions, and usage examples

Generate PDF:
```bash
pdflatex Xylella_Documentation.tex
```

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Contact

**Author**: Mauro Garavello

For questions or issues, please use the GitHub issue tracker.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{garavello2026xylella,
  title={Xylella: A Simulation Framework for Age-Structured Spatial Population Dynamics},
  author={Garavello, Mauro},
  year={2026},
  url={https://github.com/maurogaravello/Xylella}
}
```

## References

- **Finite Volume Methods**: LeVeque, R. J. (2002). Finite Volume Methods for Hyperbolic Problems
- **Age-Structured Models**: Metz, J. A., & Diekmann, O. (1986). The Dynamics of Physiologically Structured Populations
- **Nonlocal Models**: Bates, P. W. (1989). On Some Nonlocal Evolution Equations Arising in Materials Science

---

**Last Updated**: June 29, 2026  
**Version**: 1.0
