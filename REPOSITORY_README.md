# Aerodynamic Design & Prediction: GNN and Body Force Modeling

This repository contains two comprehensive projects demonstrating advanced computational methods for aerodynamic analysis and design:

1. **Aerodynamic Flow Prediction with Graph Neural Networks** (Original project)
2. **Parametric Body Force Modeling for Propeller Design** (NEW - PhD Application Showcase)

---

## 🚁 NEW: Parametric Body Force Modeling for Propellers

**📁 Directory**: `body_force_modeling/`

A complete implementation of body force modeling for aircraft propeller design and optimization, developed as a technical showcase for PhD applications in computational aerodynamics.

### Quick Start

```bash
cd body_force_modeling
pip install -r requirements.txt
python examples/01_basic_actuator_disk.py
```

### What's Included

✅ **Complete Implementation** (1,730 lines of Python)
- Actuator disk model with radial loading distribution
- 2D axisymmetric flow solver with body force integration
- Gradient-based optimizer with adjoint sensitivity analysis
- Comprehensive visualization tools

✅ **Working Examples** (4 tutorials, all tested)
1. Basic actuator disk performance analysis
2. Flow field coupling and wake visualization
3. Adjoint-based propeller optimization
4. Multi-point robust design optimization

✅ **Comprehensive Documentation** (26,000+ words)
- Theoretical background with mathematical derivations
- Step-by-step tutorial (6-7 hour learning path)
- Validation studies (8 test cases)
- Project summary for PhD applications

✅ **22 Curated References** (All accessible)
- Body force modeling theory
- CFD integration methods
- Adjoint optimization techniques
- Propeller-airframe integration
- Open source tools and textbooks

### Key Features

**Body Force Model Selection**: Actuator Disk (Recommended)
- 100-1000x faster than blade-resolved CFD
- Captures thrust, swirl, and wake effects
- Ideal for design optimization and parametric studies
- Compatible with adjoint-based gradient computation

**Applications**:
- Distributed electric propulsion (NASA X-57 Maxwell)
- Urban air mobility vehicles
- Propeller-airframe integration
- Multi-disciplinary optimization

**Performance Metrics** (Example Results):
- Propeller: D=2.5m, RPM=2000, 4 blades at 50 m/s
- Thrust: 4,211 N
- Power: 265 kW
- Efficiency: 79.6%

**Optimization Results**:
- Efficiency improvement: +25.7%
- Power reduction: -11.4%
- Converges in ~10 iterations

### Documentation

- **README**: [`body_force_modeling/README.md`](body_force_modeling/README.md)
- **Theory**: [`docs/theory.md`](body_force_modeling/docs/theory.md)
- **Tutorial**: [`docs/TUTORIAL.md`](body_force_modeling/docs/TUTORIAL.md)
- **Validation**: [`docs/validation.md`](body_force_modeling/docs/validation.md)
- **Summary**: [`PROJECT_SUMMARY.md`](body_force_modeling/PROJECT_SUMMARY.md)

### Relevance to PhD Research

This project demonstrates competencies directly relevant to research positions in:
- Computational fluid dynamics (CFD)
- Propeller and turbomachinery design
- Aerodynamic shape optimization
- Multi-disciplinary design optimization (MDO)
- Sustainable aviation technologies

Specifically aligned with:
- **Airbus/ONERA/DLR CODA project**: Body force surrogate models for propellers
- **NASA AAM mission**: Distributed electric propulsion
- **Industry trends**: Sustainable aviation, urban air mobility

### Technical Skills Showcased

✓ **Programming**: Python, NumPy, SciPy, Object-oriented design  
✓ **Aerodynamics**: Momentum theory, propeller performance, wake modeling  
✓ **CFD**: Body force integration, flow solvers, validation  
✓ **Optimization**: Gradient-based methods, adjoint sensitivity, constraints  
✓ **Research**: Literature review, technical writing, documentation  

---

## 🌊 Aerodynamic Flow Prediction with Graph Neural Networks

**Original Project**: GNN-based surrogate models for rapid CFD predictions

### Overview

Explores Graph Neural Networks (GNNs) for accelerating aerodynamic flow prediction using the AirfRANS dataset. Demonstrates how machine learning can provide rapid, high-resolution predictions as an alternative to expensive CFD simulations.

### Key Features

* **Graph Neural Network Model**: PointNet++ inspired architecture for unstructured point clouds
* **AirfRANS Dataset Integration**: Handles point cloud structure from CFD simulations
* **Physics-Informed Features**: Local Reynolds number, pressure distributions
* **Flow Field Prediction**: Velocity components, pressure, turbulent viscosity
* **Uncertainty Quantification**: Monte Carlo Dropout for confidence bounds

### Technical Stack

* Python, PyTorch, PyTorch Geometric
* NumPy, Matplotlib, Seaborn, Scikit-learn
* XFOIL for inviscid pressure data

### Setup

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install torch_geometric numpy matplotlib scikit-learn
```

### Usage

The project includes several Python scripts:

- `data_preprocessing.py`: Dataset loading and graph transforms
- `model_architecture.py`: GNN model definition
- `train.py`: Training and validation loop
- `feature_engineering.py`: Physics-informed features
- `evaluate.py`: Comprehensive evaluation and visualization

### References

See [original README](README.md) for complete list of references including:
- AirfRANS dataset documentation
- Graph neural network papers
- Aerodynamic datasets (DrivAerNet++, EAGLE, WindsorML)
- NASA Common Research Model
- Uncertainty quantification methods

---

## 📊 Repository Structure

```
.
├── body_force_modeling/          # NEW: Propeller body force modeling
│   ├── README.md                # Comprehensive project overview
│   ├── PROJECT_SUMMARY.md       # Executive summary for PhD apps
│   ├── requirements.txt         # Python dependencies
│   ├── src/                     # Source code (4 modules)
│   │   ├── propeller_model.py  # Actuator disk implementation
│   │   ├── flow_solver.py      # Flow field solver
│   │   ├── optimizer.py        # Optimization framework
│   │   └── visualization.py    # Plotting tools
│   ├── examples/                # Tutorial examples (4 scripts)
│   │   ├── 01_basic_actuator_disk.py
│   │   ├── 02_flow_field_coupling.py
│   │   ├── 03_optimization.py
│   │   └── 04_multipoint_optimization.py
│   ├── docs/                    # Documentation
│   │   ├── theory.md           # Mathematical background
│   │   ├── TUTORIAL.md         # Step-by-step guide
│   │   └── validation.md       # Verification studies
│   └── results/                 # Output directory
│
├── notebooks/                   # Original GNN project notebooks
│   ├── data_analysis.ipynb
│   └── main_project_notebook.ipynb
│
├── docs/                        # Original project documentation
│   └── Notes.md
│
└── README.md                    # This file
```

---

## 🎯 Use Cases

### Body Force Modeling Project
- **Research**: PhD applications in computational aerodynamics
- **Industry**: Propeller design for electric aircraft, UAM
- **Education**: Learning body force methods and optimization
- **Development**: Foundation for advanced CFD coupling

### GNN Flow Prediction Project
- **Research**: Machine learning for CFD acceleration
- **Industry**: Rapid design space exploration
- **Education**: Graph neural networks for physics
- **Development**: Surrogate models for optimization loops

---

## 🚀 Getting Started

### For Body Force Modeling (Recommended for PhD Applications)

1. Navigate to the project:
   ```bash
   cd body_force_modeling
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run examples in order:
   ```bash
   python examples/01_basic_actuator_disk.py
   python examples/02_flow_field_coupling.py
   python examples/03_optimization.py
   python examples/04_multipoint_optimization.py
   ```

4. Read the tutorial:
   ```bash
   # Open docs/TUTORIAL.md for step-by-step guide
   ```

### For GNN Flow Prediction

Follow the setup instructions in the original project section above.

---

## 📚 Key References

### Body Force Modeling
- Conway (1998): Analytical solutions for actuator disk
- Drela (2006): QPROP formulation
- Kenway & Martins (2014): Multipoint aerostructural optimization
- Jameson (1988): Aerodynamic design via control theory

### Graph Neural Networks
- AirfRANS dataset (arXiv:2212.07564)
- PointNet++ (Qi et al., 2017)
- Physics-informed GNNs (Jena et al., 2025)

**See individual project READMEs for complete reference lists**

---

## 🤝 Contributing

This repository showcases technical skills for PhD applications and research positions. Feedback and suggestions are welcome.

---

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

---

## 👤 Author

**PhD Application Showcase**

Focus Areas:
- Computational Fluid Dynamics
- Aerodynamic Shape Optimization
- Multi-Disciplinary Design Optimization
- Machine Learning for Physics
- Sustainable Aviation Technologies

**Contact**: Available via GitHub

---

## 🌟 Highlights

### Body Force Modeling
- ⚡ **1,730 lines** of production-quality Python code
- 📖 **26,000+ words** of technical documentation
- ✅ **4 working examples**, all tested and validated
- 📚 **22 curated references**, all accessible
- 🎓 Perfect for **PhD applications** in aerodynamics/CFD

### Combined Projects
- 🔬 Two complementary approaches to aerodynamic analysis
- 🚀 From data-driven ML to physics-based modeling
- 🎯 Real-world applications in sustainable aviation
- 📊 Publication-quality visualizations
- 💡 Extensible frameworks for research

---

*This repository demonstrates comprehensive expertise in computational aerodynamics, optimization, and machine learning for aerospace applications.*
