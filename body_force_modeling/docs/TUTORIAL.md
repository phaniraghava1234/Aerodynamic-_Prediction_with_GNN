# Step-by-Step Tutorial: Body Force Modeling for Propellers

This tutorial guides you through implementing and using body force models for propeller aerodynamic analysis, from basic concepts to advanced optimization.

## Prerequisites

- Python 3.8 or higher
- Basic understanding of aerodynamics
- Familiarity with NumPy and SciPy

## Installation

```bash
# Clone the repository
git clone https://github.com/phaniraghava1234/Aerodynamic-_Prediction_with_GNN.git
cd Aerodynamic-_Prediction_with_GNN/body_force_modeling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Tutorial Steps

### Step 1: Understanding Actuator Disk Theory (30 minutes)

**Objective**: Learn the fundamentals of actuator disk modeling

**Reading**:
1. Read `docs/theory.md` - Focus on:
   - Momentum theory foundation
   - Induced velocity concept
   - Non-dimensional parameters (J, C_T, C_P, η)

**Key Concepts**:
- Propeller represented as infinitesimally thin disk
- Adds momentum to flow via distributed forces
- Trade-off: computational efficiency vs. blade geometry detail

**Exercise**:
```python
# Calculate advance ratio for different conditions
V_inf = 50  # m/s
n = 2000 / 60  # rev/s
D = 2.5  # m

J = V_inf / (n * D)
print(f"Advance ratio: {J:.3f}")

# Typical range: 0.3 < J < 1.2 for aircraft
```

---

### Step 2: Basic Propeller Model (45 minutes)

**Objective**: Create and analyze a simple propeller model

**Run Example**:
```bash
python examples/01_basic_actuator_disk.py
```

**What's Happening**:
1. Creates `PropellerActuatorDisk` instance
2. Sets flight condition (V_inf, altitude, density)
3. Computes performance (thrust, power, efficiency)
4. Analyzes radial loading distribution

**Try Yourself**:

Edit the example to explore parameter effects:

```python
from src.propeller_model import PropellerActuatorDisk

# Create baseline propeller
prop_baseline = PropellerActuatorDisk(
    diameter=2.5,
    rpm=2000,
    pitch_angle=25.0
)

# Create high-speed variant
prop_highspeed = PropellerActuatorDisk(
    diameter=2.0,      # Smaller diameter
    rpm=2800,          # Higher RPM
    pitch_angle=30.0   # Higher pitch
)

# Compare at cruise
for prop, label in [(prop_baseline, "Baseline"), 
                     (prop_highspeed, "High-Speed")]:
    prop.set_flight_condition(V_inf=60.0)
    T, P, eta = prop.compute_performance()
    print(f"{label}: T={T:.0f}N, η={eta*100:.1f}%")
```

**Expected Results**:
- Thrust ∝ n² × D⁴
- Power ∝ n³ × D⁵
- Efficiency peaks at specific advance ratio

---

### Step 3: Flow Field Integration (1 hour)

**Objective**: Couple propeller with flow solver

**Run Example**:
```bash
python examples/02_flow_field_coupling.py
```

**What's Happening**:
1. Creates 2D axisymmetric flow domain
2. Adds propeller as body force source
3. Solves momentum equations
4. Visualizes velocity and pressure fields

**Key Outputs**:
- `ex02_velocity_field_axial.png` - Axial velocity contours
- `ex02_pressure_coefficient.png` - Pressure distribution
- `ex02_wake_profiles.png` - Downstream wake development

**Analyze the Figures**:
- Velocity increases through propeller disk
- Wake persists far downstream
- Pressure jump across disk
- Swirl effects in tangential direction

**Try Yourself**:

Modify the flow domain to study wake development:

```python
from src.flow_solver import FlowField2D
from src.propeller_model import PropellerActuatorDisk

# Create extended domain for wake analysis
flow = FlowField2D(
    x_range=(-2.0, 20.0),  # Longer downstream
    r_range=(0.0, 8.0),    # Wider radially
    resolution=150         # Higher resolution
)

prop = PropellerActuatorDisk(diameter=2.5, rpm=2000)
flow.set_freestream(50.0)
flow.add_actuator_disk(prop, x_location=0.0)
flow.solve()

# Extract wake profiles at multiple locations
for x_loc in [1, 2, 5, 10, 15, 20]:
    r, u = flow.get_wake_profile(x_loc)
    # Plot or analyze...
```

---

### Step 4: Performance Analysis (45 minutes)

**Objective**: Understand propeller characteristics

**Create Script**:

```python
from src.propeller_model import PropellerActuatorDisk
from src.visualization import plot_propeller_performance
import numpy as np

# Fixed geometry, vary speed
propeller = PropellerActuatorDisk(
    diameter=2.5,
    rpm=2000,
    num_blades=4
)

# Generate performance map
plot_propeller_performance(
    propeller,
    V_range=(20.0, 100.0),
    save_path='performance_curves.png'
)
```

**Analysis Questions**:
1. At what velocity does efficiency peak?
2. How does thrust vary with speed?
3. What's the relationship between J and η?

**Expected Insights**:
- η increases with J up to optimal point
- Thrust decreases with increasing V_inf (for fixed RPM)
- Power relatively constant (depends on J)

---

### Step 5: Basic Optimization (1 hour)

**Objective**: Optimize propeller design for efficiency

**Run Example**:
```bash
python examples/03_optimization.py
```

**What's Happening**:
1. Defines optimization problem (objective + constraints)
2. Uses gradient-based optimizer (SLSQP)
3. Employs adjoint method for efficient gradients
4. Finds optimal diameter, RPM, and pitch

**Key Results**:
- Efficiency improvement ~20-30%
- Constraint satisfaction verification
- Optimization convergence history

**Try Yourself**:

Optimize for different objectives:

```python
from src.optimizer import PropellerOptimizer
from src.propeller_model import PropellerActuatorDisk

# Optimize for minimum power (fuel economy)
optimizer_economy = PropellerOptimizer(
    objective='minimize_power',
    constraints={
        'min_thrust': 5000.0,
        'max_diameter': 3.5
    }
)

prop = PropellerActuatorDisk()
result = optimizer_economy.optimize(
    initial_design={'diameter': 2.5, 'rpm': 2000, 'pitch_angle': 25},
    propeller=prop,
    flight_condition={'V_inf': 50.0},
    method='adjoint'
)

print(f"Optimal power: {result['power']/1000:.1f} kW")
print(f"At efficiency: {result['efficiency']*100:.1f}%")
```

---

### Step 6: Advanced - Adjoint Sensitivity (1.5 hours)

**Objective**: Understand adjoint method for gradients

**Theory**:

The adjoint method computes ∂J/∂x for all design variables x in approximately the cost of 2 function evaluations, regardless of the number of variables.

**Standard Finite Difference**:
- Cost: (N+1) function evaluations for N variables
- Example: N=100 → 101 evaluations

**Adjoint Method**:
- Cost: ~2 function evaluations
- Example: N=100 → 2 evaluations
- **Speedup: 50x for N=100**

**Implementation**:

```python
from src.optimizer import PropellerOptimizer

optimizer = PropellerOptimizer(objective='maximize_efficiency')

# Compare methods
import time

# Method 1: Finite differences (slower)
t0 = time.time()
result_fd = optimizer.optimize(
    initial_design=x0,
    propeller=prop,
    flight_condition=fc,
    method='SLSQP',
    use_adjoint=False  # Finite differences
)
time_fd = time.time() - t0

# Method 2: Adjoint (faster)
t0 = time.time()
result_adj = optimizer.optimize(
    initial_design=x0,
    propeller=prop,
    flight_condition=fc,
    method='adjoint',
    use_adjoint=True
)
time_adj = time.time() - t0

print(f"Finite Difference: {time_fd:.2f}s")
print(f"Adjoint Method:    {time_adj:.2f}s")
print(f"Speedup:           {time_fd/time_adj:.1f}x")
```

---

### Step 7: Multi-Point Optimization (1 hour)

**Objective**: Robust design across multiple flight conditions

**Concept**: Optimize for performance at cruise, climb, and takeoff simultaneously

**Implementation**:

```python
from src.optimizer import PropellerOptimizer
from src.propeller_model import PropellerActuatorDisk

optimizer = PropellerOptimizer(objective='maximize_efficiency')
prop = PropellerActuatorDisk()

# Define multiple flight conditions
flight_conditions = [
    {'V_inf': 40.0, 'altitude': 0.0},      # Takeoff
    {'V_inf': 50.0, 'altitude': 1000.0},   # Climb
    {'V_inf': 70.0, 'altitude': 3000.0}    # Cruise
]

# Equal weighting
weights = [0.33, 0.34, 0.33]

result = optimizer.multi_point_optimization(
    initial_design={'diameter': 2.5, 'rpm': 2000, 'pitch_angle': 25},
    propeller=prop,
    flight_conditions=flight_conditions,
    weights=weights
)

# Analyze performance at each condition
for i, fc in enumerate(flight_conditions):
    perf = result['results_per_condition'][i]
    print(f"Condition {i+1}: η = {perf['efficiency']*100:.1f}%")
```

---

### Step 8: Distributed Propulsion (1.5 hours)

**Objective**: Model multiple propellers (distributed electric propulsion)

**Application**: Urban Air Mobility, X-57 Maxwell

**Implementation**:

```python
from src.propeller_model import PropellerArray, PropellerActuatorDisk
import numpy as np

# Create propeller array
array = PropellerArray()

# Wing-mounted distributed propulsion
# 6 propellers along wing span
y_positions = np.linspace(-3, 3, 6)  # ±3m span

for y_pos in y_positions:
    prop = PropellerActuatorDisk(
        diameter=1.2,      # Smaller props
        rpm=3000,          # Higher RPM
        num_blades=3
    )
    prop.set_flight_condition(V_inf=40.0)
    
    # Position along wing
    array.add_propeller(
        prop,
        position=(0.0, y_pos, 0.0),
        orientation=(1.0, 0.0, 0.0)
    )

# Compute total performance
T_total, P_total, eta_avg = array.compute_total_performance()

print(f"Array: {len(array)} propellers")
print(f"Total thrust: {T_total:.0f} N")
print(f"Total power:  {P_total/1000:.1f} kW")
print(f"Avg efficiency: {eta_avg*100:.1f}%")
```

---

## Summary and Best Practices

### When to Use Body Force Models:

✅ **Good for**:
- Design space exploration
- Optimization studies
- Multi-disciplinary analysis (aerodynamics + structures + controls)
- Propeller-airframe integration
- Distributed propulsion systems

❌ **Not suitable for**:
- Detailed blade design (use BEMT or blade-resolved CFD)
- Unsteady aerodynamic effects (blade passing, flutter)
- Off-design performance (stall, reverse thrust)

### Design Process Workflow:

1. **Conceptual Design** → Body force models (this toolkit)
2. **Preliminary Design** → BEMT + panel methods
3. **Detailed Design** → Blade-resolved RANS CFD
4. **Validation** → Wind tunnel / flight test

### Recommended Reading:

1. **Theory**: `docs/theory.md`
2. **Examples**: Run all examples 01-03
3. **References**: Check README.md for papers

### Next Steps:

- Implement your own propeller design problem
- Couple with airframe analysis
- Add noise constraints
- Explore aeroelastic effects

---

## Troubleshooting

**Issue**: Optimization not converging
- **Solution**: Adjust bounds, check constraints, try different initial guess

**Issue**: Flow solver unstable
- **Solution**: Reduce disk thickness, increase grid resolution

**Issue**: Unrealistic performance
- **Solution**: Verify input units, check advance ratio range

## Getting Help

- Check documentation in `docs/`
- Review example scripts in `examples/`
- Examine source code comments in `src/`

**Total Tutorial Time**: ~6-7 hours

This tutorial provides hands-on experience with all aspects of body force modeling for propeller design and optimization!
