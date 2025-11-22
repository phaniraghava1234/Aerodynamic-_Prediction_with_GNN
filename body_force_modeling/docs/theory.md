# Theoretical Background: Body Force Modeling for Propellers

## Introduction

Body force models provide a computationally efficient method for simulating propeller effects in CFD by replacing the actual blade geometry with distributed momentum and energy sources. This approach is essential for:

- Early-stage design optimization
- Multi-disciplinary optimization (MDO)
- Propeller-airframe integration studies
- Distributed electric propulsion systems

## Momentum Theory Foundation

### Classic Actuator Disk Theory

The actuator disk represents the propeller as an infinitesimally thin disk that adds momentum to the flow. Key assumptions:

1. **Inviscid flow** (viscous effects ignored in far field)
2. **Incompressible** (Mach number < 0.3)
3. **Steady flow** (time-averaged for rotating blades)
4. **Axisymmetric** (azimuthally averaged)

### Conservation Equations

The flow through an actuator disk satisfies:

**Mass Conservation:**
```
∇ · (ρV) = 0
```

**Momentum Conservation with Body Forces:**
```
ρ(V · ∇)V = -∇p + f_body
```

where `f_body` is the body force per unit volume representing the propeller.

### Axial Momentum Theory

For an ideal actuator disk (no rotation), the thrust is:

```
T = ṁ(V_e - V_∞) = ρAV_disk(V_e - V_∞)
```

where:
- `T`: Thrust force
- `ṁ`: Mass flow rate through disk
- `V_∞`: Freestream velocity
- `V_disk`: Velocity at the disk
- `V_e`: Exit velocity in far wake
- `A`: Disk area

From momentum conservation:
```
V_disk = V_∞ + v_i
V_e = V_∞ + 2v_i
```

where `v_i` is the induced velocity.

### Induced Velocity

The induced velocity at the disk is:

```
v_i = √(T/(2ρA) + (V_∞/2)²) - V_∞/2
```

For hover (V_∞ = 0):
```
v_i = √(T/(2ρA))
```

## Body Force Distribution

### Radial Distribution

Real propellers have non-uniform loading. The radial thrust distribution can be modeled as:

```
dT/dr = f(r) · T_total / ∫f(r)dr
```

Common distribution functions:
- **Constant**: `f(r) = 1` (simple, inaccurate)
- **Linear**: `f(r) = r/R` (basic approximation)
- **Optimal (Betz)**: `f(r) ∝ 1/(1-(r/R)²)` (minimum induced loss)
- **BEMT-based**: From blade element momentum theory

### Tangential Forces (Swirl)

Propeller rotation induces tangential velocity (swirl):

```
V_θ(r) = Q/(2πr²ρV_disk)
```

where `Q` is the torque.

The tangential body force:

```
f_θ = ρV_disk V_θ/Δx
```

### Prandtl Tip Loss Correction

Real propellers have finite blades. Prandtl's correction factor:

```
F = (2/π)arccos(exp(-f_tip))

f_tip = (B/2)·(R-r)/(r·sin(φ))
```

where:
- `B`: Number of blades
- `φ`: Inflow angle
- `r`: Radial position
- `R`: Propeller radius

## Non-Dimensional Parameters

### Advance Ratio

```
J = V_∞/(nD)
```

where:
- `n`: Rotational speed [rev/s]
- `D`: Diameter

Typical range: 0.3 < J < 1.2 for aircraft propellers

### Thrust Coefficient

```
C_T = T/(ρn²D⁴)
```

### Power Coefficient

```
C_P = P/(ρn³D⁵)
```

### Efficiency

```
η = (T·V_∞)/P = J·(C_T/C_P)
```

Maximum efficiency typically occurs at J ≈ 0.7-0.9 for well-designed propellers.

## CFD Integration

### Source Term Implementation

In a CFD solver, body forces appear as source terms:

**Momentum Equation:**
```
∂(ρu_i)/∂t + ∂(ρu_i u_j)/∂x_j = -∂p/∂x_i + S_i^propeller
```

where `S_i^propeller` is the propeller body force in direction i.

### Spatial Distribution

The body force is distributed using:

1. **Gaussian distribution** in axial direction:
   ```
   g(x) = (1/(σ√(2π)))exp(-(x-x_disk)²/(2σ²))
   ```

2. **Interpolation** in radial direction from blade element data

### Numerical Considerations

- Disk thickness typically 0.1-0.2 chord lengths
- Grid resolution: 5-10 cells across disk
- Time step (unsteady): Δt < 0.01/n for resolution of rotation

## Comparison with Other Methods

| Method | Fidelity | Cost | Use Case |
|--------|----------|------|----------|
| Momentum Theory | Low | Very Low | Conceptual design |
| **Actuator Disk** | **Medium** | **Low** | **Design optimization** |
| Actuator Line | High | Medium | Detailed analysis |
| Blade Element | High | Medium | Design iterations |
| Blade-Resolved | Very High | Very High | Validation |

## Advanced Topics

### Unsteady Effects

For time-accurate simulations:

```
f_body(t) = f_steady + Σ f_n cos(nBΩt + φ_n)
```

where B is blade count and Ω is rotation rate.

### Compressibility Corrections

For higher Mach numbers (0.3 < M < 0.7):

```
C_T,comp = C_T/(√(1-M²))
```

### Ducted Propellers

Duct effects modify the momentum equation:

```
T_total = T_propeller + T_duct_thrust
```

## References

1. **Glauert, H. (1935).** Airplane Propellers. In *Aerodynamic Theory*, Vol. IV, Division L. Springer.
   - Classic momentum theory derivation

2. **Conway, J. T. (1998).** Analytical Solutions for the Actuator Disk with Variable Radial Distribution of Load. *Journal of Fluid Mechanics*, 378, 185-211.
   - Exact solutions for non-uniform loading

3. **Drela, M., & Youngren, H. (2008).** XROTOR Theory Document. MIT.
   - Practical implementation details

4. **Hansen, M. O. L. (2015).** *Aerodynamics of Wind Turbines*, 3rd Edition. Earthscan.
   - Comprehensive treatment applicable to propellers

5. **Leishman, J. G. (2006).** *Principles of Helicopter Aerodynamics*. Cambridge University Press.
   - Chapter 3: Momentum theory and induced velocity
