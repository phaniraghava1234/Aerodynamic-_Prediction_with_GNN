"""
Example 1: Basic Actuator Disk Model

This example demonstrates the fundamental actuator disk model for
propeller performance prediction. It shows how to:
1. Create a propeller model
2. Set operating conditions
3. Compute performance metrics

Author: PhD Application Project
Reference: Conway, J. T. (1998). Analytical Solutions for the Actuator Disk.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.propeller_model import PropellerActuatorDisk
import numpy as np

def main():
    """Run basic actuator disk example"""
    
    print("=" * 70)
    print("Example 1: Basic Actuator Disk Model")
    print("=" * 70)
    print()
    
    # Create propeller instance
    print("Creating propeller model...")
    propeller = PropellerActuatorDisk(
        diameter=2.5,        # 2.5 meter diameter
        hub_radius=0.3,      # 0.3 meter hub radius
        rpm=2000,            # 2000 RPM
        num_blades=4,        # 4 blades
        pitch_angle=25.0     # 25 degree pitch at 75% radius
    )
    
    print(f"  {propeller}")
    print()
    
    # Set flight condition
    print("Setting flight condition...")
    V_inf = 50.0  # m/s (approximately 97 knots)
    altitude = 0.0  # Sea level
    
    propeller.set_flight_condition(V_inf, altitude)
    print(f"  Freestream velocity: {V_inf:.1f} m/s")
    print(f"  Altitude: {altitude:.0f} m")
    print(f"  Air density: {propeller.rho:.3f} kg/m³")
    print()
    
    # Compute advance ratio
    J = propeller.get_advance_ratio()
    print(f"Advance Ratio:")
    print(f"  J = V∞/(nD) = {J:.3f}")
    print()
    
    # Compute tip speed
    tip_speed = propeller.get_tip_speed()
    print(f"Tip Speed:")
    print(f"  V_tip = {tip_speed:.1f} m/s")
    print(f"  Tip Mach number: {tip_speed/340:.3f}")
    print()
    
    # Compute performance
    print("Computing propeller performance...")
    thrust, power, efficiency = propeller.compute_performance()
    
    print()
    print("Performance Results:")
    print("-" * 50)
    print(f"  Thrust:          {thrust:.1f} N")
    print(f"  Power:           {power/1000:.2f} kW")
    print(f"  Efficiency:      {efficiency*100:.1f}%")
    print(f"  Thrust Coeff:    {propeller.C_T:.4f}")
    print(f"  Power Coeff:     {propeller.C_P:.4f}")
    print()
    
    # Compute radial loading distribution
    print("Computing radial loading distribution...")
    thrust_dist, torque_dist = propeller.compute_radial_loading()
    
    print(f"  Radial stations: {len(propeller.r_stations)}")
    print(f"  Max thrust/area: {np.max(thrust_dist):.1f} N/m²")
    print(f"  Max torque/area: {np.max(torque_dist):.2f} N·m/m²")
    print()
    
    # Parametric study: vary freestream velocity
    print("Parametric Study: Performance vs. Velocity")
    print("-" * 50)
    print(f"{'V_inf [m/s]':<15} {'Thrust [N]':<15} {'Power [kW]':<15} {'η [%]':<10}")
    print("-" * 50)
    
    for V in [30, 40, 50, 60, 70, 80]:
        propeller.set_flight_condition(V)
        T, P, eta = propeller.compute_performance()
        print(f"{V:<15.1f} {T:<15.1f} {P/1000:<15.2f} {eta*100:<10.1f}")
    
    print()
    print("=" * 70)
    print("Example 1 completed successfully!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. Actuator disk provides rapid performance estimates")
    print("  2. Advance ratio J characterizes operating condition")
    print("  3. Efficiency peaks at specific advance ratio")
    print("  4. Radial loading distribution informs body force model")
    print()
    
    return propeller


if __name__ == "__main__":
    main()
