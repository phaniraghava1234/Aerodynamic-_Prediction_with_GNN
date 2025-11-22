"""
Example 2: Flow Field Coupling

This example demonstrates coupling between the actuator disk model
and a simplified flow field solver. Shows how propeller body forces
affect the surrounding flow field.

Reference: Sørensen, J. N., & Shen, W. Z. (2002). Numerical Modeling of Wind Turbine Wakes.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.propeller_model import PropellerActuatorDisk
from src.flow_solver import FlowField2D
from src.visualization import plot_velocity_field, plot_wake_profile
import numpy as np

def main():
    """Run flow field coupling example"""
    
    print("=" * 70)
    print("Example 2: Flow Field Coupling with Body Forces")
    print("=" * 70)
    print()
    
    # Create propeller
    print("Step 1: Create propeller model...")
    propeller = PropellerActuatorDisk(
        diameter=2.5,
        hub_radius=0.3,
        rpm=2000,
        num_blades=4,
        pitch_angle=25.0
    )
    print(f"  {propeller}")
    print()
    
    # Create flow field
    print("Step 2: Initialize flow field domain...")
    flow = FlowField2D(
        x_range=(-2.0, 10.0),  # -2m upstream to 10m downstream
        r_range=(0.0, 5.0),     # 0 to 5m radial extent
        resolution=100          # 100x100 grid
    )
    
    V_inf = 50.0  # m/s
    flow.set_freestream(V_inf)
    
    print(f"  Domain: x ∈ {flow.x_range}, r ∈ {flow.r_range}")
    print(f"  Grid: {flow.nx} × {flow.nr} points")
    print(f"  Freestream: {V_inf} m/s")
    print()
    
    # Add actuator disk to flow field
    print("Step 3: Add actuator disk to flow field...")
    x_propeller = 0.0  # Propeller at x=0
    thickness = 0.1     # Disk thickness for force distribution
    
    flow.add_actuator_disk(
        propeller,
        x_location=x_propeller,
        thickness=thickness
    )
    print(f"  Disk location: x = {x_propeller} m")
    print(f"  Disk thickness: {thickness} m")
    print()
    
    # Solve flow field
    print("Step 4: Solve flow field with body forces...")
    flow.solve(method='momentum')
    print("  Flow field solved!")
    print()
    
    # Analyze results
    print("Step 5: Analyze flow field...")
    
    # Velocity at disk center
    u_disk, v_disk, w_disk = flow.get_velocity_at_point(x_propeller, propeller.radius/2)
    print(f"  Velocity at disk center:")
    print(f"    Axial:      {u_disk:.2f} m/s")
    print(f"    Radial:     {v_disk:.2f} m/s")
    print(f"    Tangential: {w_disk:.2f} m/s")
    print()
    
    # Wake velocity at various downstream locations
    print("  Wake centerline velocity:")
    print(f"  {'Location [m]':<20} {'Velocity [m/s]':<20} {'ΔV/V_inf [%]':<15}")
    print("-" * 55)
    
    for x_wake in [0.5, 1.0, 2.0, 5.0, 10.0]:
        u_wake, _, _ = flow.get_velocity_at_point(x_wake, 0.01)
        delta_v = ((u_wake - V_inf) / V_inf) * 100
        print(f"  x = {x_wake:<18.1f} {u_wake:<20.2f} {delta_v:<15.1f}")
    
    print()
    
    # Visualize results
    print("Step 6: Generate visualizations...")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot axial velocity field
    plot_velocity_field(
        flow,
        component='axial',
        save_path=os.path.join(results_dir, 'ex02_velocity_field_axial.png'),
        show_propeller=True,
        show_streamlines=True
    )
    print("  ✓ Saved: ex02_velocity_field_axial.png")
    
    # Plot velocity magnitude
    plot_velocity_field(
        flow,
        component='magnitude',
        save_path=os.path.join(results_dir, 'ex02_velocity_magnitude.png'),
        show_propeller=True,
        show_streamlines=False
    )
    print("  ✓ Saved: ex02_velocity_magnitude.png")
    
    # Plot pressure coefficient
    plot_velocity_field(
        flow,
        component='pressure',
        save_path=os.path.join(results_dir, 'ex02_pressure_coefficient.png'),
        show_propeller=True,
        show_streamlines=False
    )
    print("  ✓ Saved: ex02_pressure_coefficient.png")
    
    # Plot wake profiles
    x_locations = [0.5, 1.0, 2.0, 5.0]
    plot_wake_profile(
        flow,
        x_locations=x_locations,
        save_path=os.path.join(results_dir, 'ex02_wake_profiles.png')
    )
    print("  ✓ Saved: ex02_wake_profiles.png")
    
    print()
    print("=" * 70)
    print("Example 2 completed successfully!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. Body forces modify the flow field around the propeller")
    print("  2. Wake velocity deficit persists downstream")
    print("  3. Swirl effects create tangential velocity components")
    print("  4. Results can be visualized for physical insight")
    print(f"\nResults saved to: {results_dir}")
    print()
    
    return flow, propeller


if __name__ == "__main__":
    main()
