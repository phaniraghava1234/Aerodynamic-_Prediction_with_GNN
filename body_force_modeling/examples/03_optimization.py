"""
Example 3: Propeller Optimization with Adjoint Method

This example demonstrates gradient-based optimization of propeller
design using adjoint sensitivity analysis. The goal is to maximize
efficiency while satisfying thrust and geometric constraints.

Reference: 
    - Martins, J. R. R. A., & Ning, A. (2021). Engineering Design Optimization.
    - Jameson, A. (1988). Aerodynamic Design via Control Theory.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.propeller_model import PropellerActuatorDisk
from src.optimizer import PropellerOptimizer
from src.visualization import plot_optimization_history, plot_propeller_performance
import numpy as np

def main():
    """Run propeller optimization example"""
    
    print("=" * 70)
    print("Example 3: Propeller Optimization with Adjoint Method")
    print("=" * 70)
    print()
    
    # Define optimization problem
    print("Step 1: Define optimization problem...")
    print()
    print("  Objective:    Maximize propeller efficiency")
    print("  Constraints:")
    print("    - Minimum thrust: 5000 N")
    print("    - Maximum diameter: 3.5 m")
    print("    - Maximum tip speed: 220 m/s (Mach 0.65)")
    print("    - Maximum RPM: 3000")
    print()
    
    # Create optimizer
    optimizer = PropellerOptimizer(
        objective='maximize_efficiency',
        constraints={
            'min_thrust': 5000.0,      # N
            'max_tip_speed': 220.0,    # m/s (keep subsonic)
            'max_diameter': 3.5,       # m
        },
        bounds={
            'min_diameter': 2.0,
            'max_diameter': 3.5,
            'min_rpm': 1000.0,
            'max_rpm': 3000.0,
            'min_pitch': 15.0,
            'max_pitch': 40.0
        }
    )
    
    # Initial design
    print("Step 2: Set initial design...")
    initial_design = {
        'diameter': 2.5,       # m
        'rpm': 2000.0,         # RPM
        'pitch_angle': 25.0    # degrees
    }
    
    print(f"  Initial diameter: {initial_design['diameter']:.2f} m")
    print(f"  Initial RPM: {initial_design['rpm']:.0f}")
    print(f"  Initial pitch: {initial_design['pitch_angle']:.1f}°")
    print()
    
    # Create propeller
    propeller = PropellerActuatorDisk(
        diameter=initial_design['diameter'],
        rpm=initial_design['rpm'],
        pitch_angle=initial_design['pitch_angle'],
        num_blades=4
    )
    
    # Evaluate initial performance
    flight_condition = {
        'V_inf': 50.0,  # m/s (cruise speed)
        'altitude': 0.0  # m
    }
    
    propeller.set_flight_condition(flight_condition['V_inf'])
    T_init, P_init, eta_init = propeller.compute_performance()
    
    print("  Initial Performance:")
    print(f"    Thrust:     {T_init:.1f} N")
    print(f"    Power:      {P_init/1000:.2f} kW")
    print(f"    Efficiency: {eta_init*100:.1f}%")
    print()
    
    # Run optimization
    print("Step 3: Run optimization with adjoint method...")
    print()
    
    result = optimizer.optimize(
        initial_design=initial_design,
        propeller=propeller,
        flight_condition=flight_condition,
        method='adjoint',  # Use adjoint for efficient gradients
        use_adjoint=True,
        maxiter=50
    )
    
    print()
    
    # Display results
    print("=" * 70)
    print("Optimization Results")
    print("=" * 70)
    print()
    
    print(f"Optimization Status: {'SUCCESS' if result['success'] else 'FAILED'}")
    print()
    
    print("Optimal Design:")
    print(f"  Diameter:    {result['diameter']:.3f} m  (change: {result['diameter']-initial_design['diameter']:+.3f})")
    print(f"  RPM:         {result['rpm']:.0f}  (change: {result['rpm']-initial_design['rpm']:+.0f})")
    print(f"  Pitch:       {result['pitch_angle']:.2f}°  (change: {result['pitch_angle']-initial_design['pitch_angle']:+.2f})")
    print()
    
    print("Optimal Performance:")
    print(f"  Thrust:      {result['thrust']:.1f} N  (change: {result['thrust']-T_init:+.1f})")
    print(f"  Power:       {result['power']/1000:.2f} kW  (change: {(result['power']-P_init)/1000:+.2f})")
    print(f"  Efficiency:  {result['efficiency']*100:.1f}%  (change: {(result['efficiency']-eta_init)*100:+.1f})")
    print()
    
    # Performance improvement
    eta_improvement = ((result['efficiency'] - eta_init) / eta_init) * 100
    power_change = ((result['power'] - P_init) / P_init) * 100
    
    print("Performance Improvement:")
    print(f"  Efficiency increase: {eta_improvement:+.1f}%")
    print(f"  Power change:        {power_change:+.1f}%")
    print()
    
    # Verify constraints
    print("Constraint Verification:")
    tip_speed = propeller.get_tip_speed()
    print(f"  Tip speed: {tip_speed:.1f} m/s (limit: 220 m/s) {'✓' if tip_speed <= 220 else '✗'}")
    print(f"  Thrust:    {result['thrust']:.1f} N (min: 5000 N) {'✓' if result['thrust'] >= 5000 else '✗'}")
    print(f"  Diameter:  {result['diameter']:.2f} m (max: 3.5 m) {'✓' if result['diameter'] <= 3.5 else '✗'}")
    print()
    
    # Generate visualizations
    print("Step 4: Generate visualization...")
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    os.makedirs(results_dir, exist_ok=True)
    
    # Optimization history
    plot_optimization_history(
        result['history'],
        save_path=os.path.join(results_dir, 'ex03_optimization_history.png')
    )
    print("  ✓ Saved: ex03_optimization_history.png")
    
    # Performance curves
    plot_propeller_performance(
        propeller,
        V_range=(30.0, 80.0),
        save_path=os.path.join(results_dir, 'ex03_optimal_performance.png')
    )
    print("  ✓ Saved: ex03_optimal_performance.png")
    
    print()
    print("=" * 70)
    print("Example 3 completed successfully!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. Adjoint method enables efficient gradient computation")
    print("  2. Optimization improves efficiency while satisfying constraints")
    print("  3. Design variables are coupled (D, RPM, pitch affect each other)")
    print("  4. Constraint handling ensures feasible and safe designs")
    print(f"\nResults saved to: {results_dir}")
    print()
    
    return result, propeller, optimizer


if __name__ == "__main__":
    main()
