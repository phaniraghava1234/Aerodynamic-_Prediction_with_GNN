"""
Example 4: Multi-Point Optimization

This example demonstrates robust propeller design optimization across
multiple flight conditions. This is essential for aircraft that operate
across a wide range of speeds (takeoff, climb, cruise).

Reference: Kenway, G. K. W., & Martins, J. R. R. A. (2014). 
           Multipoint High-Fidelity Aerostructural Optimization.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.propeller_model import PropellerActuatorDisk
from src.optimizer import PropellerOptimizer
from src.visualization import plot_comparison
import numpy as np

def main():
    """Run multi-point optimization example"""
    
    print("=" * 70)
    print("Example 4: Multi-Point Robust Optimization")
    print("=" * 70)
    print()
    
    # Define mission profile
    print("Step 1: Define mission profile...")
    print()
    
    flight_conditions = [
        {'V_inf': 35.0, 'altitude': 0.0, 'name': 'Takeoff'},
        {'V_inf': 45.0, 'altitude': 1000.0, 'name': 'Climb'},
        {'V_inf': 60.0, 'altitude': 3000.0, 'name': 'Cruise'},
        {'V_inf': 50.0, 'altitude': 2000.0, 'name': 'Descent'}
    ]
    
    print("  Mission Flight Conditions:")
    for i, fc in enumerate(flight_conditions):
        print(f"    {i+1}. {fc['name']:<10}: V={fc['V_inf']:.0f} m/s, h={fc['altitude']:.0f} m")
    
    print()
    
    # Assign weights based on time spent in each phase
    weights = [0.15, 0.25, 0.50, 0.10]  # Cruise is most important
    
    print("  Optimization Weights:")
    for i, (fc, w) in enumerate(zip(flight_conditions, weights)):
        print(f"    {fc['name']:<10}: {w*100:.0f}%")
    print()
    
    # Create optimizer
    print("Step 2: Setup multi-point optimizer...")
    optimizer = PropellerOptimizer(
        objective='maximize_efficiency',
        constraints={
            'min_thrust': 4500.0,     # Minimum thrust at all conditions
            'max_tip_speed': 220.0,
            'max_diameter': 3.2
        },
        bounds={
            'min_diameter': 2.2,
            'max_diameter': 3.2,
            'min_rpm': 1500.0,
            'max_rpm': 2500.0,
            'min_pitch': 20.0,
            'max_pitch': 35.0
        }
    )
    print("  ✓ Optimizer configured")
    print()
    
    # Initial design (typical cruise-optimized propeller)
    initial_design = {
        'diameter': 2.6,
        'rpm': 2000.0,
        'pitch_angle': 27.0
    }
    
    print("Step 3: Evaluate initial design at all conditions...")
    
    propeller = PropellerActuatorDisk(
        diameter=initial_design['diameter'],
        rpm=initial_design['rpm'],
        pitch_angle=initial_design['pitch_angle'],
        num_blades=4
    )
    
    print(f"\n  Initial Design: D={initial_design['diameter']}m, "
          f"RPM={initial_design['rpm']:.0f}, pitch={initial_design['pitch_angle']}°")
    print()
    print(f"  {'Condition':<12} {'Thrust [N]':<12} {'Power [kW]':<12} {'η [%]':<10}")
    print("  " + "-" * 50)
    
    initial_performance = []
    for fc in flight_conditions:
        propeller.set_flight_condition(fc['V_inf'], altitude=fc['altitude'])
        T, P, eta = propeller.compute_performance()
        initial_performance.append({'T': T, 'P': P, 'eta': eta})
        print(f"  {fc['name']:<12} {T:<12.1f} {P/1000:<12.2f} {eta*100:<10.1f}")
    
    print()
    
    # Run multi-point optimization
    print("Step 4: Run multi-point optimization...")
    print("  (This optimizes for weighted average performance)")
    print()
    
    result = optimizer.multi_point_optimization(
        initial_design=initial_design,
        propeller=propeller,
        flight_conditions=flight_conditions,
        weights=weights
    )
    
    print()
    
    # Display results
    print("=" * 70)
    print("Multi-Point Optimization Results")
    print("=" * 70)
    print()
    
    print(f"Optimization Status: {'SUCCESS' if result['success'] else 'FAILED'}")
    print()
    
    print("Optimal Design:")
    print(f"  Diameter:  {result['diameter']:.3f} m  "
          f"(Δ = {result['diameter']-initial_design['diameter']:+.3f})")
    print(f"  RPM:       {result['rpm']:.0f}  "
          f"(Δ = {result['rpm']-initial_design['rpm']:+.0f})")
    print(f"  Pitch:     {result['pitch_angle']:.2f}°  "
          f"(Δ = {result['pitch_angle']-initial_design['pitch_angle']:+.2f})")
    print()
    
    # Compare performance at each condition
    print("Performance Comparison:")
    print()
    print(f"  {'Condition':<12} {'Weight':<8} {'η_init [%]':<12} "
          f"{'η_opt [%]':<12} {'Δη [%]':<10}")
    print("  " + "-" * 60)
    
    weighted_improvement = 0.0
    
    for i, (fc, w) in enumerate(zip(flight_conditions, weights)):
        eta_init = initial_performance[i]['eta'] * 100
        eta_opt = result['results_per_condition'][i]['efficiency'] * 100
        delta_eta = eta_opt - eta_init
        weighted_improvement += w * delta_eta
        
        print(f"  {fc['name']:<12} {w*100:<8.0f} {eta_init:<12.1f} "
              f"{eta_opt:<12.1f} {delta_eta:<10.1f}")
    
    print("  " + "-" * 60)
    print(f"  {'Weighted Avg':<12} {'100':<8} {'':<12} {'':<12} "
          f"{weighted_improvement:<10.1f}")
    print()
    
    # Thrust requirement check
    print("Constraint Verification:")
    print(f"  {'Condition':<12} {'Thrust [N]':<12} {'Min Required':<15} {'Status':<10}")
    print("  " + "-" * 50)
    
    for i, fc in enumerate(flight_conditions):
        T = result['results_per_condition'][i]['thrust']
        min_req = optimizer.constraints['min_thrust']
        status = '✓ PASS' if T >= min_req else '✗ FAIL'
        print(f"  {fc['name']:<12} {T:<12.1f} {min_req:<15.1f} {status:<10}")
    
    print()
    
    # Generate comparison visualization
    print("Step 5: Generate comparison plots...")
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create propellers for comparison
    prop_initial = PropellerActuatorDisk(
        diameter=initial_design['diameter'],
        rpm=initial_design['rpm'],
        pitch_angle=initial_design['pitch_angle']
    )
    
    prop_optimal = PropellerActuatorDisk(
        diameter=result['diameter'],
        rpm=result['rpm'],
        pitch_angle=result['pitch_angle']
    )
    
    # Compare at cruise condition
    plot_comparison(
        propellers=[prop_initial, prop_optimal],
        labels=['Initial Design', 'Optimized Design'],
        V_inf=60.0,  # Cruise
        save_path=os.path.join(results_dir, 'ex04_design_comparison.png')
    )
    print("  ✓ Saved: ex04_design_comparison.png")
    
    print()
    print("=" * 70)
    print("Example 4 completed successfully!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  1. Multi-point optimization ensures robust design across mission")
    print("  2. Weighting allows prioritization of critical flight phases")
    print("  3. Trade-offs exist between performance at different conditions")
    print("  4. Constraints must be satisfied at ALL operating points")
    print()
    print("Design Insights:")
    
    # Analyze design changes
    if result['diameter'] > initial_design['diameter']:
        print("  • Larger diameter improves efficiency but reduces RPM")
    if result['rpm'] < initial_design['rpm']:
        print("  • Lower RPM reduces tip speed (noise/structural benefits)")
    if result['pitch_angle'] != initial_design['pitch_angle']:
        print(f"  • Pitch adjusted to balance performance across mission")
    
    print(f"\nResults saved to: {results_dir}")
    print()
    
    return result, propeller, optimizer


if __name__ == "__main__":
    main()
