"""
Visualization Tools for Propeller Body Force Modeling

Provides plotting and visualization capabilities for flow fields,
propeller performance, and optimization results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm
from typing import Optional, Tuple, List
import warnings

# Set publication-quality plot parameters
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def plot_velocity_field(
    flow_field,
    component: str = 'axial',
    save_path: Optional[str] = None,
    show_propeller: bool = True,
    show_streamlines: bool = True,
    figsize: Tuple[float, float] = (12, 6)
):
    """
    Plot flow field velocity contours
    
    Args:
        flow_field: FlowField2D instance
        component: Velocity component to plot ('axial', 'magnitude', 'pressure')
        save_path: Path to save figure (if None, displays interactively)
        show_propeller: Show propeller disk location
        show_streamlines: Overlay streamlines
        figsize: Figure size in inches
    """
    if flow_field.u is None:
        raise ValueError("Flow field not solved. Call flow_field.solve() first.")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Select field to plot
    if component == 'axial':
        field = flow_field.u
        label = 'Axial Velocity [m/s]'
        cmap = 'viridis'
    elif component == 'magnitude':
        field = flow_field.compute_velocity_magnitude()
        label = 'Velocity Magnitude [m/s]'
        cmap = 'plasma'
    elif component == 'pressure':
        field = flow_field.compute_pressure_coefficient()
        label = 'Pressure Coefficient $C_p$'
        cmap = 'RdBu_r'
    else:
        raise ValueError(f"Unknown component: {component}")
    
    # Contour plot
    levels = 20
    contour = ax.contourf(
        flow_field.X,
        flow_field.R,
        field,
        levels=levels,
        cmap=cmap
    )
    
    # Colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(label)
    
    # Streamlines
    if show_streamlines:
        try:
            stream_x, stream_r = flow_field.compute_streamlines(num_streamlines=15)
            for sx, sr in zip(stream_x, stream_r):
                ax.plot(sx, sr, 'k-', linewidth=0.5, alpha=0.5)
        except:
            warnings.warn("Could not compute streamlines")
    
    # Show propeller disk
    if show_propeller and len(flow_field.actuator_disks) > 0:
        for disk_info in flow_field.actuator_disks:
            prop = disk_info['propeller']
            x_loc = disk_info['x_location']
            
            # Draw disk
            ax.plot([x_loc, x_loc], [prop.hub_radius, prop.radius], 
                   'r-', linewidth=3, label='Propeller Disk')
            
            # Hub
            hub_circle = Circle((x_loc, 0), prop.hub_radius, 
                               fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(hub_circle)
    
    ax.set_xlabel('Axial Position [m]')
    ax.set_ylabel('Radial Position [m]')
    ax.set_title(f'Flow Field: {label}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if show_propeller and len(flow_field.actuator_disks) > 0:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_propeller_performance(
    propeller,
    V_range: Tuple[float, float] = (20.0, 100.0),
    n_points: int = 50,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 5)
):
    """
    Plot propeller performance curves (thrust, power, efficiency vs. V_inf)
    
    Args:
        propeller: PropellerActuatorDisk instance
        V_range: Range of freestream velocities [m/s]
        n_points: Number of points to evaluate
        save_path: Path to save figure
        figsize: Figure size
    """
    V_inf_array = np.linspace(V_range[0], V_range[1], n_points)
    
    thrust_array = []
    power_array = []
    efficiency_array = []
    advance_ratio_array = []
    
    for V_inf in V_inf_array:
        propeller.set_flight_condition(V_inf)
        T, P, eta = propeller.compute_performance()
        J = propeller.get_advance_ratio()
        
        thrust_array.append(T)
        power_array.append(P / 1000.0)  # Convert to kW
        efficiency_array.append(eta * 100.0)  # Convert to percentage
        advance_ratio_array.append(J)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Thrust vs. velocity
    ax1.plot(V_inf_array, thrust_array, 'b-', linewidth=2)
    ax1.set_xlabel('Freestream Velocity [m/s]')
    ax1.set_ylabel('Thrust [N]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Thrust vs. Velocity')
    
    # Power vs. velocity
    ax2.plot(V_inf_array, power_array, 'r-', linewidth=2)
    ax2.set_xlabel('Freestream Velocity [m/s]')
    ax2.set_ylabel('Power [kW]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Power vs. Velocity')
    
    # Efficiency vs. advance ratio
    ax3.plot(advance_ratio_array, efficiency_array, 'g-', linewidth=2)
    ax3.set_xlabel('Advance Ratio J')
    ax3.set_ylabel('Efficiency [%]', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Efficiency vs. Advance Ratio')
    ax3.axhline(y=max(efficiency_array), color='g', linestyle='--', alpha=0.5)
    
    # Add propeller info
    info_text = (f"Propeller: D={propeller.diameter:.2f}m, "
                f"RPM={propeller.rpm:.0f}, "
                f"Blades={propeller.num_blades}")
    fig.suptitle(info_text, fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_radial_loading(
    propeller,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
):
    """
    Plot radial distribution of thrust and torque
    
    Args:
        propeller: PropellerActuatorDisk instance
        save_path: Path to save figure
        figsize: Figure size
    """
    thrust_dist, torque_dist = propeller.compute_radial_loading()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Thrust distribution
    ax1.plot(propeller.r_stations, thrust_dist, 'b-', linewidth=2)
    ax1.fill_between(propeller.r_stations, 0, thrust_dist, alpha=0.3)
    ax1.set_xlabel('Radial Position [m]')
    ax1.set_ylabel('Thrust per Area [N/m²]')
    ax1.set_title('Radial Thrust Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=propeller.hub_radius, color='r', linestyle='--', 
                alpha=0.5, label='Hub')
    ax1.axvline(x=propeller.radius, color='r', linestyle='--', 
                alpha=0.5, label='Tip')
    ax1.legend()
    
    # Torque distribution
    ax2.plot(propeller.r_stations, torque_dist, 'g-', linewidth=2)
    ax2.fill_between(propeller.r_stations, 0, torque_dist, alpha=0.3, color='g')
    ax2.set_xlabel('Radial Position [m]')
    ax2.set_ylabel('Torque per Area [N·m/m²]')
    ax2.set_title('Radial Torque Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=propeller.hub_radius, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(x=propeller.radius, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_optimization_history(
    history: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8)
):
    """
    Plot optimization history
    
    Args:
        history: Optimization history dictionary
        save_path: Path to save figure
        figsize: Figure size
    """
    iterations = range(len(history['objective']))
    
    fig = plt.figure(figsize=figsize)
    
    # Objective value
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(iterations, history['objective'], 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Convergence History')
    ax1.grid(True, alpha=0.3)
    
    # Design variables
    ax2 = plt.subplot(2, 2, 2)
    design_vars = np.array(history['design_vars'])
    ax2.plot(iterations, design_vars[:, 0], 'r-', label='Diameter [m]', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Diameter [m]')
    ax2.set_title('Diameter Evolution')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(iterations, design_vars[:, 1], 'g-', label='RPM', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('RPM')
    ax3.set_title('RPM Evolution')
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(iterations, design_vars[:, 2], 'm-', label='Pitch [°]', linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Pitch Angle [°]')
    ax4.set_title('Pitch Angle Evolution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_wake_profile(
    flow_field,
    x_locations: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
):
    """
    Plot wake velocity profiles at multiple downstream locations
    
    Args:
        flow_field: FlowField2D instance
        x_locations: List of axial positions to plot
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(x_locations)))
    
    for i, x_loc in enumerate(x_locations):
        r_profile, u_profile = flow_field.get_wake_profile(x_loc)
        
        # Normalize velocity
        u_norm = u_profile / flow_field.V_inf
        
        ax.plot(r_profile, u_norm, color=colors[i], linewidth=2,
               label=f'x = {x_loc:.1f}m')
    
    ax.set_xlabel('Radial Position [m]')
    ax.set_ylabel('Normalized Axial Velocity u/V∞')
    ax.set_title('Wake Velocity Profiles')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Freestream')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(
    propellers: List,
    labels: List[str],
    V_inf: float = 50.0,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
):
    """
    Compare performance of multiple propeller designs
    
    Args:
        propellers: List of PropellerActuatorDisk instances
        labels: Labels for each design
        V_inf: Freestream velocity for comparison
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    designs = []
    thrusts = []
    powers = []
    efficiencies = []
    
    for i, (prop, label) in enumerate(zip(propellers, labels)):
        prop.set_flight_condition(V_inf)
        T, P, eta = prop.compute_performance()
        
        designs.append(label)
        thrusts.append(T)
        powers.append(P / 1000.0)
        efficiencies.append(eta * 100.0)
    
    x_pos = np.arange(len(designs))
    
    # Bar chart for thrust and power
    width = 0.35
    ax1.bar(x_pos - width/2, thrusts, width, label='Thrust [N]', alpha=0.8)
    ax1_twin = ax1.twinx()
    ax1_twin.bar(x_pos + width/2, powers, width, color='orange', 
                label='Power [kW]', alpha=0.8)
    
    ax1.set_xlabel('Design')
    ax1.set_ylabel('Thrust [N]', color='C0')
    ax1_twin.set_ylabel('Power [kW]', color='orange')
    ax1.set_title(f'Performance Comparison at V∞={V_inf} m/s')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(designs, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Bar chart for efficiency
    ax2.bar(x_pos, efficiencies, color='green', alpha=0.8)
    ax2.set_xlabel('Design')
    ax2.set_ylabel('Efficiency [%]')
    ax2.set_title('Propeller Efficiency')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(designs, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
