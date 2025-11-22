"""
Flow Field Solver with Actuator Disk Body Forces

Implements a simplified 2D axisymmetric flow solver with body force source terms
for propeller aerodynamic analysis.

References:
    - Sørensen, J. N., & Shen, W. Z. (2002). Numerical Modeling of Wind Turbine Wakes.
    - Conway, J. T. (1998). Analytical Solutions for the Actuator Disk.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy.interpolate import RegularGridInterpolator
import warnings


class FlowField2D:
    """
    2D Axisymmetric Flow Field Solver with Body Force Sources
    
    This class implements a simplified flow solver that includes body force
    source terms from actuator disk models. The flow is assumed to be:
    - 2D axisymmetric (cylindrical coordinates)
    - Incompressible
    - Steady-state
    
    Attributes:
        x_range (tuple): Axial domain extent (x_min, x_max) [m]
        r_range (tuple): Radial domain extent (r_min, r_max) [m]
        resolution (int): Number of grid points in each direction
    """
    
    def __init__(
        self,
        x_range: Tuple[float, float] = (-2.0, 10.0),
        r_range: Tuple[float, float] = (0.0, 5.0),
        resolution: int = 100
    ):
        """
        Initialize flow field domain
        
        Args:
            x_range: Axial extent (x_min, x_max) in meters
            r_range: Radial extent (r_min, r_max) in meters
            resolution: Number of grid points in each direction
        """
        self.x_range = x_range
        self.r_range = r_range
        self.nx = resolution
        self.nr = resolution
        
        # Create computational grid
        self.x = np.linspace(x_range[0], x_range[1], self.nx)
        self.r = np.linspace(r_range[0], r_range[1], self.nr)
        self.X, self.R = np.meshgrid(self.x, self.r, indexing='ij')
        
        # Flow variables
        self.V_inf = 50.0  # Freestream velocity [m/s]
        self.rho = 1.225  # Air density [kg/m³]
        
        # Velocity components
        self.u = None  # Axial velocity [m/s]
        self.v = None  # Radial velocity [m/s]
        self.w = None  # Tangential velocity [m/s]
        self.p = None  # Pressure [Pa]
        
        # Body force sources
        self.body_force_x = np.zeros_like(self.X)  # Axial force [N/m³]
        self.body_force_r = np.zeros_like(self.X)  # Radial force [N/m³]
        self.body_force_theta = np.zeros_like(self.X)  # Tangential force [N/m³]
        
        # Actuator disk locations
        self.actuator_disks = []
        
    def set_freestream(self, V_inf: float, rho: float = 1.225):
        """
        Set freestream conditions
        
        Args:
            V_inf: Freestream velocity [m/s]
            rho: Air density [kg/m³]
        """
        self.V_inf = V_inf
        self.rho = rho
        
    def add_actuator_disk(
        self,
        propeller,
        x_location: float = 0.0,
        thickness: float = 0.1
    ):
        """
        Add an actuator disk to the flow field
        
        Args:
            propeller: PropellerActuatorDisk instance
            x_location: Axial location of disk [m]
            thickness: Disk thickness for force distribution [m]
        """
        self.actuator_disks.append({
            'propeller': propeller,
            'x_location': x_location,
            'thickness': thickness
        })
        
        # Ensure propeller has correct flight condition
        propeller.set_flight_condition(self.V_inf, rho=self.rho)
        
        # Compute radial loading
        thrust_per_area, torque_per_area = propeller.compute_radial_loading()
        
        # Distribute forces on grid
        self._distribute_disk_forces(
            x_location,
            thickness,
            propeller.r_stations,
            thrust_per_area,
            torque_per_area
        )
        
    def _distribute_disk_forces(
        self,
        x_loc: float,
        thickness: float,
        r_stations: np.ndarray,
        thrust_per_area: np.ndarray,
        torque_per_area: np.ndarray
    ):
        """
        Distribute actuator disk forces onto computational grid
        
        Args:
            x_loc: Axial location of disk
            thickness: Disk thickness
            r_stations: Radial stations of loading
            thrust_per_area: Thrust per unit area at each radial station
            torque_per_area: Torque per unit area at each radial station
        """
        # Axial distribution (Gaussian)
        x_dist = np.exp(-((self.X - x_loc) / (thickness/2.0))**2)
        x_norm = np.sum(x_dist, axis=0, keepdims=True)
        x_norm[x_norm == 0] = 1.0  # Avoid division by zero
        x_dist = x_dist / x_norm
        
        # Radial distribution (interpolate from propeller loading)
        for i, x_val in enumerate(self.x):
            for j, r_val in enumerate(self.r):
                # Find closest radial station
                if r_val <= r_stations[-1]:
                    # Interpolate thrust
                    thrust_interp = np.interp(r_val, r_stations, thrust_per_area)
                    torque_interp = np.interp(r_val, r_stations, torque_per_area)
                    
                    # Add to body forces (force per unit volume)
                    # Convert from force per area to force per volume
                    force_density_x = thrust_interp * x_dist[i, j] / thickness
                    force_density_theta = torque_interp * x_dist[i, j] / thickness / (r_val + 1e-6)
                    
                    self.body_force_x[i, j] += force_density_x
                    self.body_force_theta[i, j] += force_density_theta
    
    def solve(self, method: str = 'momentum'):
        """
        Solve flow field with body forces
        
        Args:
            method: Solution method ('momentum' or 'potential')
        """
        if method == 'momentum':
            self._solve_momentum_theory()
        elif method == 'potential':
            self._solve_potential_flow()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _solve_momentum_theory(self):
        """
        Solve using momentum theory (simplified)
        
        This provides a quick analytical solution based on momentum conservation.
        """
        # Initialize velocity fields
        self.u = np.ones_like(self.X) * self.V_inf
        self.v = np.zeros_like(self.X)
        self.w = np.zeros_like(self.X)
        
        # Add induced velocities from body forces
        for disk_info in self.actuator_disks:
            prop = disk_info['propeller']
            x_loc = disk_info['x_location']
            
            # Compute induced velocity
            x_rel = self.x - x_loc
            v_induced = prop.compute_induced_velocity(x_rel)
            
            # Add to axial velocity (broadcast to 2D)
            for i in range(self.nx):
                # Radial decay
                r_effect = np.exp(-(self.r / (2.0 * prop.radius))**2)
                self.u[i, :] += v_induced[i] * r_effect
        
        # Compute pressure from Bernoulli equation
        V_mag = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        p_inf = 0.0  # Reference pressure at infinity
        self.p = p_inf + 0.5 * self.rho * (self.V_inf**2 - V_mag**2)
    
    def _solve_potential_flow(self):
        """
        Solve using potential flow with source panels
        
        More sophisticated method using velocity potential.
        """
        # Placeholder for potential flow solver
        # Would implement panel method or finite difference solution
        warnings.warn("Potential flow solver not fully implemented. Using momentum theory.")
        self._solve_momentum_theory()
    
    def get_velocity_at_point(self, x: float, r: float) -> Tuple[float, float, float]:
        """
        Get velocity components at a specific point
        
        Args:
            x: Axial position [m]
            r: Radial position [m]
        
        Returns:
            Tuple of (u, v, w) velocity components [m/s]
        """
        if self.u is None:
            raise ValueError("Flow field not solved. Call solve() first.")
        
        # Interpolate from grid
        u_interp = RegularGridInterpolator(
            (self.x, self.r),
            self.u,
            bounds_error=False,
            fill_value=self.V_inf
        )
        v_interp = RegularGridInterpolator(
            (self.x, self.r),
            self.v,
            bounds_error=False,
            fill_value=0.0
        )
        w_interp = RegularGridInterpolator(
            (self.x, self.r),
            self.w,
            bounds_error=False,
            fill_value=0.0
        )
        
        u_val = float(u_interp([x, r]))
        v_val = float(v_interp([x, r]))
        w_val = float(w_interp([x, r]))
        
        return u_val, v_val, w_val
    
    def compute_pressure_coefficient(self) -> np.ndarray:
        """
        Compute pressure coefficient C_p = (p - p_inf) / (0.5 * rho * V_inf^2)
        
        Returns:
            Pressure coefficient field
        """
        if self.p is None:
            raise ValueError("Flow field not solved. Call solve() first.")
        
        q_inf = 0.5 * self.rho * self.V_inf**2
        C_p = self.p / q_inf
        
        return C_p
    
    def compute_velocity_magnitude(self) -> np.ndarray:
        """
        Compute total velocity magnitude
        
        Returns:
            Velocity magnitude field [m/s]
        """
        if self.u is None:
            raise ValueError("Flow field not solved. Call solve() first.")
        
        V_mag = np.sqrt(self.u**2 + self.v**2 + self.w**2)
        return V_mag
    
    def integrate_forces(self, body_surface: Callable) -> Tuple[float, float]:
        """
        Integrate pressure forces on a body surface
        
        Args:
            body_surface: Function defining surface r(x)
        
        Returns:
            Tuple of (drag, lift) forces [N]
        """
        # Placeholder for force integration
        # Would integrate pressure over surface
        raise NotImplementedError("Force integration not yet implemented")
    
    def get_wake_profile(self, x_location: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get velocity profile at a downstream location
        
        Args:
            x_location: Axial position to extract profile [m]
        
        Returns:
            Tuple of (radial_positions, axial_velocities)
        """
        if self.u is None:
            raise ValueError("Flow field not solved. Call solve() first.")
        
        # Find nearest x index
        idx = np.argmin(np.abs(self.x - x_location))
        
        return self.r, self.u[idx, :]
    
    def compute_streamlines(
        self,
        num_streamlines: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute streamlines in the flow field
        
        Args:
            num_streamlines: Number of streamlines to compute
        
        Returns:
            Tuple of (x_lines, r_lines) for plotting
        """
        if self.u is None:
            raise ValueError("Flow field not solved. Call solve() first.")
        
        # Simple streamline integration using Euler method
        streamlines_x = []
        streamlines_r = []
        
        # Seed points at inlet
        r_seeds = np.linspace(self.r_range[0] + 0.1, self.r_range[1] - 0.1, num_streamlines)
        
        for r_seed in r_seeds:
            # Integrate streamline
            x_line = [self.x_range[0]]
            r_line = [r_seed]
            
            dx = (self.x_range[1] - self.x_range[0]) / (self.nx * 2)
            
            while x_line[-1] < self.x_range[1] and 0 < r_line[-1] < self.r_range[1]:
                x_curr = x_line[-1]
                r_curr = r_line[-1]
                
                # Get velocity at current point
                u_curr, v_curr, _ = self.get_velocity_at_point(x_curr, r_curr)
                
                # Euler step
                if np.abs(u_curr) > 1e-6:
                    x_new = x_curr + dx
                    r_new = r_curr + v_curr * dx / u_curr
                    
                    x_line.append(x_new)
                    r_line.append(r_new)
                else:
                    break
            
            streamlines_x.append(np.array(x_line))
            streamlines_r.append(np.array(r_line))
        
        return streamlines_x, streamlines_r
    
    def __repr__(self) -> str:
        """String representation of flow field"""
        return (f"FlowField2D(x={self.x_range}, r={self.r_range}, "
                f"grid={self.nx}x{self.nr}, "
                f"V_inf={self.V_inf:.1f} m/s)")
