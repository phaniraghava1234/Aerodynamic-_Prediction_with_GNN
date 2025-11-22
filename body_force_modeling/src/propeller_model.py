"""
Parametric Actuator Disk Model for Propeller Aerodynamics

This module implements a parametric body force model based on actuator disk theory
for efficient propeller performance prediction and optimization.

References:
    - Conway, J. T. (1998). Analytical Solutions for the Actuator Disk. J. Fluid Mech.
    - Drela, M. (2006). QPROP Formulation. MIT.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import warnings


class PropellerActuatorDisk:
    """
    Actuator Disk Model for Propeller Performance Prediction
    
    This class implements a parametric body force model where the propeller
    is represented as a thin disk with distributed momentum sources.
    
    Attributes:
        diameter (float): Propeller diameter [m]
        hub_radius (float): Hub radius [m]
        rpm (float): Rotational speed [rev/min]
        num_blades (int): Number of blades
        pitch_angle (float): Blade pitch angle at 75% radius [degrees]
        chord_distribution (str): Radial chord distribution type
    """
    
    def __init__(
        self,
        diameter: float = 2.5,
        hub_radius: float = 0.3,
        rpm: float = 2000.0,
        num_blades: int = 4,
        pitch_angle: float = 25.0,
        chord_distribution: str = 'linear'
    ):
        """
        Initialize actuator disk propeller model
        
        Args:
            diameter: Propeller diameter in meters
            hub_radius: Hub radius in meters
            rpm: Rotational speed in revolutions per minute
            num_blades: Number of propeller blades
            pitch_angle: Pitch angle at 75% radius in degrees
            chord_distribution: Type of chord distribution ('linear', 'elliptical', 'constant')
        """
        self.diameter = diameter
        self.radius = diameter / 2.0
        self.hub_radius = hub_radius
        self.rpm = rpm
        self.num_blades = num_blades
        self.pitch_angle = pitch_angle
        self.chord_distribution = chord_distribution
        
        # Operating conditions (set via set_advance_ratio or set_flight_condition)
        self.V_inf = None  # Freestream velocity [m/s]
        self.rho = 1.225  # Air density [kg/m³] at sea level
        self.mu = 1.81e-5  # Dynamic viscosity [Pa·s]
        
        # Performance coefficients (computed)
        self.C_T = None  # Thrust coefficient
        self.C_P = None  # Power coefficient
        self.C_Q = None  # Torque coefficient
        
        # Radial discretization for distributed forces
        self.n_radial = 50
        self._initialize_geometry()
        
    def _initialize_geometry(self):
        """Initialize radial discretization and geometric properties"""
        # Radial stations from hub to tip
        self.r_stations = np.linspace(self.hub_radius, self.radius, self.n_radial)
        self.r_norm = self.r_stations / self.radius  # Normalized radius
        
        # Compute annular areas for integration
        r_inner = np.zeros(self.n_radial)
        r_outer = np.zeros(self.n_radial)
        r_inner[0] = self.hub_radius
        r_inner[1:] = 0.5 * (self.r_stations[:-1] + self.r_stations[1:])
        r_outer[:-1] = r_inner[1:]
        r_outer[-1] = self.radius
        
        self.annular_areas = np.pi * (r_outer**2 - r_inner**2)
        self.disk_area = np.pi * self.radius**2
        
    def set_flight_condition(
        self,
        V_inf: float,
        altitude: float = 0.0,
        rho: Optional[float] = None
    ):
        """
        Set flight condition for propeller analysis
        
        Args:
            V_inf: Freestream velocity [m/s]
            altitude: Altitude above sea level [m] (used if rho not provided)
            rho: Air density [kg/m³] (if None, computed from altitude)
        """
        self.V_inf = V_inf
        
        if rho is not None:
            self.rho = rho
        else:
            # Simple atmospheric model
            self.rho = 1.225 * np.exp(-altitude / 10400.0)
            
    def get_advance_ratio(self) -> float:
        """
        Compute advance ratio J = V_inf / (n * D)
        
        Returns:
            Advance ratio (dimensionless)
        """
        if self.V_inf is None:
            raise ValueError("Flight condition not set. Call set_flight_condition first.")
        
        n = self.rpm / 60.0  # Convert to rev/s
        J = self.V_inf / (n * self.diameter)
        return J
    
    def set_advance_ratio(self, V_inf: float):
        """Convenience method to set advance ratio via freestream velocity"""
        self.set_flight_condition(V_inf)
        
    def get_tip_speed(self) -> float:
        """
        Compute propeller tip speed
        
        Returns:
            Tip speed [m/s]
        """
        omega = self.rpm * 2.0 * np.pi / 60.0  # rad/s
        return omega * self.radius
    
    def compute_thrust_coefficient(self) -> float:
        """
        Compute thrust coefficient using simplified momentum theory
        
        This uses a semi-empirical relationship based on advance ratio
        and blade geometry. For more accurate predictions, integrate
        with Blade Element Momentum Theory (BEMT).
        
        Returns:
            Thrust coefficient C_T
        """
        J = self.get_advance_ratio()
        
        # Semi-empirical model based on typical propeller data
        # C_T = a0 + a1*J + a2*J^2
        # These coefficients are representative for moderate activity factor propellers
        a0 = 0.12 - 0.002 * (self.pitch_angle - 25.0)
        a1 = -0.08
        a2 = 0.02
        
        C_T = a0 + a1 * J + a2 * J**2
        
        # Physical limits
        C_T = np.clip(C_T, 0.0, 0.5)
        
        self.C_T = C_T
        return C_T
    
    def compute_power_coefficient(self) -> float:
        """
        Compute power coefficient using simplified momentum theory
        
        Returns:
            Power coefficient C_P
        """
        J = self.get_advance_ratio()
        
        # Semi-empirical model
        # C_P = b0 + b1*J + b2*J^2 + b3*J^3
        b0 = 0.08 - 0.001 * (self.pitch_angle - 25.0)
        b1 = -0.05
        b2 = 0.03
        b3 = -0.005
        
        C_P = b0 + b1 * J + b2 * J**2 + b3 * J**3
        
        # Physical limits
        C_P = np.clip(C_P, 0.001, 0.3)  # Avoid division by zero
        
        self.C_P = C_P
        self.C_Q = C_P / (2.0 * np.pi)  # Torque coefficient
        return C_P
    
    def compute_performance(self) -> Tuple[float, float, float]:
        """
        Compute propeller performance: thrust, power, and efficiency
        
        Returns:
            Tuple of (thrust [N], power [W], efficiency)
        """
        if self.V_inf is None:
            raise ValueError("Flight condition not set. Call set_flight_condition first.")
        
        # Compute coefficients
        C_T = self.compute_thrust_coefficient()
        C_P = self.compute_power_coefficient()
        
        # Dimensional quantities
        n = self.rpm / 60.0  # rev/s
        
        thrust = C_T * self.rho * n**2 * self.diameter**4
        power = C_P * self.rho * n**3 * self.diameter**5
        
        # Propeller efficiency
        J = self.get_advance_ratio()
        if C_P > 1e-6:
            efficiency = J * C_T / C_P
        else:
            efficiency = 0.0
            
        efficiency = np.clip(efficiency, 0.0, 1.0)
        
        return thrust, power, efficiency
    
    def compute_radial_loading(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radial distribution of thrust and torque
        
        This provides the distributed body force for CFD coupling.
        Uses a simplified distribution based on blade element theory.
        
        Returns:
            Tuple of (thrust_distribution [N/m], torque_distribution [N])
        """
        # Ensure performance coefficients are computed
        if self.C_T is None or self.C_Q is None:
            self.compute_performance()
        
        # Total thrust and torque
        n = self.rpm / 60.0
        total_thrust = self.C_T * self.rho * n**2 * self.diameter**4
        total_torque = self.C_Q * self.rho * n**2 * self.diameter**5
        
        # Radial loading distribution (simplified - could use BEMT for accuracy)
        # Using a distribution that peaks around 70-75% radius
        r_peak = 0.75
        
        # Thrust distribution (normalized)
        thrust_shape = np.exp(-((self.r_norm - r_peak) / 0.2)**2)
        thrust_shape = thrust_shape / np.sum(thrust_shape * self.annular_areas)
        
        # Torque distribution (normalized)  
        torque_shape = self.r_norm * thrust_shape  # Torque ~ r * thrust
        torque_shape = torque_shape / np.sum(torque_shape * self.r_stations * self.annular_areas)
        
        # Dimensional distributions
        thrust_per_area = total_thrust * thrust_shape  # [N/m²]
        torque_per_area = total_torque * torque_shape  # [N·m/m²]
        
        return thrust_per_area, torque_per_area
    
    def compute_induced_velocity(self, x_locations: np.ndarray) -> np.ndarray:
        """
        Compute axial induced velocity at specified axial locations
        
        Uses momentum theory to estimate the velocity field induced by the propeller.
        
        Args:
            x_locations: Axial positions relative to propeller disk [m]
                        (negative = upstream, positive = downstream)
        
        Returns:
            Induced velocity [m/s] at each x location
        """
        if self.C_T is None:
            self.compute_performance()
        
        # Momentum theory: induced velocity at disk
        # v_i = sqrt(T / (2 * rho * A))
        thrust, _, _ = self.compute_performance()
        v_disk = np.sqrt(thrust / (2.0 * self.rho * self.disk_area))
        
        # Variation with axial distance (simplified model)
        v_induced = np.zeros_like(x_locations)
        
        for i, x in enumerate(x_locations):
            if x < 0:
                # Upstream: gradual acceleration
                v_induced[i] = v_disk * (1.0 - np.exp(x / self.radius))
            else:
                # Downstream: wake expansion
                v_induced[i] = 2.0 * v_disk * np.exp(-x / (4.0 * self.radius))
        
        return v_induced
    
    def get_design_variables(self) -> Dict[str, float]:
        """
        Get current design variables for optimization
        
        Returns:
            Dictionary of design variables
        """
        return {
            'diameter': self.diameter,
            'rpm': self.rpm,
            'pitch_angle': self.pitch_angle,
            'num_blades': self.num_blades
        }
    
    def set_design_variables(self, design_vars: Dict[str, float]):
        """
        Set design variables from optimization
        
        Args:
            design_vars: Dictionary of design variables to update
        """
        if 'diameter' in design_vars:
            self.diameter = design_vars['diameter']
            self.radius = self.diameter / 2.0
            self._initialize_geometry()
            
        if 'rpm' in design_vars:
            self.rpm = design_vars['rpm']
            
        if 'pitch_angle' in design_vars:
            self.pitch_angle = design_vars['pitch_angle']
            
        if 'num_blades' in design_vars:
            self.num_blades = int(design_vars['num_blades'])
    
    def __repr__(self) -> str:
        """String representation of propeller"""
        return (f"PropellerActuatorDisk(D={self.diameter:.2f}m, "
                f"RPM={self.rpm:.0f}, "
                f"Blades={self.num_blades}, "
                f"Pitch={self.pitch_angle:.1f}°)")


class PropellerArray:
    """
    Model for multiple propellers (e.g., distributed electric propulsion)
    
    Manages an array of propellers with individual positions and orientations.
    """
    
    def __init__(self):
        """Initialize empty propeller array"""
        self.propellers = []
        self.positions = []  # (x, y, z) coordinates
        self.orientations = []  # Direction vectors
        
    def add_propeller(
        self,
        propeller: PropellerActuatorDisk,
        position: Tuple[float, float, float],
        orientation: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    ):
        """
        Add a propeller to the array
        
        Args:
            propeller: PropellerActuatorDisk instance
            position: (x, y, z) position in space [m]
            orientation: (x, y, z) direction vector (unit vector)
        """
        self.propellers.append(propeller)
        self.positions.append(np.array(position))
        
        # Normalize orientation vector
        orient = np.array(orientation)
        orient = orient / np.linalg.norm(orient)
        self.orientations.append(orient)
        
    def compute_total_performance(self) -> Tuple[float, float, float]:
        """
        Compute total performance of propeller array
        
        Returns:
            Tuple of (total_thrust [N], total_power [W], average_efficiency)
        """
        total_thrust = 0.0
        total_power = 0.0
        
        for prop in self.propellers:
            T, P, _ = prop.compute_performance()
            total_thrust += T
            total_power += P
        
        # Average efficiency weighted by power
        if total_power > 0:
            avg_efficiency = total_thrust * self.propellers[0].V_inf / total_power
        else:
            avg_efficiency = 0.0
            
        return total_thrust, total_power, avg_efficiency
    
    def __len__(self) -> int:
        """Number of propellers in array"""
        return len(self.propellers)
