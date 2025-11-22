"""
Body Force Modeling for Propeller Aerodynamics

A Python package for parametric body force modeling of propellers
with optimization and CFD integration capabilities.
"""

from .propeller_model import PropellerActuatorDisk, PropellerArray
from .flow_solver import FlowField2D
from .optimizer import PropellerOptimizer
from .visualization import (
    plot_velocity_field,
    plot_propeller_performance,
    plot_radial_loading,
    plot_optimization_history,
    plot_wake_profile,
    plot_comparison
)

__version__ = "1.0.0"
__author__ = "PhD Application Project"

__all__ = [
    'PropellerActuatorDisk',
    'PropellerArray',
    'FlowField2D',
    'PropellerOptimizer',
    'plot_velocity_field',
    'plot_propeller_performance',
    'plot_radial_loading',
    'plot_optimization_history',
    'plot_wake_profile',
    'plot_comparison'
]
