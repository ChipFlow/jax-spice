"""Analysis context for JAX-SPICE

Holds simulation state and parameters passed to device evaluations.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class AnalysisContext:
    """Context passed to device evaluations during analysis
    
    Provides simulation state like current time, timestep, and
    analysis-specific parameters.
    
    Attributes:
        time: Current simulation time (None for DC analysis)
        dt: Current timestep (None for DC analysis)
        temperature: Circuit temperature in Kelvin
        analysis_type: Type of analysis ('dc', 'tran', 'ac')
        iteration: Current Newton-Raphson iteration number
        prev_solution: Previous timepoint solution (for transient)
    """
    time: Optional[float] = None
    dt: Optional[float] = None
    temperature: float = 300.15  # Room temp in Kelvin
    analysis_type: str = 'dc'
    iteration: int = 0
    prev_solution: Optional[Dict[str, float]] = None
    
    # Integration coefficients for reactive elements
    # For backward Euler: qdot = (q - q_prev) / dt
    # Stored as: qdot = c0 * q + c1 * q_prev + rhs_correction
    c0: float = 0.0  # Coefficient for current charge
    c1: float = 0.0  # Coefficient for previous charge
    rhs_correction: float = 0.0  # RHS correction term
    
    def is_dc(self) -> bool:
        """Check if this is a DC analysis"""
        return self.time is None
    
    def is_transient(self) -> bool:
        """Check if this is a transient analysis"""
        return self.time is not None
