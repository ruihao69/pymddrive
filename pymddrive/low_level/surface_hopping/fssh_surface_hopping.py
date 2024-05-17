from pymddrive.my_types import RealVector, ComplexVector, ComplexOperator, RealOperator, RealVectorOperator
from .._low_level.surface_hopping import (
    fssh_surface_hopping_dm,
    fssh_surface_hopping_wf,
)

from typing import Tuple, Union

def fssh_surface_hopping(
    dt: float,
    current_active_surface: int, 
    P_current: RealVector,
    rho_or_psi: Union[ComplexVector, ComplexOperator],
    evals: RealVector,
    v_dot_d: RealOperator,
    d: RealVectorOperator,
    mass: Union[float, RealVector],
) -> Tuple[bool, int, RealVector]:
    if isinstance(rho_or_psi, ComplexOperator):
        return fssh_surface_hopping_dm(dt, current_active_surface, P_current, rho_or_psi, evals, v_dot_d, d, mass)
    else:
        return fssh_surface_hopping_wf(dt, current_active_surface, P_current, rho_or_psi, evals, v_dot_d, d, mass)