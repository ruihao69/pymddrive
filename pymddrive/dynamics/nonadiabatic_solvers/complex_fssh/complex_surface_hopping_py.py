# %%
import numpy as np
from numba import njit

from pymddrive.dynamics.nonadiabatic_solvers.fssh.surface_hopping_py import (
    momentum_rescale,
    hop,
    compute_hopping_probabilities,
)
from pymddrive.dynamics.nonadiabatic_solvers.complex_fssh.complex_fssh_math_utils import get_rescale_direction
from pymddrive.my_types import RealVector, ComplexVectorOperator, ComplexOperator

from typing import Tuple, Union


def complex_fssh_surface_hopping_py(
    dt: float,
    current_active_surface: int, 
    P_current: RealVector,
    rho_or_psi: RealVector,
    evals: RealVector,
    v_dot_d: ComplexOperator, # 
    d: ComplexVectorOperator, # note that in conventional FSSH, d and v_dot_d must be real objects !
    mass: Union[float, RealVector],
) -> Tuple[bool, int, RealVector]:
    hopping_probabilities = compute_hopping_probabilities(current_active_surface, dt, v_dot_d, rho_or_psi)
    
    target_surface = hop(current_active_surface, hopping_probabilities)
    if target_surface == current_active_surface:
        return False, current_active_surface, P_current
    else:
        dE = evals[target_surface] - evals[current_active_surface]
        direction = get_rescale_direction(complex_direction=d[current_active_surface, target_surface])
        sucess, P_new = momentum_rescale(direction, P_current, mass, dE)
        final_active_surface = target_surface if sucess else current_active_surface
        return sucess, final_active_surface, P_new
        
