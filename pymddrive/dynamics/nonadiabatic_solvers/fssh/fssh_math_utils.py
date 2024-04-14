import numpy as np
from numba import njit

from pymddrive.my_types import ComplexOperator, ComplexVector, ActiveSurface

from typing import Union

def initialize_active_surface(
    rho_or_psi: Union[ComplexVector, ComplexOperator],
) -> ActiveSurface:
    _active_surf = initialize_active_surface_impl_rho(rho_or_psi) if isinstance(rho_or_psi, ComplexOperator) else initialize_active_surface_impl_psi(rho_or_psi)
    return np.array([_active_surf], dtype=np.int64)

@njit
def initialize_active_surface_impl_rho(
    rho: ComplexOperator,
) -> int:
    random_number = np.random.rand()
    init_state = 0
    random_number: float = np.random.rand()
    init_state: int = 0
    accumulative_prob: float = 0.0
    tr_rho: float = np.trace(rho).real # normalize the probability
    while (init_state < rho.shape[0]):
        accumulative_prob += rho[init_state, init_state].real / tr_rho
        if accumulative_prob > random_number:
            break
        init_state += 1
    return init_state

@njit
def initialize_active_surface_impl_psi(
    psi: ComplexVector,
) -> int:
    random_number = np.random.rand()
    init_state = 0
    accumulative_prob: float = 0.0
    while (init_state < psi.shape[0]):
        accumulative_prob += np.abs(psi[init_state])**2
        if accumulative_prob > random_number:
            break
        init_state += 1
    return init_state
    