# %%
import numpy as np
from numba import njit

from pymddrive.my_types import ComplexVectorOperator, GenericVectorOperator, ComplexOperator

from typing import Tuple

@njit
def apply_decoherence(
    rho: ComplexOperator,
    active_state: int,
    F_hellmann_feynman: GenericVectorOperator,
    delta_F: GenericVectorOperator,
    delta_R: ComplexVectorOperator,
    delta_P: ComplexVectorOperator,
    dt: float,
) -> Tuple[bool, ComplexOperator, ComplexVectorOperator, ComplexVectorOperator]:
    new_rho = rho.copy()
    new_delta_R = delta_R.copy()    
    new_delta_P = delta_P.copy()
    n_elec: int = F_hellmann_feynman.shape[0]
    decohere_count: int = 0
    reset_count: int = 0
    for nn in range(n_elec):
        if nn == active_state:
            continue
        random_number = np.random.rand()
        gamma_reset, gamma_decoherence = evaluate_decoherence_rates(
            active_state=active_state,
            target_state=nn,
            F_hellmann_feynman=F_hellmann_feynman,
            delta_F=delta_F,
            delta_R=delta_R,
            dt=dt,
        )
        if random_number < gamma_decoherence:
            decohere_count += 1
            new_rho[nn, :] = 0.0
            new_rho[:, nn] = 0.0
            new_rho[active_state, active_state] += rho[nn, nn]
            for ll in range(n_elec):
                if ll == active_state or ll == nn:
                    continue
                new_rho[active_state, ll] *= np.sqrt(np.real((rho[active_state, active_state] + rho[nn, nn]) / rho[active_state, active_state]))
                new_rho[ll, active_state] = np.conj(new_rho[active_state, ll])
            new_delta_R[nn, :, :] = 0.0 
            new_delta_R[:, nn, :] = 0.0
            new_delta_P[nn, :, :] = 0.0
            new_delta_P[:, nn, :] = 0.0
        elif random_number < gamma_reset:
            reset_count += 1
            new_delta_R[nn, :, :] = 0.0
            new_delta_R[:, nn, :] = 0.0
            new_delta_P[nn, :, :] = 0.0
            new_delta_P[:, nn, :] = 0.0
    return (decohere_count + reset_count) > 0, new_rho, new_delta_R, new_delta_P
            

        
            
            


@njit
def evaluate_decoherence_rates(
    active_state: int,
    target_state: int,
    F_hellmann_feynman: GenericVectorOperator,
    delta_F: GenericVectorOperator,
    delta_R: ComplexVectorOperator,
    dt: float,
) -> Tuple[float, float]:
    gamma_reset = evaluate_reset_rate(target_state, delta_F, delta_R, dt)
    second_term = evaluate_decoherence_rate_second_term(active_state, target_state, F_hellmann_feynman, delta_R, dt)
    gamma_decoherence = - gamma_reset + second_term
    return gamma_reset, gamma_decoherence

@njit
def evaluate_decoherence_rate_second_term(
    active_state: int,
    target_state: int,
    F_hellmann_feynman: GenericVectorOperator,
    delta_R: ComplexVectorOperator,
    dt: float,
) -> float:
    dim_nucl = F_hellmann_feynman.shape[-1]
    rate_tmp: complex = 0.0 + 0.0j
    for kk in range(dim_nucl):
        rate_tmp += F_hellmann_feynman[active_state, target_state, kk] * delta_R[target_state, target_state, kk]
    return -2.0 * dt * np.abs(rate_tmp)

@njit
def evaluate_reset_rate(
    target_state: int,
    delta_F: GenericVectorOperator,
    delta_R: ComplexVectorOperator,
    dt: float,
) -> float:
    dim_nucl = delta_F.shape[-1]
    rate: float = 0.0
    for kk in range(dim_nucl):
        rate += -dt / 2.0 * np.real(delta_R[target_state, target_state, kk] * delta_F[target_state, target_state, kk])
    return rate
    
