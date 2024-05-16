# %%
import numpy as np
from numba import njit

from pymddrive.my_types import RealVector, GenericOperator, RealOperator, RealVectorOperator, ComplexOperator, ComplexVector

from typing import Tuple, Union

def momentum_rescale(
     d_ij: RealVector,
     momentum: RealVector,
     mass: Union[float, RealVector],
     dE: float,
 ) -> Tuple[bool, np.ndarray]:
     a = 0.5 * np.dot(d_ij / mass, d_ij)
     b = np.dot(momentum / mass, d_ij)
     c = dE

     roots = np.roots([a, b, c])
     if np.iscomplex(roots).any():
         return False, momentum
     else:
         argmin = np.argmin(np.abs(roots))
         kappa = roots[argmin]
         return True, momentum + kappa * d_ij
     
@njit
def hop(
    hopping_probabilities: RealVector
) -> int:
    random_number = np.random.rand()
    cum_prob: float = 0.0
    surf_index: int = 0
    while cum_prob < random_number:
        cum_prob += hopping_probabilities[surf_index]
        if cum_prob > random_number:
            break
        surf_index += 1
        
    return surf_index

@njit
def compute_hopping_probabilities_wf(
    active_surface: int,
    dt: float, 
    v_dot_d: RealVector,
    psi: ComplexVector,
) -> RealVector:
    probabilities = np.zeros(psi.shape[0], dtype=np.float64)
    probabilities[active_surface] = 1.0
    for ii in range(psi.shape[0]):
        if ii == active_surface:
            continue
        tmp = -2.0 * dt * np.real(v_dot_d[active_surface, ii] * psi[ii] / psi[active_surface])
        probabilities[ii] = tmp if tmp > 0.0 else 0.0
        probabilities[active_surface] -= probabilities[ii]
    
    return probabilities

@njit
def compute_hopping_probabilities_dm(
    active_surface: int,
    dt: float, 
    v_dot_d: RealVector,
    rho: ComplexOperator,
) -> RealVector:
    probabilities = np.zeros(rho.shape[0], dtype=np.float64)
    probabilities[active_surface] = 1.0
    for ii in range(rho.shape[0]):
        if ii == active_surface:
            continue
        tmp = 2.0 * dt * np.real(v_dot_d[active_surface, ii] * rho[ii, active_surface] / rho[active_surface, active_surface])
        probabilities[ii] = tmp if tmp > 0.0 else 0.0
        probabilities[active_surface] -= probabilities[ii]
    
    return probabilities

def compute_hopping_probabilities(
    active_surface: int,
    dt: float, 
    v_dot_d: RealVector,
    rho_or_psi: Union[ComplexVector, ComplexOperator],
) -> RealVector:
    if rho_or_psi.ndim == 2:
        return compute_hopping_probabilities_dm(active_surface, dt, v_dot_d, rho_or_psi)
    else:
        return compute_hopping_probabilities_wf(active_surface, dt, v_dot_d, rho_or_psi)
    

def fssh_surface_hopping_py(
    dt: float,
    current_active_surface: int, 
    P_current: RealVector,
    rho_or_psi: RealVector,
    evals: RealVector,
    v_dot_d: RealOperator, # 
    d: RealVectorOperator, # note that in conventional FSSH, d and v_dot_d must be real objects !
    mass: Union[float, RealVector],
) -> Tuple[bool, int, RealVector]:
    hopping_probabilities = compute_hopping_probabilities(current_active_surface, dt, v_dot_d, rho_or_psi)
    
    target_surface = hop(hopping_probabilities)
    if target_surface == current_active_surface:
        return False, current_active_surface, P_current
    else:
        dE = evals[target_surface] - evals[current_active_surface]
        direction = d[current_active_surface, target_surface]
        sucess, P_new = momentum_rescale(direction, P_current, mass, dE)
        final_active_surface = target_surface if sucess else current_active_surface
        return sucess, final_active_surface, P_new
        
