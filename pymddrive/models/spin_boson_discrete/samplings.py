import numpy as np

from pymddrive.my_types import RealVector

from typing import Tuple, List

def boltzmann_sampling(
    n_ensemble: int,    
    kT: float,
    omega_alpha: RealVector, 
    R_eq: RealVector,
) -> Tuple[List[RealVector], List[RealVector]]:
    beta = 1 / kT
    
    sigma_R_alpha = 1.0 / np.sqrt(beta) / omega_alpha
    sigma_P_alpha = np.sqrt(1.0 / beta) * np.ones_like(omega_alpha)
    
    R_ensemble = [np.random.normal(R_eq, sigma_R_alpha) for _ in range(n_ensemble)]
    P_enemble = [np.random.normal(0, sigma_P_alpha) for _ in range(n_ensemble)]
    
    return R_ensemble, P_enemble

def wigner_sampling(
    n_enemble: int,
    kT: float,
    omega_alpha: RealVector,
    R_eq: RealVector,
) -> Tuple[List[RealVector], List[RealVector]]:
    beta = 1 / kT
    
    sigma_dimless = np.sqrt(0.5 / np.tanh(0.5 * beta * omega_alpha))
    dimless_to_P = np.sqrt(omega_alpha)
    dimless_to_R = np.sqrt(1.0 / omega_alpha)
    
    sigma_P = sigma_dimless * dimless_to_P
    sigma_R = sigma_dimless * dimless_to_R
    
    R_ensemble = [np.random.normal(R_eq, sigma_R) for _ in range(n_enemble)]
    P_enemble = [np.random.normal(0, sigma_P) for _ in range(n_enemble)]
    return R_ensemble, P_enemble
    
    
    
    
    
    