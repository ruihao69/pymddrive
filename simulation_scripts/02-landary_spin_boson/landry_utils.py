import numpy as np

from pymddrive.my_types import RealVector

from typing import Tuple

def sample_boltzmann(
    n_samples: int, 
    kT: float, # thermal energy
    Omega: float, # harmonic frequency
    mass: float, # reduced mass of the nuclear mode
    R_eq: float, # equilibrium position 
) -> Tuple[RealVector, RealVector]:
    beta = 1.0 / kT
    
    # sample the momentum
    sigma_momentum = np.sqrt(mass / beta)
    momentum_samples = np.random.normal(0, sigma_momentum, n_samples)
    
    # sample the position
    sigma_R = 1.0 / np.sqrt(beta * mass) / Omega
    position_samples = np.random.normal(R_eq, sigma_R, n_samples)
    return position_samples, momentum_samples
    