import numpy as np
from numpy.typing import NDArray

from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.dynamics.misc_utils import HamiltonianRetureType

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Cache: 
    ##########
    # hamiltonian related cache
    ##########
    hamiltonian: Optional[HamiltonianBase] = None
    hami_return: Optional[HamiltonianRetureType] = None
    
    ##########
    # ehrenfest related cache
    ##########
    meanF: Optional[NDArray[np.float64]] = None
    
    ##########
    # surface hopping related cache 
    ##########
    active_surface: Optional[int] = None
    
    ##########
    # Langevin related cache
    ##########
    F_langevin: Optional[NDArray[np.float64]] = None
    
    