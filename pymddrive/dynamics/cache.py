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
    # treat the lanvegin force as first-order
    # that is, only update the friction and random force once per MD step
    # meaning each RK step will use the same friction and random force
    F_langevin: Optional[NDArray[np.float64]] = None 
    
    