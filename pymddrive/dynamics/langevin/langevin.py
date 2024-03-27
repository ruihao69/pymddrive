import numpy as np
from numpy.typing import NDArray

from .langevin_base import LangevinBase

from dataclasses import dataclass
from typing import Union, Optional

@dataclass(frozen=True)
class Langevin(LangevinBase):
    kT: float
    mass: Union[float, NDArray[np.float64]]
    gamma: Union[float, NDArray[np.float64]]
    
    # override the get_gamma method in LangevinBase
    def get_gamma(self, t: float, R: Optional[NDArray[np.float64]]=None) -> NDArray[np.float64]:
        return self.gamma
    
    def get_mass(self) -> Union[float, NDArray[np.float64]]:
        return self.mass
    
    def get_kT(self) -> float:
        return self.kT
    
    def get_beta(self) -> float:
        return 1.0 / self.kT