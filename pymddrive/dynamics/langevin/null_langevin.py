import numpy as np

from pymddrive.dynamics.langevin.langevin_base import LangevinBase
from pymddrive.my_types import RealVector


class NullLangevin(LangevinBase):
    def __init__(self,) -> None:
        pass
    
    # override the get_gamma method in LangevinBase
    def get_gamma(self, t: float, R: RealVector):
        raise NotImplementedError(f"Trying to call get_gamma in NullLangevin")
    
    def get_mass(self):
        raise NotImplementedError(f"Trying to call get_mass in NullLangevin")
    
    def get_kT(self):
        raise NotImplementedError(f"Trying to call get_kT in NullLangevin")
    
    def get_beta(self):
        raise NotImplementedError(f"Trying to call get_beta in NullLangevin")
    
    def evaluate_langevin(
        self,
        t: float,
        R: RealVector,
        P: RealVector,
        dt: float,
    ) -> RealVector:
        return np.zeros_like(P)