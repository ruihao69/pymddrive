from pymddrive.dynamics.langevin.langevin_base import LangevinBase
from pymddrive.my_types import RealVector

from typing import Union


class Langevin(LangevinBase):
    def __init__(self, kT: float, mass: Union[float, RealVector], gamma: Union[float, RealVector]) -> None:
        self.kT = kT
        self.mass = mass
        self.gamma = gamma
    
    def get_gamma(self, t: float, R: RealVector) -> Union[float, RealVector]:
        return self.gamma
    
    def get_mass(self) -> Union[float, RealVector]:
        return self.mass
    
    def get_kT(self) -> float:
        return self.kT
    
    def get_beta(self) -> float:
        return 1 / self.kT