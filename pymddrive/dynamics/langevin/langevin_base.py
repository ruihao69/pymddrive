import numpy as np

from pymddrive.my_types import RealVector

from abc import ABC, abstractmethod
from typing import Union


class LangevinBase(ABC):
    
    @abstractmethod
    def get_gamma(self, t: float, R: RealVector) -> Union[float, RealVector]:
        pass
    
    @abstractmethod
    def get_mass(self) -> Union[float, RealVector]:
        pass
    
    @abstractmethod
    def get_kT(self) -> float:
        pass
    
    @abstractmethod
    def get_beta(self) -> float:
        pass
    
    def frictional_force(
        self,
        gamma: Union[float, RealVector],
        P: RealVector,
    ) -> RealVector:
        return -gamma * P
    
    def random_force(
        self,
        dt: float,
        kT: float,
        mass: Union[float, RealVector],
        gamma: Union[float, RealVector],
        P: RealVector,
    ) -> RealVector:
        random_force = np.zeros_like(P)
        D = kT * gamma * mass
        random_force[:] = np.random.normal(0, 1, P.shape) * np.sqrt(2 * D / dt)
        return random_force
    
    def evaluate_langevin(
        self,
        t: float,
        R: RealVector,
        P: RealVector,
        dt: float,
    ) -> RealVector:
        gamma = self.get_gamma(t, R)
        mass = self.get_mass()
        kT = self.get_kT()
        
        friction = self.frictional_force(gamma, P)
        random_force = self.random_force(dt, kT, mass, gamma, P)
        return friction + random_force 