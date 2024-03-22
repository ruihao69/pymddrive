import numpy as np
from numpy.typing import NDArray

from abc import ABC, abstractmethod
from typing import Optional

class LangevinBase(ABC):
    # def __init__(self, kT: Optional[float]=None) -> None:
    #     self.kT: float = kT if kT is not None else None
    #     self.beta: float = 1.0 / kT if kT is not None else None
        
    def __repr__(self) -> str:
        return f"LangevinBase(kT={self.kT})"
    
    @abstractmethod
    def get_gamma(self, t: float, R: Optional[NDArray[np.float64]]=None) -> NDArray[np.float64]:
        pass
    
    @abstractmethod
    def get_kT(self) -> Optional[float]:
        pass
    
    @abstractmethod
    def get_beta(self) -> Optional[float]:
        pass
    
    def frictional_force(
        self,
        gamma: NDArray[np.float64],
        P: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return -gamma * P
    
    def random_force(
        self,
        dt: float,
        kT: float,
        gamma: NDArray[np.float64],
        P: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        random_force = np.zeros_like(P)
        D = kT * gamma
        random_force[:] = np.random.normal(0, 1, P.shape) * np.sqrt(2 * D / dt)
        return random_force
    
    def evaluate_langevin(
        self,
        t: float,
        R: NDArray[np.float64],
        P: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        # get all the necessary parameters 
        gamma = self.get_gamma(t, R)
        kT = self.get_kT()
        
        friction = self.frictional_force(gamma, P)
        random_force = self.random_force(dt, kT, gamma, P)
        return friction + random_force 
        # return friction
    
    
    