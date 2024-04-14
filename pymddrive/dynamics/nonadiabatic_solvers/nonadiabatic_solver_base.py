import numpy as np

from pymddrive.my_types import RealVector
from pymddrive.low_level.states import State

from abc import ABC, abstractmethod
from typing import Union, Tuple
from collections import namedtuple

NonadiabaticProperties = namedtuple('NonadiabaticProperties', 'R P adiabatic_populations diabatic_populations KE PE')

class NonadiabaticSolverBase(ABC):
    @abstractmethod
    def derivative(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def callback(self, *args, **kwargs):    
        pass
    
    @classmethod
    @abstractmethod
    def initialize(cls, *args, **kwargs):
        pass 
    
    @staticmethod
    def calculate_KE(P: RealVector, mass: Union[float, RealVector]) -> float:
        return np.sum(0.5 * P**2 / mass)
   
    @abstractmethod 
    def calculate_properties(self, t: float, s: State) -> NonadiabaticProperties:
        pass
    
    @abstractmethod
    def get_dim_nuclear(self) -> int:
        pass
    
    @abstractmethod
    def get_dim_electronic(self) -> int:
        pass
    
    @abstractmethod
    def update_F_langevin(self, F_langevin: RealVector) -> None:
        pass