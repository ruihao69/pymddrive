from pymddrive.my_types import GenericOperator, GenericVectorOperator, RealVector

from abc import ABC, abstractmethod
from typing import Union, Optional, Callable

class HamiltonianBase(ABC):
    def __init__(
        self,
        dim: int,
    ) -> None:
        self.dim: int = dim
        self.last_evecs: Optional[GenericOperator] = None
        self.last_deriv_couplings: Optional[GenericVectorOperator] = None
     
    @abstractmethod
    def H(self, t: float, R: RealVector) -> GenericOperator:  
        pass
    
    @abstractmethod
    def dHdR(self, t: float, R: RealVector) -> GenericVectorOperator: 
        pass
    
    def update_last_evecs(self, evecs: GenericOperator) -> None:
        self.last_evecs = evecs
        
    def update_last_deriv_couplings(self, deriv_couplings: GenericVectorOperator) -> None:
        self.last_deriv_couplings = deriv_couplings
        
    def get_friction(self, ) -> Optional[Union[float, RealVector, Callable]]:
        return None
    
    def get_kT(self, ) -> Optional[float]:
        return None