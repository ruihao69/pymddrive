import attr
import numpy as np
from attrs import define, field

from pymddrive.my_types import GenericOperator, GenericVectorOperator, RealVector

from abc import ABC, abstractmethod
from typing import Union, Optional, Callable

@define
class HamiltonianBase(ABC):
    dim: int = field(on_setattr=attr.setters.frozen)
    _last_evecs: GenericOperator = field(default=np.zeros((0, 0)))
    _last_deriv_couplings: GenericVectorOperator = field(default=np.zeros((0, 0, 0)))

    @abstractmethod
    def H(self, t: float, R: RealVector) -> GenericOperator:  
        pass
    
    @abstractmethod
    def dHdR(self, t: float, R: RealVector) -> GenericVectorOperator: 
        pass
    
    def update_last_evecs(self, evecs: GenericOperator) -> None:
        if self._last_evecs.size == 0:
            self._last_evecs = np.copy(evecs)
        else:
            np.copyto(self._last_evecs, evecs)
        
    def update_last_deriv_couplings(self, deriv_couplings: GenericVectorOperator) -> None:
        if self._last_deriv_couplings.size == 0:
            self._last_deriv_couplings = np.copy(deriv_couplings)
        else:
            np.copyto(self._last_deriv_couplings, deriv_couplings)
        
    def get_friction(self, ) -> Optional[Union[float, RealVector, Callable]]:
        return None
    
    def get_kT(self, ) -> Optional[float]:
        return None