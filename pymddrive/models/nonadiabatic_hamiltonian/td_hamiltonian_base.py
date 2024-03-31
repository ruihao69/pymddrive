import attr
import numpy as np
from attrs import define, field

from pymddrive.my_types import RealVector, GenericOperator, GenericVectorOperator
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase

from abc import ABC, abstractmethod

@define
class TD_HamiltonianBase(HamiltonianBase):
    
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
        
    def H(self, t: float, R: RealVector) -> GenericOperator:
        return self.H0(R) + self.H1(t, R)
    
    def dHdR(self, t: float, R: RealVector) -> GenericVectorOperator:
        return self.dH0dR(R) + self.dH1dR(t, R)
    
    @abstractmethod 
    def H0(self, R: RealVector) -> GenericOperator:
        pass
    
    @abstractmethod 
    def H1(self, t: float, RealVector) -> GenericOperator:
        pass
    
    @abstractmethod
    def dH0dR(self, R: RealVector) -> GenericVectorOperator:
        pass
    
    @abstractmethod 
    def dH1dR(self, t: float, R: RealVector) -> GenericVectorOperator: 
        pass