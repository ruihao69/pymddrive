from pymddrive.my_types import RealVector, GenericOperator, GenericVectorOperator
from pymddrive.models.nonadiabatic_hamiltonian.hamiltonian_base import HamiltonianBase
from pymddrive.pulses import PulseBase as Pulse

from abc import abstractmethod
from typing import Union

class TD_HamiltonianBase(HamiltonianBase):
    def __init__(
        self,
        dim: int,
        pulse: Pulse,
    ) -> None:
        """ Time-dependent nonadiabatic Hamiltonian. """
        """ The time dependence is defined by a 'Pulse' object. """
        """ The pulse consists of a carrier frequency <Omega> and an envelope <E(t)>. """
        super().__init__(dim)
        self.pulse = pulse
        
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