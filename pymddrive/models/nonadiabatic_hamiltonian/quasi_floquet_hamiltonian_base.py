import attr
from attrs import define, field

from pymddrive.my_types import RealVector, GenericOperator, GenericVectorOperator
from pymddrive.pulses import PulseBase as Pulse
from pymddrive.pulses import get_carrier_frequency
from pymddrive.models.nonadiabatic_hamiltonian.hamiltonian_base import HamiltonianBase
from pymddrive.models.floquet import FloquetType, ValidEnvolpeFunctions
from pymddrive.models.floquet import get_floquet_space_dim
from pymddrive.models.floquet import get_envelope_function_type, get_floquet_type
from pymddrive.models.floquet import get_HF, get_dHF_dR

from typing import Callable
from abc import abstractmethod

@define
class QuasiFloquetHamiltonianBase(HamiltonianBase):
    envelope_pulse: Pulse = field(default=None, on_setattr=attr.setters.frozen)
    ultrafast_pulse: Pulse = field(default=None, on_setattr=attr.setters.frozen)
    driving_Omega: float = field(default=None, on_setattr=attr.setters.frozen)
    floquet_type: FloquetType = field(default=None, on_setattr=attr.setters.frozen)
    envelope_function_type: ValidEnvolpeFunctions= field(default=None, on_setattr=attr.setters.frozen)
    NF: int = field(default=None, on_setattr=attr.setters.frozen)
    
    def __init__(
        self,
        dim: int,
        ultrafast_pulse: Pulse,
        envelope_pulse: Pulse,
        NF: int,
    ) -> None:
        """ Quasi-Floquet Hamiltonian for a time-dependent Hamiltonian """
        """ whose time dependence is definded by a 'Pulse' object. """
        
        super().__init__(dim)
        
        object.__setattr__(self, "driving_Omega", get_carrier_frequency(ultrafast_pulse))
        # assert (self.driving_Omega>0) and (self.driving_Omega is not None), "The carrier frequency must be a positive number."
        assert self.driving_Omega is not None, "The carrier frequency must be a positive number."
        
        object.__setattr__(self, "floquet_type", get_floquet_type(ultrafast_pulse))
        object.__setattr__(self, "envelope_function_type", get_envelope_function_type(ultrafast_pulse)) 
        
        assert self.envelope_function_type.value == envelope_pulse.__class__.__name__, \
            f"Invalid envelope function type. Expected {self.envelope_function_type.value}, yet got {envelope_pulse.__class__.__name__}"
        
        object.__setattr__(self, "NF", NF)
        object.__setattr__(self, "ultrafast_pulse", ultrafast_pulse)
        object.__setattr__(self, "envelope_pulse", envelope_pulse)
        
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
        
    def H(self, t: float, R: RealVector) -> GenericOperator:
        H0 = self.H0(R)
        H1 = self.H1(t, R)
        return self._get_HF(H0, H1, self.driving_Omega, self.NF)
    
    def dHdR(self, t: float, R: RealVector) -> GenericVectorOperator:
        dH0dR = self.dH0dR(R)
        dH1dR = self.dH1dR(t, R)
        return self._get_dHF_dR(dH0dR, dH1dR, self.NF)
    
    def get_floquet_space_dim(self) -> int:
        return get_floquet_space_dim(self.dim, self.NF)
        
    def get_carrier_frequency(self) -> float:
        return self.driving_Omega
    
    def _get_HF(self, H0: GenericOperator, H1: GenericOperator, Omega: float, NF: int) -> GenericOperator:
        return get_HF(self.floquet_type, H0, H1, Omega, NF)
    
    def _get_dHF_dR(self, dH0dR: GenericVectorOperator, dH1dR: GenericVectorOperator, NF: int) -> GenericVectorOperator:
        return get_dHF_dR(self.floquet_type, dH0dR, dH1dR, NF)