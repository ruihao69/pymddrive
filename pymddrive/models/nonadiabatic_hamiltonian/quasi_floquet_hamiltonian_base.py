from pymddrive.my_types import RealVector, GenericOperator, GenericVectorOperator
from pymddrive.pulses import PulseBase as Pulse
from pymddrive.pulses import get_carrier_frequency
from pymddrive.models.nonadiabatic_hamiltonian.hamiltonian_base import HamiltonianBase
from pymddrive.models.floquet import FloquetType
from pymddrive.models.floquet import get_floquet_space_dim
from pymddrive.models.floquet import get_envelope_function_type, get_floquet_type
from pymddrive.low_level.floquet import get_HF_cos, get_dHF_dR_cos

from typing import Union
from abc import abstractmethod
    
class QuasiFloquetHamiltonianBase(HamiltonianBase):
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
        
        self.Omega = get_carrier_frequency(ultrafast_pulse)
        # assert (self.Omega>0) and (self.Omega is not None), "The carrier frequency must be a positive number."
        assert self.Omega is not None, "The carrier frequency must be a positive number."
        
        self.floquet_type = get_floquet_type(ultrafast_pulse)
        self.envelope_function_type = get_envelope_function_type(envelope_pulse)
        
        assert self.envelope_function_type.value == envelope_pulse.__class__.__name__, \
            f"Invalid envelope function type. Expected {self.envelope_function_type.value}, yet got {envelope_pulse.__class__.__name__}"
        
        self.NF = NF
        # self._ultrafast_pulse = ultrafast_pulse
        self.envelope_pulse = envelope_pulse
        
        if self.floquet_type == FloquetType.COSINE:
            self._get_HF = get_HF_cos
            self._get_dHF_dR = get_dHF_dR_cos
        elif self.floquet_type == FloquetType.SINE:
            raise NotImplementedError("The sine type of Floquet Hamiltonian is not implemented yet.") 
        elif self.floquet_type == FloquetType.EXPONENTIAL:
            raise NotImplementedError("The exponential type of Floquet Hamiltonian is not implemented yet.")
        else:
            raise NotImplementedError("The Hamiltonian type is not implemented yet.")
        
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
        return self._get_HF(H0, H1, self.Omega, self.NF)
    
    def dHdR(self, t: float, R: RealVector) -> GenericVectorOperator:
        dH0dR = self.dH0dR(R)
        dH1dR = self.dH1dR(t, R)
        return self._get_dHF_dR(dH0dR, dH1dR, self.NF)
    
    def get_floquet_space_dim(self) -> int:
        return get_floquet_space_dim(self.dim, self.NF)
        
    def get_carrier_frequency(self) -> float:
        return self.Omega