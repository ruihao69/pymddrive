import attr
import numpy as np
from attrs import define, field

from pymddrive.my_types import RealVector, RealOperator, GenericVectorOperator, RealVectorOperator
from pymddrive.models.nonadiabatic_hamiltonian import QuasiFloquetHamiltonianBase
from pymddrive.models.spin_boson_discrete.spin_boson import SpinBoson
from pymddrive.models.landry_spin_boson.math_utils import get_sigma_x, mu_Et, dmu_dR_Et
from pymddrive.pulses import PulseBase as Pulse 


@define
class SpinBosonPulsedFloquet(QuasiFloquetHamiltonianBase):
    omega_alpha: RealVector = field(default=None, on_setattr=attr.setters.frozen)
    g_alpha: RealVector = field(default=None, on_setattr=attr.setters.frozen)   
    E: float = field(default=31.25, on_setattr=attr.setters.frozen)
    V: float = field(default=0.5, on_setattr=attr.setters.frozen)
    Omega: float = field(default=0.1, on_setattr=attr.setters.frozen)
    lambd: float = field(default=1.0, on_setattr=attr.setters.frozen)
    kT: float = field(default=1.0, on_setattr=attr.setters.frozen)
    dim: int = field(default=2, init=False) 
    mu_in_au: float = field(default=0.04, on_setattr=attr.setters.frozen)
    dimless2au: float = field(default=0.00095, init=False)
    
    def __init__(
        self,
        ultrafast_pulse: Pulse,
        envelope_pulse: Pulse,
        NF: int,
        omega_alpha: RealVector,
        g_alpha: RealVector,
        E: float = 31.25,
        V: float = 0.5,
        Omega: float = 0.1,
        lambd: float = 1.0,
        kT: float = 1.0,
        mu_in_au: float = 0.04,
    ) -> None:
        super().__init__(dim=2, ultrafast_pulse=ultrafast_pulse, envelope_pulse=envelope_pulse, NF=NF)
        object.__setattr__(self, "omega_alpha", omega_alpha)
        object.__setattr__(self, "g_alpha", g_alpha)
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "V", V)
        object.__setattr__(self, "Omega", Omega)
        object.__setattr__(self, "lambd", lambd)
        object.__setattr__(self, "kT", kT)
        object.__setattr__(self, "mu_in_au", mu_in_au)
        object.__setattr__(self, "dimless2au", 0.00095)
    
    def transition_dipole(self, R: RealVector) -> RealOperator:
        return self.mu_in_au / self.dimless2au * get_sigma_x()
    
    def transition_dipole_gradient(self, R: RealVector) -> RealVectorOperator:
        return np.zeros((2, 2, len(R)))
    
    def H0(self, R: RealVector) -> RealOperator: 
        H0 = SpinBoson.H_q(self.E, self.V)
        H0 += SpinBoson.V_c(self.omega_alpha, R)
        H0 += SpinBoson.V_qc(self.g_alpha, R)
        return H0
    
    def H1(self, t: float, R: RealVector) -> RealOperator:
        mu = self.transition_dipole(R)
        Et = self.envelope_pulse(t)
        return mu_Et(mu, Et)
    
    def dH0dR(self, R: RealVector) -> GenericVectorOperator:
        dHdR0 = SpinBoson.grad_V_c(self.omega_alpha, R)
        dHdR0 += SpinBoson.grad_V_qc(self.g_alpha)
        return dHdR0
    
    def dH1dR(self, t: float, R: RealVector) -> GenericVectorOperator:
        return dmu_dR_Et(self.transition_dipole_gradient(R), self.envelope_pulse(t))
    
    def get_offsets(self, ) -> RealVector:
        return self.g_alpha / self.omega_alpha**2
    
    def get_donor_R(self, ) -> RealVector:
        return -self.get_offsets()
    
    def get_acceptor_R(self, ) -> RealVector:
        return self.get_offsets()
    
    def get_kT(self, ) -> float:
        return self.kT
    
    def get_dim_nuclear(self, ) -> int:
        return len(self.omega_alpha)    
   