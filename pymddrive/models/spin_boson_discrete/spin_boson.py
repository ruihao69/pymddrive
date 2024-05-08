import attr
import numpy as np
from attrs import define, field

from pymddrive.my_types import RealVector, RealOperator, GenericOperator, GenericVectorOperator, RealVectorOperator
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.models.landry_spin_boson.math_utils import get_sigma_z, get_sigma_x, get_sigma_y

@define
class SpinBoson(HamiltonianBase):
    omega_alpha: RealVector = field(default=None, on_setattr=attr.setters.frozen)
    g_alpha: RealVector = field(default=None, on_setattr=attr.setters.frozen)   
    E: float = field(default=0.5, on_setattr=attr.setters.frozen)
    V: float = field(default=0.5, on_setattr=attr.setters.frozen)
    Omega: float = field(default=0.1, on_setattr=attr.setters.frozen)
    lambd: float = field(default=1.0, on_setattr=attr.setters.frozen)
    kT: float = field(default=1.0, on_setattr=attr.setters.frozen)
    dim: int = field(default=2, init=False) 
    
    @staticmethod
    def H_q(E: float, V: float):
        return E * get_sigma_z() + V * get_sigma_x()
    
    @staticmethod
    def V_c(omega_alpha: RealVector, R: RealVector):
        V_classical = np.multiply(omega_alpha**2, R**2)
        return 0.5 * np.sum(V_classical) * np.eye(2)
    
    @staticmethod
    def V_qc(g_alpha: RealVector, R: RealVector):
        V_quantum_classical = np.sum(g_alpha * R)
        return V_quantum_classical * get_sigma_z()
    
    def H(self, t: float, R: RealVector) -> RealOperator:
        H = self.H_q(self.E, self.V)
        H += self.V_c(self.omega_alpha, R)
        H += self.V_qc(self.g_alpha, R)
        return H
    
    @staticmethod
    def grad_V_c(omega_alpha: RealVector, R: RealVector):
        grad_V_classical = np.multiply(omega_alpha**2, R)
        return grad_V_classical[None, None, :] * np.eye(2)[:, :, None]
    
    @staticmethod
    def grad_V_qc(g_alpha: RealVector):
        sigma_z = get_sigma_z()
        return g_alpha[None, None, :] * sigma_z[:, :, None]
    
    def dHdR(self, t: float, R: RealVector) -> GenericVectorOperator:
        dHdR = self.grad_V_c(self.omega_alpha, R)
        dHdR += self.grad_V_qc(self.g_alpha)
        return dHdR
    
    def get_offsets(self, ) -> RealVector:
        return self.g_alpha / self.omega_alpha**2
    
    def get_donor_R(self, ) -> RealVector:
        return -self.get_offsets()
    
    def get_acceptor_R(self, ) -> RealVector:
        return self.get_offsets()
    
    def get_kT(self, ) -> float:
        return self.kT
    
    

        
    