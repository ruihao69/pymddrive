# %%
import attr
import numpy as np
from attrs import define, field

from pymddrive.my_types import RealVector, RealOperator, RealVectorOperator, GenericVectorOperator, GenericOperator
from pymddrive.models.nonadiabatic_hamiltonian import TD_HamiltonianBase
from pymddrive.pulses import PulseBase as Pulse
from pymddrive.models.tullyone.tullyone import TullyOne

@define
class TullyOneTD_type1(TD_HamiltonianBase):
    pulse: Pulse = field(default=None, on_setattr=attr.setters.frozen)
    A: float = field(default=0.01, on_setattr=attr.setters.frozen)
    B: float = field(default=1.6, on_setattr=attr.setters.frozen)
    C: float = field(default=0.005, on_setattr=attr.setters.frozen)
    D: float = field(default=1.0, on_setattr=attr.setters.frozen)
    dim: int = field(default=2, init=False)
    
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOneTD_type1(A={self.A}, B={self.B}, C={self.C}, D={self.D}, pulse={self.pulse})" 
    
    def _H0_vector(self, R: RealVector) -> RealVectorOperator:
        V11 = TullyOne.V11(R, self.A, self.B)
        V12 = TullyOne.V12(R, self.C, self.D)
        return np.array([[V11, V12], [V12, -V11]])
     
    def H0(self, R: RealVector) -> RealOperator:
        return np.sum(self._H0_vector(R), axis=-1)
    
    def _H1_vector(self, t: float, R: RealVector) -> GenericVectorOperator:
        V12 = self.pulse(t) * np.ones_like(R)
        _zeros = np.zeros_like(V12)
        return np.array([[_zeros, V12], [V12, _zeros]])
    
    def H1(self, t: float, R: RealVector) -> GenericOperator:
        return np.sum(self._H1_vector(t, R), axis=-1)
    
    def _H_vector(self, t: float, R: RealVector) -> GenericVectorOperator:
        return self._H0_vector(R) + self._H1_vector(t, R)   
    
    def dH0dR(self, R: RealVector) -> RealVectorOperator:
        dV11dR = TullyOne.dV11dR(R, self.A, self.B)
        dV12dR = TullyOne.dV12dR(R, self.C, self.D)  
        return np.array([[dV11dR, dV12dR], [dV12dR, -dV11dR]])
    
    def dH1dR(self, t: float, R: RealVector) -> GenericVectorOperator:
        return np.zeros_like(self._H1_vector(t, R), dtype=np.complex128) if self.is_complex(t) else np.zeros_like(self._H1_vector(t, R), dtype=np.float64)
        
    def is_complex(self, t: float) -> bool:
        return isinstance(self.pulse(t), complex)
    
# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pymddrive.models.nonadiabatic_hamiltonian import vectorized_diagonalization
    from pymddrive.models.nonadiabatic_hamiltonian import evaluate_nonadiabatic_couplings
    from pymddrive.pulses import MorletReal
    
    mr = MorletReal(A=1, t0=4, tau=1, Omega=10.0, phi=0)
    
    tully_one = TullyOneTD_type1(pulse=mr, A=0.01, B=1.6, C=0.005, D=1.0)
    
    R = np.linspace(-10, 10, 1000)
    H = tully_one._H_vector(0.0, R)
    
    time = np.linspace(0, 12, 3000) 
    pulse = np.array([tully_one.pulse(tt) for tt in time])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(time, pulse, lw=.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    plt.show()
    
    evals_list, evecs_list = vectorized_diagonalization(H)
    
    nac_list = np.zeros((H.shape[0], H.shape[0], H.shape[-1], R.shape[0]), dtype=H.dtype)
    F_list = np.zeros((H.shape[0], H.shape[-1], R.shape[0]), dtype=H.dtype)
    for kk in range(H.shape[-1]):
        dHdR = tully_one.dHdR(0.0, np.array([R[kk]]))
        nac_list[..., kk], F_list[..., kk] = evaluate_nonadiabatic_couplings(dHdR, evals_list[:, kk], evecs_list[:, :, kk])

    
    print(H.shape)
    print(evals_list.shape)
    print(evecs_list.shape)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(R, H[0, 0, :], label="H11")
    ax.plot(R, H[1, 1, :], label="H22")
    
    ax.plot(R, evals_list[0, :], label="E1")
    ax.plot(R, evals_list[1, :], label="E2")
    
    ax.plot(R, -nac_list[0, 1, 0, :]/50, label="NAC12")    
    
    ax.legend()
    plt.show()
    
        
    
# %%