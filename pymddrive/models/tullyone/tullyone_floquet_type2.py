# %%
import attr
import numpy as np
from attrs import define, field

from pymddrive.my_types import RealVector, RealOperator, RealVectorOperator, GenericVectorOperator, GenericOperator
from pymddrive.models.nonadiabatic_hamiltonian import QuasiFloquetHamiltonianBase
from pymddrive.pulses import PulseBase as Pulse
from pymddrive.models.tullyone.tullyone import TullyOne

@define
class TullyOneFloquet_type2(QuasiFloquetHamiltonianBase):
    A: float = field(default=0.01, on_setattr=attr.setters.frozen)
    B: float = field(default=1.6, on_setattr=attr.setters.frozen)
    C: float = field(default=0.005, on_setattr=attr.setters.frozen)
    D: float = field(default=1.0, on_setattr=attr.setters.frozen)
    dim: int = field(default=2, init=False)
     
    def __init__(
        self, 
        ultrafast_pulse: Pulse,
        envelope_pulse: Pulse,
        NF: int,
        A: float = 0.01,
        B: float = 1.6,
        C: float = 0.005,
        D: float = 1.0,
    ) -> None:
        super().__init__(dim=2, ultrafast_pulse=ultrafast_pulse, envelope_pulse=envelope_pulse, NF=NF)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "D", D)
        
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOneFloquet_type2(A={self.A}, B={self.B}, C={self.C}, D={self.D}, envelope pulse={self.envelope_pulse}, ultrafast pulse={self.ultrafast_pulse}, NF={self.NF}, Omega={self.Omega}, floquet_type={self.floquet_type})" 
   
    def _H0_vector(self, R: RealVector) -> RealVectorOperator:
        V11 = TullyOne.V11(R, self.A, self.B)
        _zeros = np.zeros_like(V11)
        return np.array([[V11, _zeros], [_zeros, -V11]])
    
    def H0(self, R: RealVector) -> RealOperator: 
        return np.sum(self._H0_vector(R), axis=-1)
    
    def _H1_vector(self, t: float, R: RealVector) -> GenericVectorOperator:
        V12 = self.envelope_pulse(t) * np.ones_like(R)
        _zeros = np.zeros_like(V12)
        return np.array([[_zeros, V12], [V12, _zeros]])
     
    def H1(self, t: float, R: RealVector) -> GenericOperator:
        return np.sum(self._H1_vector(t, R), axis=-1)
   
    def dH0dR(self, R: RealVector) -> RealVectorOperator:
        dV11dR = TullyOne.dV11dR(R, self.A, self.B)
        _zeros = np.zeros_like(dV11dR)
        return np.array([[dV11dR, _zeros], [_zeros, -dV11dR]])
   
    def dH1dR(self, t: float, R: RealVector) -> GenericVectorOperator:
        dV12dR = TullyOne.dV12dR(R, self.C, self.D) * self.envelope_pulse(t)
        _zeros = np.zeros_like(dV12dR)
        return np.array([[_zeros, dV12dR], [dV12dR, _zeros]])

# %%
if __name__ == "__main__":
    from pymddrive.pulses import MorletReal, Gaussian
    from pymddrive.models.nonadiabatic_hamiltonian import vectorized_diagonalization
    from pymddrive.models.nonadiabatic_hamiltonian import evaluate_nonadiabatic_couplings
    import matplotlib.pyplot as plt
    NF = 2
    mr = MorletReal(A=0.001, t0=0, tau=300, Omega=0.003, phi=0)
    # mr = MorletReal(A=1, t0=0, tau=500, Omega=0.003, phi=0)
    gaussian = Gaussian.from_quasi_floquet_morlet_real(mr)
    
    tullyone_floquet_type2 = TullyOneFloquet_type2(ultrafast_pulse=mr, envelope_pulse=gaussian, NF=NF)
    t = np.linspace(-1000, 1000, 1000) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, [tullyone_floquet_type2.envelope_pulse(tt) for tt in t], label='envelope')
    ax.plot(t, [tullyone_floquet_type2.ultrafast_pulse(tt) for tt in t], label='ultrafast')
    ax.set_xlabel("t")  
    ax.set_ylabel("Pulse")
    ax.legend()
     
    
    R = np.linspace(-10, 10, 1000)[:, np.newaxis]
    
    dimF = tullyone_floquet_type2.get_floquet_space_dim()
    H = np.zeros((dimF, dimF, R.size))
    dHdR = np.zeros((dimF, dimF, R.shape[1], R.shape[0]))   
    
    for kk in range(R.size):
        H[..., kk] = tullyone_floquet_type2.H(0.0, R[kk])
        dHdR[..., kk] = tullyone_floquet_type2.dHdR(0.0, R[kk])
    
    print(np.round(H[:, :, 0], 3))
        
    evals_list, evecs_list = vectorized_diagonalization(H)
    
    print(tullyone_floquet_type2.get_carrier_frequency())
    print(tullyone_floquet_type2.Omega)
    
     
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for kk in range(dimF):
    # ax.set_xlabel("R")
    # ax.set_ylabel("Eigenvalues")
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for kk in range(dimF):
        ax.plot(R, evals_list[kk], lw=.5, label=f"Eigenvalue {kk}")
        ax.plot(R, H[kk, kk, :], lw=.5, label=rf"$H_{kk}{kk}$")
        ax.set_xlabel("R")
        ax.set_ylabel("Diabatic Potential")
    ax.legend()
   
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    for kk in range(dimF):
        for jj in range(kk+1, dimF):
            ax.plot(R, dHdR[kk, jj, 0, :], label=f"dHdR{kk}{jj}")
    ax.set_xlabel("R")
    ax.set_ylabel("dHdR")
    ax.legend()
    

# %%