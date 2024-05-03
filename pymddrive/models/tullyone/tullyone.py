# %% Use tully one to test the nonadiabatic hamiltonian abc class
import numpy as np
import attr
from attrs import define, field

from pymddrive.my_types import RealVector, RealOperator, RealVectorOperator
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase

@define
class TullyOne(HamiltonianBase):
    A: float = field(default=0.01, on_setattr=attr.setters.frozen)
    B: float = field(default=1.6, on_setattr=attr.setters.frozen)
    C: float = field(default=0.005, on_setattr=attr.setters.frozen)
    D: float = field(default=1.0, on_setattr=attr.setters.frozen)
    dim: int = field(default=2, init=False)
    
        
    @staticmethod
    def V11(
        R: RealVector,
        A: float,
        B: float,
    ) -> RealVector:
        sign = np.sign(R)
        return sign * A * (1 - np.exp(-sign * B * R))
    
    @staticmethod
    def V12(
        R: RealVector,
        C: float,
        D: float,
    ) -> RealVector:
        return C * np.exp(-D * R**2)
    
    @staticmethod
    def dV11dR(
        R: RealVector,
        A: float,
        B: float,
    ) -> RealVector:
        return A * B * np.exp(-np.abs(R) * B) 
    
    @staticmethod
    def dV12dR(
        R: RealVector,
        C: float,
        D: float,
    ) -> RealVector:
        return -2 * C * D * R * np.exp(-D * R**2)  

    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOne(A={self.A}, B={self.B}, C={self.C}, D={self.D})"
    
    def H(self, t: float, R: RealVector) -> RealOperator:
        return np.sum(self._H_vector(t, R), axis=-1)
    
    def dHdR(self, t: float, R: RealVector) -> RealVectorOperator: 
        dV11dR = self.dV11dR(R, self.A, self.B)
        dV12dR = self.dV12dR(R, self.C, self.D)
        return np.array([[dV11dR, dV12dR], [dV12dR, -dV11dR]])
    
    def _H_vector(self, t: float, R: RealVector) -> RealVectorOperator:
        V11 = self.V11(R, self.A, self.B)
        V12 = self.V12(R, self.C, self.D)
        return np.array([[V11, V12], [np.conj(V12), -V11]])
    
# %% Test the TullyOne class
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pymddrive.models.nonadiabatic_hamiltonian import vectorized_diagonalization
    from pymddrive.models.nonadiabatic_hamiltonian import evaluate_nonadiabatic_couplings
    tully_one = TullyOne()
    R = np.linspace(-10, 10, 1000)
    H = tully_one._H_vector(0.0, R).astype(np.complex128)
    
    evals_list, evecs_list = vectorized_diagonalization(H)
    
    nac_list = np.zeros((H.shape[0], H.shape[0], H.shape[-1], R.shape[0]), dtype=H.dtype)
    F_list = np.zeros((H.shape[0], H.shape[-1], R.shape[0]), dtype=H.dtype)
    for kk in range(H.shape[-1]):
        dHdR = tully_one.dHdR(0.0, np.array([R[kk]])).astype(np.complex128)
        nac_list[..., kk], F_list[..., kk] = evaluate_nonadiabatic_couplings(dHdR, evals_list[:, kk], evecs_list[:, :, kk])
        
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(R, H[0, 0, :], label="H11")
    ax.plot(R, H[1, 1, :], label="H22")
    
    ax.plot(R, evals_list[0, :], label="E1")
    ax.plot(R, evals_list[1, :], label="E2")
    
    # ax.plot(R, -nac_list[0, 1, 0, :]/50, label="NAC12")    
    
    # ax.legend()
    # plt.show()
    
    
        
    
# %%
