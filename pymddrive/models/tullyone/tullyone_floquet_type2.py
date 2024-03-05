import numpy as np
from numpy.typing import ArrayLike

from pymddrive.models.floquet import FloquetType
from pymddrive.models.nonadiabatic_hamiltonian import QuasiFloquetHamiltonianBase
from pymddrive.pulses import PulseBase as Pulse
from pymddrive.pulses import ZeroPulse
from pymddrive.models.tullyone.tullyone import TullyOne, _construct_2D_H

from typing import Union
from numbers import Real

class TullyOneFloquet_type2(QuasiFloquetHamiltonianBase):
    def __init__(
        self, 
        A: Real = 0.01,
        B: Real = 1.6,
        C: Real = 0.005,
        D: Real = 1.0,
        orig_pulse: Pulse = ZeroPulse(),
        floq_pulse: Pulse = ZeroPulse(),
        NF: int = 5,
        Omega: Union[float, None] = None,
        floquet_type: FloquetType = FloquetType.COSINE
    ) -> None:
        super().__init__(dim=2, NF=NF, orig_pulse=orig_pulse, floq_pulse=floq_pulse, Omega=Omega, floquet_type=floquet_type)
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOneFloquet_type2(A={self.A}, B={self.B}, C={self.C}, D={self.D}, pulse={self.pulse}, NF={self.NF}, Omega={self.Omega}, floquet_type={self.floquet_type})" 
    
    def H0(self, r: Union[Real, ArrayLike]) -> ArrayLike:
        V11 = TullyOne.V11(r, self.A, self.B)
        return _construct_2D_H(r, V11, np.zeros_like(V11), -V11)
    
    def H1(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        V12 = self.pulse(t) * TullyOne.V12(r, self.C, self.D)
        return _construct_2D_H(r, np.zeros_like(r), V12, np.zeros_like(r))
    
    def dH0dR(self, r: Union[Real, ArrayLike]) -> ArrayLike:
        dV11dR = TullyOne.dV11dR(r, self.A, self.B)
        return np.array([[dV11dR, np.zeros_like(dV11dR)], [np.zeros_like(dV11dR), -dV11dR]])
    
    def dH1dR(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        dV12dR = TullyOne.dV12dR(r, self.C, self.D) * self.pulse(t)
        return np.array([[np.zeros_like(dV12dR), dV12dR], [dV12dR, np.zeros_like(dV12dR)]])

