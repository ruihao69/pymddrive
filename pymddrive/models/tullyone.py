# %% Use tully one to test the nonadiabatic hamiltonian abc class
import numpy as np
import scipy.sparse as sp

from enum import Enum, unique

from typing import Tuple, Union
from numbers import Real
from numpy.typing import ArrayLike

from pymddrive.models.nonadiabatic_hamiltonian import (
    NonadiabaticHamiltonian, TD_NonadiabaticHamiltonian, FloquetHamiltonian
)
from pymddrive.models.nonadiabatic_hamiltonian import (
    diagonalize_hamiltonian, evaluate_nonadiabatic_couplings
)

from pymddrive.pulses.pulses import Pulse
from pymddrive.pulses.morlet import MorletReal
from pymddrive.models.floquet import FloquetType

class TullyOne(NonadiabaticHamiltonian):
    def __init__(
        self,
        A: Real = 0.01,
        B: Real = 1.6,
        C: Real = 0.005,
        D: Real = 1.0,
    ) -> None:
        super().__init__(dim=2)
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
    @staticmethod
    def V11(
        r: Union[Real, ArrayLike],
        A: Real,
        B: Real,
    ) -> Union[Real, ArrayLike]:
        sign = np.sign(r)
        return sign * A * (1 - np.exp(-sign * B * r))
    
    @staticmethod
    def V12(
        r: Union[Real, ArrayLike],
        C: Real,
        D: Real,
    ) -> Union[Real, ArrayLike]:
        return C * np.exp(-D * r**2)
    
    @staticmethod
    def V11dR(
        r: Union[Real, ArrayLike],
        A: Real,
        B: Real,
    ) -> Union[Real, ArrayLike]:
        return A * B * np.exp(-np.abs(r) * B) 
    
    @staticmethod
    def V12dR(
        r: Union[Real, ArrayLike],
        C: Real,
        D: Real,
    ) -> Union[Real, ArrayLike]:
        return -2 * C * D * r * np.exp(-D * r**2)  

    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOne(A={self.A}, B={self.B}, C={self.C}, D={self.D})"
    
    def H(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        V11 = self.V11(r, self.A, self.B)
        V12 = self.V12(r, self.C, self.D)
        return _construct_2D_H(r, V11, V12, -V11)
    
    def dHdR(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        dV11dR = self.V11dR(r, self.A, self.B)
        dV12dR = self.V12dR(r, self.C, self.D)
        return np.array([[dV11dR, dV12dR], [dV12dR, -dV11dR]])
    
    def __call__(self, 
        t: Real, r: Union[Real, ArrayLike],
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        H = self.H(t, r)
        evals, evecs = diagonalize_hamiltonian(H) 
        dHdR = self.dHdR(t, r)
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        return H, evals, evecs, d, F
    
class TullyOneTD_type1(TD_NonadiabaticHamiltonian):
    def __init__(
        self,
        A: Real = 0.01,
        B: Real = 1.6,
        C: Real = 0.005,
        D: Real = 1.0,
        pulse: Pulse = Pulse(),
    ) -> None:
        super().__init__(dim=2, pulse=pulse)
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOneTD_type1(A={self.A}, B={self.B}, C={self.C}, D={self.D}, pulse={self.pulse})" 
    
    def H0(self, r: Union[Real, ArrayLike]) -> ArrayLike:
        V11 = TullyOne.V11(r, self.A, self.B)
        V12 = TullyOne.V12(r, self.C, self.D)
        return _construct_2D_H(r, V11, V12, -V11)
    
    def H1(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        V12 = self.pulse(t) * np.ones_like(r)
        return _construct_2D_H(r, np.zeros_like(r), V12, np.zeros_like(r))
    
    def dH0dR(self, r: Union[Real, ArrayLike]) -> ArrayLike:
        dV11dR = TullyOne.V11dR(r, self.A, self.B)
        dV12dR = TullyOne.V12dR(r, self.C, self.D)  
        return np.array([[dV11dR, dV12dR], [dV12dR, -dV11dR]])
    
    def dH1dR(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        return np.zeros((2, 2)) if isinstance(r, Real) else np.zeros((2, 2, len(r)))
    
    def __call__(self, 
        t: Real, r: Union[Real, ArrayLike],
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        H = self.H(t, r)
        evals, evecs = diagonalize_hamiltonian(H) 
        dHdR = self.dHdR(t, r)
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        return H, evals, evecs, d, F
    
class TullyOneTD_type2(TD_NonadiabaticHamiltonian):
    def __init__(
        self,
        A: Real = 0.01,
        B: Real = 1.6,
        C: Real = 0.005,
        D: Real = 1.0,
        pulse: Pulse = Pulse(),
    ) -> None:
        super().__init__(dim=2, pulse=pulse) 
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOneTD_type1(A={self.A}, B={self.B}, C={self.C}, D={self.D}, pulse={self.pulse})" 
    
    def H0(self, r: Union[Real, ArrayLike]) -> ArrayLike:
        V11 = TullyOne.V11(r, self.A, self.B)
        # V12 = TullyOne.V12(r, self.C, self.D)
        return _construct_2D_H(r, V11, np.zeros_like(V11), -V11)
    
    def H1(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        V12 = self.pulse(t) * TullyOne.V12(r, self.C, self.D)
        return _construct_2D_H(r, np.zeros_like(r), V12, np.zeros_like(r))
    
    def dH0dR(self, r: Union[Real, ArrayLike]) -> ArrayLike:
        dV11dR = TullyOne.V11dR(r, self.A, self.B)
        # dV12dR = TullyOne.V12dR(r, self.C, self.D)  
        return np.array([[dV11dR, np.zeros_like(dV11dR)], [np.zeros_like(dV11dR), -dV11dR]])
    
    def dH1dR(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        dV12dR = TullyOne.V12dR(r, self.C, self.D) * self.pulse(t)
        return np.array([[np.zeros_like(dV12dR), dV12dR], [dV12dR, np.zeros_like(dV12dR)]])
    
    def __call__(self, 
        t: Real, r: Union[Real, ArrayLike],
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        H = self.H(t, r)
        evals, evecs = diagonalize_hamiltonian(H) 
        dHdR = self.dHdR(t, r)
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        return H, evals, evecs, d, F
    
class TullyOneFloquet_type1(FloquetHamiltonian):
    def __init__(
        self, 
        A: Real = 0.01,
        B: Real = 1.6,
        C: Real = 0.005,
        D: Real = 1.0,
        pulse: Pulse = Pulse(),
        NF: int = 5,
        Omega: Union[float, None] = None,
        floquet_type: FloquetType = FloquetType.COSINE
    ) -> None:
        super().__init__(dim=2, pulse=pulse, NF=NF, Omega=Omega, floquet_type=floquet_type)
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOneFloquet_type1(A={self.A}, B={self.B}, C={self.C}, D={self.D}, pulse={self.pulse}, NF={self.NF}, Omega={self.Omega}, floquet_type={self.floquet_type})" 
    
    def H0(self, r: Union[Real, ArrayLike]) -> ArrayLike:
        V11 = TullyOne.V11(r, self.A, self.B)
        V12 = TullyOne.V12(r, self.C, self.D)
        return _construct_2D_H(r, V11, V12, -V11)
    
    def H1(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        V12 = self.pulse(t) * np.ones_like(r)
        return _construct_2D_H(r, np.zeros_like(r), V12, np.zeros_like(r))
    
    def dH0dR(self, r: Union[Real, ArrayLike]) -> ArrayLike:
        dV11dR = TullyOne.V11dR(r, self.A, self.B)
        dV12dR = TullyOne.V12dR(r, self.C, self.D)
        return np.array([[dV11dR, dV12dR], [dV12dR, -dV11dR]])
    
    def dH1dR(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        return np.zeros((2, 2)) if isinstance(r, Real) else np.zeros((2, 2, len(r)))
    
    def __call__(self, 
        t: Real, r: Union[Real, ArrayLike],
    ) -> Tuple[sp.csr_matrix, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        HF: sp.csr_matrix  = self.H(t, r)
        evals, evecs = diagonalize_hamiltonian(HF) 
        dHdR: sp.csr_matrix = self.dHdR(t, r)
        d, F = evaluate_nonadiabatic_couplings(dHdR.toarray(), evals, evecs)
        return HF, evals, evecs, d, F 
 
    
class TullyOneFloquet_type2(FloquetHamiltonian):
    def __init__(
        self, 
        A: Real = 0.01,
        B: Real = 1.6,
        C: Real = 0.005,
        D: Real = 1.0,
        pulse: Pulse = Pulse(),
        NF: int = 5,
        Omega: Union[float, None] = None,
        floquet_type: FloquetType = FloquetType.COSINE
    ) -> None:
        super().__init__(dim=2, pulse=pulse, NF=NF, Omega=Omega, floquet_type=floquet_type)
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOneFloquet_type2(A={self.A}, B={self.B}, C={self.C}, D={self.D}, pulse={self.pulse}, NF={self.NF}, Omega={self.Omega}, floquet_type={self.floquet_type})" 
    
    def H0(self, r: Union[Real, ArrayLike]) -> ArrayLike:
        V11 = TullyOne.V11(r, self.A, self.B)
        # V12 = TullyOne.V12(r, self.C, self.D)
        return _construct_2D_H(r, V11, np.zeros_like(V11), -V11)
    
    def H1(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        V12 = self.pulse(t) * TullyOne.V12(r, self.C, self.D)
        return _construct_2D_H(r, np.zeros_like(r), V12, np.zeros_like(r))
    
    def dH0dR(self, r: Union[Real, ArrayLike]) -> ArrayLike:
        dV11dR = TullyOne.V11dR(r, self.A, self.B)
        # dV12dR = TullyOne.V12dR(r, self.C, self.D)
        return np.array([[dV11dR, np.zeros_like(dV11dR)], [np.zeros_like(dV11dR), -dV11dR]])
    
    def dH1dR(self, t: Real, r: Union[Real, ArrayLike]) -> ArrayLike:
        dV12dR = TullyOne.V12dR(r, self.C, self.D) * self.pulse(t)
        return np.array([[np.zeros_like(dV12dR), dV12dR], [dV12dR, np.zeros_like(dV12dR)]])
    
    def __call__(self, 
        t: Real, r: Union[Real, ArrayLike],
    ) -> Tuple[sp.csr_matrix, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        HF: sp.csr_matrix  = self.H(t, r)
        evals, evecs = diagonalize_hamiltonian(HF) 
        dHdR: sp.csr_matrix = self.dHdR(t, r)
        d, F = evaluate_nonadiabatic_couplings(dHdR.toarray(), evals, evecs)
        return HF, evals, evecs, d, F

def _construct_2D_H(
    r: Union[Real, ArrayLike],
    V11: Union[Real, ArrayLike],
    V12: Union[Real, ArrayLike],
    V22: Union[Real, ArrayLike],
) -> ArrayLike:
    if isinstance(r, Real):
        return np.array([[V11, V12], [np.conj(V12), V22]])
    elif isinstance(r, np.ndarray):
        try:
            return np.sum(np.array([[V11, V12], [V12.conj(), V22]]), axis=-1)
        except ValueError:
            raise ValueError(f"The input array 'r' must be either a number or a 1D array. 'r' input here has dimension of {r.ndim}.")
    else:
        raise NotImplemented
   
class TullyOnePulseTypes(Enum): 
    NO_PULSE = "NoPulse"
    PULSE_TYPE1 = "PulseType1"
    PULSE_TYPE2 = "PulseType2"
    PULSE_TYPE3 = "PulseType3"
    
class TD_Methods(Enum):
    BRUTE_FORCE = "BruteForce"
    FLOQUET = "Floquet"

def get_tullyone(
    A: Real = 0.01, B: Real = 1.6, C: Real = 0.005, D: Real = 1.0, # Tully parameters
    t0: Union[Real, None] = None, Omega: Union[Real, None] = None, 
    tau: Union[Real, None] = None, # pulse parameters
    pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.NO_PULSE,
    td_method: TD_Methods = TD_Methods.BRUTE_FORCE,
):
    if pulse_type == TullyOnePulseTypes.NO_PULSE:
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOne(A=A, B=B, C=C, D=D)
        else:
            raise ValueError(f"You are trying to get a floquet model for a time independent Hamiltonian.")
    else:
        if (t0 is None) or (Omega is None) or (tau is None):
            raise ValueError(f"You need to provide the pulse parameters t0, Omega, and tau for Time-dependent problems.")
    
    if pulse_type == TullyOnePulseTypes.PULSE_TYPE1:
        pulse = MorletReal(A=C, t0=t0, tau=tau, Omega=Omega, phi=0)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=0, D=0, pulse=pulse)
        elif td_method == TD_Methods.FLOQUET:
            return TullyOneFloquet_type1(A=A, B=B, C=0, D=0, pulse=pulse)
        else:
            raise ValueError(f"Invalid TD method: {td_method}")
        
    elif pulse_type == TullyOnePulseTypes.PULSE_TYPE2:
        pulse = MorletReal(A=1.0, t0=t0, tau=tau, Omega=Omega, phi=0)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type2(A=A, B=B, C=C, D=D, pulse=pulse)
        elif td_method == TD_Methods.FLOQUET:
            return TullyOneFloquet_type2(A=A, B=B, C=C, D=D, pulse=pulse)
        else:
            raise ValueError(f"Invalid TD method: {td_method}")
        
    elif pulse_type == TullyOnePulseTypes.PULSE_TYPE3:
        pulse = MorletReal(A=C/2, t0=t0, tau=tau, Omega=Omega, phi=0)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type2(A=A, B=B, C=C/2, D=D, pulse=pulse)
        elif td_method == TD_Methods.FLOQUET:
            return TullyOneFloquet_type2(A=A, B=B, C=C/2, D=D, pulse=pulse)
        else:
            raise ValueError(f"Invalid TD method: {td_method}")
    else:
        raise ValueError(f"Invalid pulse type: {pulse_type}")
        
def _test_main():
    from pymddrive.pulses.pulses import Pulse, UnitPulse, CosinePulse
    
    tullyone = TullyOne()
    r = np.linspace(-10, 10, 1000)
    E0 = np.zeros_like(r)
    E1 = np.zeros_like(r)
    F0 = np.zeros_like(r)
    F1 = np.zeros_like(r)
    d12 = np.zeros_like(r)
    t = 0
    for ii, rr in enumerate(r):
        _, evals, _, d, F = tullyone(t, rr)
        E0[ii], E1[ii] = evals
        F0[ii], F1[ii] = F
        d12[ii] = d[0, 1]
        
    import matplotlib.pyplot as plt
        
    plt.plot(r, E0, label="E0")
    plt.plot(r, E1, label="E1")
    plt.legend()
    plt.show()
    
    plt.plot(r, F0, label="F0")
    plt.plot(r, F1, label="F1")
    plt.legend()
    plt.show()
    
    plt.plot(r, d12, label="d12")
    plt.legend()
    plt.show()
    
    
           
    up = Pulse()
    t = np.linspace(0, 10, 1000)
    
    tullyoneTD1 = TullyOneTD_type1(pulse=up)
    r = np.linspace(-10, 10, 1000)
    E0 = np.zeros_like(r)
    E1 = np.zeros_like(r)
    F0 = np.zeros_like(r)
    F1 = np.zeros_like(r)
    d12 = np.zeros_like(r)
    t = 0
    for ii, rr in enumerate(r):
        _, evals, _, d, F = tullyoneTD1(t, rr)
        E0[ii], E1[ii] = evals
        F0[ii], F1[ii] = F
        d12[ii] = d[0, 1]
    
    plt.plot(r, E0, label="E0")
    plt.plot(r, E1, label="E1")
    plt.legend()
    plt.show()
    
    plt.plot(r, F0, label="F0")
    plt.plot(r, F1, label="F1")
    plt.legend()
    plt.show()
    
    plt.plot(r, d12, label="d12")
    plt.legend()
    plt.show() 
    
    up = UnitPulse() 
    tullyoneTD2 = TullyOneTD_type2(pulse=up)
    r = np.linspace(-10, 10, 1000)
    E0 = np.zeros_like(r)
    E1 = np.zeros_like(r)
    F0 = np.zeros_like(r)
    F1 = np.zeros_like(r)
    d12 = np.zeros_like(r)
    t = 0
    for ii, rr in enumerate(r):
        _, evals, _, d, F = tullyoneTD2(t, rr)
        E0[ii], E1[ii] = evals
        F0[ii], F1[ii] = F
        d12[ii] = d[0, 1]
    
    plt.plot(r, E0, label="E0")
    plt.plot(r, E1, label="E1")
    plt.legend()
    plt.show()
    
    plt.plot(r, F0, label="F0")
    plt.plot(r, F1, label="F1")
    plt.legend()
    plt.show()
    
    plt.plot(r, d12, label="d12")
    plt.legend()
    plt.show() 
    
    cp = CosinePulse(A=0.01, Omega=0.03) 
    tullyoneFloquet1 = TullyOneFloquet_type1(pulse=cp, NF=1)
    dimF = tullyoneFloquet1.get_floquet_space_dim()
    E_out = np.zeros((len(r), dimF))
    F_out = np.zeros((len(r), dimF))
    for ii, rr in enumerate(r):
        _, evals, _, d, F = tullyoneFloquet1(t, rr)
        E_out[ii, :] = evals
        F_out[ii, :] = F
        
    for ii in range(dimF): 
        plt.plot(r, E_out[:, ii], label=f"E{ii}")
    plt.legend()
    plt.show()
    
    for ii in range(dimF): 
        plt.plot(r, F_out[:, ii], label=f"F{ii}")
    plt.legend()
    plt.show()
    
    tol=1e-6 
    d_sparse = sp.csr_matrix(d[np.where(d > tol)]) 
    print(f"Number of non-zero elements in d: {d_sparse.nnz}")
    print(f"d: {d_sparse}")
    
    cp = CosinePulse(A=1.0, Omega=0.03) 
    tullyoneFloquet2 = TullyOneFloquet_type2(pulse=cp, NF=1)
    dimF = tullyoneFloquet2.get_floquet_space_dim()
    E_out = np.zeros((len(r), dimF))
    F_out = np.zeros((len(r), dimF))
    for ii, rr in enumerate(r):
        _, evals, _, d, F = tullyoneFloquet2(t, rr)
        E_out[ii, :] = evals
        F_out[ii, :] = F
        
    for ii in range(dimF): 
        plt.plot(r, E_out[:, ii], label=f"E{ii}")
    plt.legend()
    plt.show()
    
    for ii in range(dimF): 
        plt.plot(r, F_out[:, ii], label=f"F{ii}")
    plt.legend()
    plt.show()
    
    tol=1e-6 
    d_sparse = sp.csr_matrix(d[np.where(d > tol)]) 
    
    print(f"Number of non-zero elements in d: {d_sparse.nnz}")
    print(f"d: {d_sparse}")
    
    
# %% Test the TullyOne class
if __name__ == "__main__":
    _test_main()
# %%
