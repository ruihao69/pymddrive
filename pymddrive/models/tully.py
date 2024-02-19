# %% The package
import numpy as np
import scipy.linalg as LA
from pymddrive.models.scatter import NonadiabaticHamiltonian

from numpy.typing import ArrayLike
from typing import Any, Union

class TullyOne(NonadiabaticHamiltonian):
    def __init__(
        self,
        A: float = 0.01,
        B: float = 1.6,
        C: float = 0.005,
        D: float = 1.0,
    ) -> None:
        self.A = A 
        self.B = B
        self.C = C
        self.D = D
        
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOne(A={self.A}, B={self.B}, C={self.C}, D={self.D})"
    
    def __call__(
        self, 
        x: Union[float, ArrayLike],
        use_numerical_eigen: bool = False, 
        *args: Any,
        **kwargs: Any,
    ):
        if isinstance(x, float):
            H = self.get_H(x)
            evals, evecs = self.diagonalize(H, use_numerical_eigen)
            dHdx = self.get_dHdx(x)
            d, F = TullyOne.get_nonadiabatic_couplings(dHdx, evecs, evals)
            d12 = d[0, 1]
            
            return H, evals, evecs, d12, F
            
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("The batch mode is not implemented at this time.")
        else:
            raise ValueError("The input x should be either float or numpy.ndarray.")
            
        
    @staticmethod 
    def V11(
        x: Union[float, ArrayLike],
        A: float,
        B: float,
    ):
        sign = np.sign(x)
        return sign * A * (1 - np.exp(-sign * B * x))
    
    @staticmethod
    def V12(
        x: Union[float, ArrayLike],
        C: float,
        D: float,
    ):
        return C * np.exp(-D * x**2)
    
    def get_H(
        self, 
        x: Union[float, ArrayLike]
    ):
        if isinstance(x, float):
            V11 = TullyOne.V11(x, self.A, self.B)
            V22 = -V11
            V12 = V21 = TullyOne.V12(x, self.C, self.D)
            return np.array(
                [[V11, V12], 
                 [V21, V22]]
            )
        elif isinstance(x, np.ndarray):
            V11 = TullyOne.V11(x, self.A, self.B)
            V22 = -V11
            V12 = TullyOne.V12(x, self.C, self.D)
            return np.array(
                [[V11, V12], 
                 [V12, V22]]
            ).transpose(2, 0, 1)
            
    def get_dHdx(self, x: float):
        dV11 = self.A * self.B * np.exp(-np.abs(x) * self.B)
        dV12 = -2 * self.C * self.D * x * np.exp(-self.D * x**2)
        return np.array(
            [[dV11, dV12], 
             [dV12, -dV11]]
        )
            
    def _diagonalize_numerical(self, H):
        return NonadiabaticHamiltonian.diagonalize_numerical(H)
    
    def _diagonalize_analytical(self, H):
        return NonadiabaticHamiltonian.diagonalize_twoD_real_symmetric(H)
         
    def diagonalize(
        self, 
        H: ArrayLike,
        use_numerical_eigen: bool = False,
    ):
        if use_numerical_eigen:
            return self._diagonalize_numerical(H)
        else :
            return self._diagonalize_analytical(H)
        
# %% Temprarory test code

if __name__ == "__main__":
    x = np.linspace(-10, 10, 1000)
    
    tl1 = TullyOne()    
    Eg_out = np.zeros_like(x)
    Ee_out = np.zeros_like(x)
    Fg_out = np.zeros_like(x)
    Fe_out = np.zeros_like(x)
    d12_out = np.zeros_like(x)
    for i, xx in enumerate(x):
        H, evals, evecs, d12, F = tl1(xx, use_numerical_eigen=True)
        Eg_out[i] = evals[0]
        Ee_out[i] = evals[1]
        d12_out[i] = d12
        Fg_out[i] = F[0]
        Fe_out[i] = F[1]
        
    import matplotlib.pyplot as plt
    fig = plt.figure(dpi=200, figsize=(3, 2*2))
    ax_E = fig.add_subplot(211)
    ax_E.plot(x, Eg_out, label="E ground")
    ax_E.plot(x, Ee_out, label="E excited")
    ax_E.plot(x, np.array(d12_out) / 50, label="d12 / 50")
    ax_E.legend()
    ax_E.set_ylabel("Energy (a.u.)")
    ax_E.set_xlabel("x (a.u.)")
    
    ax_F = fig.add_subplot(212)
    ax_F.plot(x, Fg_out, ls='-',label="F ground")
    ax_F.plot(x, Fe_out, ls='--', label="F excited")
    ax_F.legend()
    ax_F.set_ylabel("Adiabatic Forces (a.u.)")
    ax_F.set_xlabel("x (a.u.)")
    
    fig.tight_layout()
    
    fig = plt.figure(dpi=200, figsize=(3, 2))
    ax = fig.add_subplot(111)
    ax.plot(x, d12_out, label="d12")
    ax.legend()
    ax.set_ylabel("Nonadiabatic Coupling (a.u.)")
    ax.set_xlabel("x (a.u.)")    

# %%
