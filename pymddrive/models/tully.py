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
        t: float,
        r: Union[float, ArrayLike],
        use_numerical_eigen: bool = False, 
        *args: Any,
        **kwargs: Any,
    ):
        H = self.get_H(t, r)
        evals, evecs = self.diagonalize(H, use_numerical_eigen)
        dHdr = self.get_dHdr(t, r)
        # if isinstance(x, np.ndarray):
        #     # Transpose the dHdx so that the first dimention matches the shape of nuclear coordinates
        #     dHdx = dHdx.transpose(2, 0, 1)
            
        d, F = NonadiabaticHamiltonian.get_nonadiabatic_couplings(dHdr, evecs, evals) 
        
        return H, evals, evecs, d, F
        
    @staticmethod 
    def V11(
        r: Union[float, ArrayLike],
        A: float,
        B: float,
    ):
        sign = np.sign(r)
        return sign * A * (1 - np.exp(-sign * B * r))
    
    @staticmethod
    def V12(
        r: Union[float, ArrayLike],
        C: float,
        D: float,
    ):
        return C * np.exp(-D * r**2)
    
    @staticmethod
    def _H(
        r: Union[float, ArrayLike],
        V11: Union[float, ArrayLike],
        V12: Union[float, ArrayLike],
    ) -> ArrayLike:
        if isinstance(r, float):
            return np.array(
                [[V11, V12], 
                 [V12, -V11]]
            )
        elif isinstance(r, np.ndarray):
            try:
                return np.sum(np.array(
                    [[V11, V12], 
                     [V12, -V11]]
                ), axis=-1)
            except:
                print(r, type(r))
                print(V11, type(V11))
                print(V12, type(V12))
                raise RuntimeError
        else:
            raise ValueError("The input r should be either float or numpy.ndarray.")
        
    @staticmethod
    def _dHdr(dV11: Union[float, ArrayLike], dV12: Union[float, ArrayLike]) -> ArrayLike:   
        return np.array(
            [[dV11, dV12], 
             [dV12, -dV11]]
        )
      
    def get_H(
        self, 
        t: float,
        r: Union[float, ArrayLike]
    ):
        _ = t
        V11 = TullyOne.V11(r, self.A, self.B)
        V12 = TullyOne.V12(r, self.C, self.D)
        return self._H(r, V11, V12)
            
    def get_dHdr(self, t: float, r: Union[float, ArrayLike]):
        _ = t
        dV11 = self.A * self.B * np.exp(-np.abs(r) * self.B)
        dV12 = -2 * self.C * self.D * r * np.exp(-self.D * r**2)
        return TullyOne._dHdr(dV11, dV12)
            
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
        else:
            return self._diagonalize_analytical(H)
        
# %% Temprarory test code

if __name__ == "__main__":
    
    # Testing scalar input
    t = t0 = 0.0
    x = np.linspace(-10, 10, 1000)
    tl1 = TullyOne()    
    Eg_out = np.zeros_like(x)
    Ee_out = np.zeros_like(x)
    Fg_out = np.zeros_like(x)
    Fe_out = np.zeros_like(x)
    d12_out = np.zeros_like(x)
    for i, xx in enumerate(x):
        H, evals, evecs, d12, F = tl1(t, xx, use_numerical_eigen=True)
        Eg_out[i] = evals[0]
        Ee_out[i] = evals[1]
        d12_out[i] = d12[0, 1]
        Fg_out[i] = F[0]
        Fe_out[i] = F[1]
        
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')
    fig = plt.figure(dpi=300, figsize=(3.5, 2.12*2))
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
    
    # Testing array input
    x = np.linspace(-10, 10, 1000).reshape(-1, 1)
    tl1 = TullyOne()    
    Eg_out = np.zeros(x.shape[0])
    Ee_out = np.zeros(x.shape[0])
    Fg_out = np.zeros(x.shape[0])
    Fe_out = np.zeros(x.shape[0])
    d12_out = np.zeros(x.shape[0])
    for i, xx in enumerate(x):
        H, evals, evecs, d12, F = tl1(t, xx, use_numerical_eigen=True)
        Eg_out[i] = evals[0]
        Ee_out[i] = evals[1]
        d12_out[i] = d12[0, 0, 1]
        Fg_out[i] = F[0, 0]
        Fe_out[i] = F[1, 0]
        
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
    
    import time
    
    def benchmark_eval(x, N=10000):
        tl1 = TullyOne()    
        Eg_out = np.zeros(x.shape[0])
        Ee_out = np.zeros(x.shape[0])
        Fg_out = np.zeros(x.shape[0])
        Fe_out = np.zeros(x.shape[0])
        d12_out = np.zeros(x.shape[0])
        start = time.time() 
        for i, xx in enumerate(x):
            H, evals, evecs, d12, F = tl1(t, xx, use_numerical_eigen=True)
            # Eg_out[i] = evals[0]
            # Ee_out[i] = evals[1]
            # d12_out[i] = d12[0, 0, 1]
            # Fg_out[i] = F[0, 0]
            # Fe_out[i] = F[1, 0]
            
        time_elapsed = time.time() - start
        return time_elapsed
    Ntest = 100000 
    x = np.random.normal(0, 10, Ntest)
    print(f"Elapsed time for {Ntest} scalar evaluations: {benchmark_eval(x, Ntest):.3f} s")
    print(f"Elapsed time for {Ntest} array evaluations: {benchmark_eval(x.reshape(-1, 1), Ntest):.3f} s")
        
# %%
