# %% The package
import numpy as np
import scipy.linalg as LA
from numba import jit

from numpy.typing import ArrayLike
from typing import Any, Union

class NonadiabaticHamiltonian:
    def __init__(
        self,
        ndim: int = 2,
    ) -> None:
        self.dim = ndim
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
        
    @staticmethod
    def diagonalize_numerical(
        H: ArrayLike,
        use_lapack: Union[bool, None] = None,
        enforce_gauge: bool = True,
    ):
        if use_lapack is None:
            if H.shape[0] > 24:
                use_lapack = False
            else:
                use_lapack = True
                
        if use_lapack:
            if H.dtype == np.float64:
                evals, evecs = LA.lapack.dsyev(H)[:2]
                # return LA.lapack.dsyev(H)[:2]
            elif H.dtype == np.complex128:
                evals, evecs = LA.lapack.zheev(H)[:2]
                # return LA.lapack.zheev(H)[:2]
        else: 
            evals, evecs = LA.eigh(H)
            # return LA.eigh(H)
            
        if enforce_gauge:
            if np.iscomplexobj(evecs):
                negate_phase = np.exp(-1j * np.angle(evecs.diagonal()))
                evecs *= negate_phase
            else:
                sign = np.sign(evecs.diagonal())
                evecs *= sign
                
        return evals, evecs
    
    @staticmethod
    def diagonalize_twoD_real_symmetric(H: ArrayLike):
        a = H[0, 0] 
        b = H[1, 1]
        c = H[0, 1]
        
        lambda1 = 0.5 * (a + b - np.sqrt((a - b)**2 + 4 * c**2))
        lambda2 = 0.5 * (a + b + np.sqrt((a - b)**2 + 4 * c**2))
        
        evals = np.array([lambda1, lambda2])
        theta = np.arctan2(2 * c, b - a) / 2
        evecs = np.array(
            [[ np.cos(theta), np.sin(theta)], 
             [-np.sin(theta), np.cos(theta)]]
        )
        return evals, evecs
        
    
    @staticmethod
    def diabatic_to_adiabatic(
        H: ArrayLike,
        U: ArrayLike,
        out: ArrayLike = None,
    ):
        if out is not None:
            np.dot(H, U, out=out)
            np.dot(U.conj().T, out, out=out)
        else:
            return np.dot(U.conj().T, np.dot(H, U))
        
    @staticmethod
    def get_nonadiabatic_couplings(
        dHdx: ArrayLike,
        evecs: ArrayLike,
        evals: ArrayLike,
        out_d: ArrayLike = None,
        out_F: ArrayLike = None,
    ):
        if out_d is not None and out_F is not None:
            NonadiabaticHamiltonian.diabatic_to_adiabatic(dHdx, evecs, out=out_d)
            out_F[:] = - np.diag(out_d).astype(np.float64)
            np.fill_diagonal(out_d, 0)
        else:
            out_d = NonadiabaticHamiltonian.diabatic_to_adiabatic(dHdx, evecs)
            out_F = - np.diag(out_d).astype(np.float64)
            np.fill_diagonal(out_d, 0)
        if np.iscomplexobj(out_d): 
            NonadiabaticHamiltonian._evaluate_nonadiabatic_couplings_cplx(out_d, evals)  
        else:
            NonadiabaticHamiltonian._evaluate_nonadiabatic_couplings_real(out_d, evals)
        return out_d, out_F
        
    @staticmethod
    @jit(nopython=True)
    def _evaluate_nonadiabatic_couplings_real(
        out_d: ArrayLike,
        evals: ArrayLike,
    ):
        dim = evals.shape[0]
        for ii in range(dim):
            for jj in range(ii+1, dim):
                out_d[ii, jj] /= (evals[jj] - evals[ii])
                out_d[jj, ii] = -out_d[ii, jj] 
    
    @staticmethod
    @jit(nopython=True)
    def _evaluate_nonadiabatic_couplings_cplx(
        out_d: ArrayLike,
        evals: ArrayLike,
    ):
        dim = evals.shape[0]
        for ii in range(dim):
            for jj in range(ii+1, dim):
                out_d[ii, jj] /= (evals[jj] - evals[ii])    
                out_d[jj, ii] = -out_d[ii, jj].conj() 
    

# %% The tepmerary test code
if __name__ == "__main__":
    print("Hello.")
# %%
