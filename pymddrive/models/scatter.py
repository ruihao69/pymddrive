# %% The package
import numpy as np
import scipy.linalg as LA
from numba import jit

from numpy.typing import ArrayLike
from typing import Any, Union, Tuple

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
    def _evaluate_nonadiabatic_couplings_scalar(
        out_d: ArrayLike,
        evals: ArrayLike
    ) -> None:
        if np.iscomplexobj(evals):
            _evaluate_nonadiabatic_couplings_cplx(out_d, evals)
        else:
            _evaluate_nonadiabatic_couplings_real(out_d, evals)
    
    @staticmethod
    def _evaluate_nonadiabatic_couplings_array(
        out_d: ArrayLike,
        evals: ArrayLike
    ) -> None:
        if np.iscomplexobj(evals):
            _evaluate_nonadiabatic_couplings_cplx_array(out_d, evals)
        else:
            _evaluate_nonadiabatic_couplings_real_array(out_d, evals)
            
    @staticmethod
    def get_nonadiabatic_couplings(
        dHdx: ArrayLike,
        evecs: ArrayLike,
        evals: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike]:
        if (dHdx.ndim == 2): 
            return _get_nonadiabatic_couplings_scalar(dHdx, evecs, evals)
        elif dHdx.ndim == 3:
            return _get_nonadiabatic_couplings_array(dHdx, evecs, evals)
        else:
            raise ValueError("The input dHdx must be either 2D or 3D.")
            
@jit(nopython=True) 
def _evaluate_nonadiabatic_couplings_real(
    out_d: ArrayLike,
    evals: ArrayLike
) -> None:
    dim = evals.shape[0]
    for ii in range(dim):
        out_d[ii, ii] = 0
        for jj in range(ii+1, dim):
            out_d[ii, jj] /= (evals[jj] - evals[ii])
            out_d[jj, ii] = -out_d[ii, jj]
            
@jit(nopython=True)
def _evaluate_nonadiabatic_couplings_cplx(
    out_d: ArrayLike,
    evals: ArrayLike
) -> None:
    dim = evals.shape[0]
    for ii in range(dim):
        out_d[ii, ii] = 0
        for jj in range(ii+1, dim):
            out_d[ii, jj] /= (evals[jj] - evals[ii])
            out_d[jj, ii] = -out_d[ii, jj].conj()
            
@jit(nopython=True) 
def _evaluate_nonadiabatic_couplings_real_array(
    out_d: ArrayLike,
    evals: ArrayLike
) -> None:
    dim_qm = evals.shape[0]
    dim_cl = out_d.shape[0]
    for ii in range(dim_cl):
        for jj in range(dim_qm):
            out_d[ii, jj, jj] = 0
            for kk in range(jj+1, dim_qm):
                out_d[ii, jj, kk] /= (evals[kk] - evals[jj])
                out_d[ii, kk, jj] = -out_d[ii, jj, kk]
            
@jit(nopython=True)
def _evaluate_nonadiabatic_couplings_cplx_array(
    out_d: ArrayLike,
    evals: ArrayLike
) -> None:
    dim_qm = evals.shape[0]
    dim_cl = out_d.shape[0]
    for ii in range(dim_cl):
        for jj in range(dim_qm):
            out_d[ii, jj, jj] = 0
            for kk in range(jj+1, dim_qm):
                out_d[ii, jj, kk] /= (evals[kk] - evals[jj])
                out_d[ii, kk, jj] = -out_d[ii, jj, kk].conj()

def _get_nonadiabatic_couplings_scalar(
    dHdx: ArrayLike,
    evecs: ArrayLike,
    evals: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    out_d = NonadiabaticHamiltonian.diabatic_to_adiabatic(dHdx, evecs)
    out_F = - np.diagonal(out_d).astype(np.float64)
    NonadiabaticHamiltonian._evaluate_nonadiabatic_couplings_scalar(out_d, evals)
    return out_d, out_F

def _get_nonadiabatic_couplings_array(
    dHdx: ArrayLike,
    evecs: ArrayLike,
    evals: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    out_d = np.array(tuple(
        np.dot(evecs.conj().T, np.dot(dHdx[:, :, kk], evecs)) for kk in range(dHdx.shape[-1])
    ))
    # out_d = np.einsum('ji,jkn,kl->nil', evecs.conj(), dHdx, evecs)
    out_F = -np.diagonal(out_d, axis1=1, axis2=2).T.astype(np.float64)
    NonadiabaticHamiltonian._evaluate_nonadiabatic_couplings_array(out_d, evals)
    return out_d, out_F
            

# %% The tepmerary test code
if __name__ == "__main__":
    print("Hello.")
# %%
