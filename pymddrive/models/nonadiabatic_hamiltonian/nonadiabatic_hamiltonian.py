import numpy as np
from numba import njit
from nptyping import NDArray, Shape
# import scipy.sparse as sp

from pymddrive.my_types import RealVector, GenericOperator, GenericVectorOperator
from pymddrive.models.nonadiabatic_hamiltonian.hamiltonian_base import HamiltonianBase as Hamiltonian

from typing import Tuple, Any


def evaluate_hamiltonian(
    t: float, 
    R: RealVector, 
    hamiltonian: Hamiltonian
) -> Tuple[GenericOperator, GenericVectorOperator]:
    H = hamiltonian.H(t, R)
    dHdR = hamiltonian.dHdR(t, R)
    return H, dHdR

@njit
def evaluate_nonadiabatic_couplings(
    dHdR: GenericVectorOperator,
    evals: RealVector,
    evecs: GenericOperator,
) -> Tuple[NDArray[Shape['A, A, B'], Any], NDArray[Shape['A, B'], Any], NDArray[Shape['A, A, B'], Any]]:
    n_elec = dHdR.shape[0]
    n_nucl = dHdR.shape[-1]
    dtype = dHdR.dtype
    
    d = np.zeros((n_elec, n_elec, n_nucl), dtype=dtype)
    F_hellmann_feynman = np.zeros((n_elec, n_elec, n_nucl), dtype=dtype)
    F = np.zeros((n_elec, n_nucl), dtype=dtype)
    
    for kk in range(dHdR.shape[-1]):
        dHdR_kk = np.ascontiguousarray(dHdR[:, :, kk])
        F_hellmann_feynman[:, :, kk] = -np.dot(evecs.T.conjugate(), np.dot(dHdR_kk, evecs))
    
    for ii in range(dHdR.shape[0]):
        F[ii, :] = F_hellmann_feynman[ii, ii, :]
        for jj in range(ii+1, dHdR.shape[0]):
            d[ii, jj, :] = F_hellmann_feynman[ii, jj, :] / (evals[ii] - evals[jj])
            d[jj, ii, :] = -d[ii, jj, :].conjugate()
            
    return d, F, F_hellmann_feynman

def vectorized_diagonalization(
    Hv: GenericVectorOperator,
) -> Tuple[RealVector, GenericOperator]:
    evals_tmp, evecs_tmp = np.linalg.eigh(Hv.transpose(2, 0, 1))
    return evals_tmp.T, evecs_tmp.transpose(1, 2, 0)