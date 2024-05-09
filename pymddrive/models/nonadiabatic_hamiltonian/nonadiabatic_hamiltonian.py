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
) -> Tuple[NDArray[Shape['A, A, B'], Any], NDArray[Shape['A, B'], Any]]:
    d = np.zeros_like(dHdR)
    F = np.zeros((dHdR.shape[1], dHdR.shape[2]), dtype=dHdR.dtype)
    
    _op = np.zeros((dHdR.shape[0], dHdR.shape[0]), dtype=dHdR.dtype)
    for kk in range(dHdR.shape[-1]):
        _op[:] = np.ascontiguousarray(dHdR[:, :, kk])
        d[:, :, kk] = np.dot(evecs.T.conjugate(), np.dot(_op, evecs))
    
    for ii in range(dHdR.shape[0]):
        F[ii, :] = - d[ii, ii, :]
        d[ii, ii, :] = 0.0
        for jj in range(ii+1, dHdR.shape[0]):
            d[ii, jj, :] = d[ii, jj, :] / (evals[jj] - evals[ii])
            d[jj, ii, :] = -d[ii, jj, :].conjugate() 
    return d, F

def vectorized_diagonalization(
    Hv: GenericVectorOperator,
) -> Tuple[RealVector, GenericOperator]:
    evals_tmp, evecs_tmp = np.linalg.eigh(Hv.transpose(2, 0, 1))
    return evals_tmp.T, evecs_tmp.transpose(1, 2, 0)