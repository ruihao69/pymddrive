import numpy as np
from numba import njit
# import scipy.sparse as sp

from pymddrive.my_types import RealVector, GenericOperator, GenericVectorOperator, GenericDiagonalVectorOperator
from pymddrive.models.nonadiabatic_hamiltonian.hamiltonian_base import HamiltonianBase as Hamiltonian
from pymddrive.models.nonadiabatic_hamiltonian.math_utils import diabatic_to_adiabatic, diagonalization 

from typing import Tuple, Union, Optional


def evaluate_hamiltonain(
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
) -> Tuple:
    d = np.zeros_like(dHdR)
    F = np.zeros((dHdR.shape[1], dHdR.shape[2]), dtype=dHdR.dtype)
    
    _op = np.zeros((dHdR.shape[0], dHdR.shape[0]), dtype=dHdR.dtype)
    for kk in range(dHdR.shape[-1]):
        _op = np.ascontiguousarray(dHdR[:, :, kk])
        d[:, :, kk] = np.dot(evecs.T.conjugate(), np.dot(_op, evecs))
    
    for ii in range(dHdR.shape[0]):
        F[ii, :] = - d[ii, ii, :]
        d[ii, ii, :] = 0.0
        for jj in range(ii+1, dHdR.shape[0]):
            d[ii, jj, :] = d[ii, jj, :] / (evals[jj] - evals[ii])
            d[jj, ii, :] = -d[ii, jj, :].conjugate() 
    return d, F