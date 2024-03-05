import numpy as np
from numpy.typing import ArrayLike
from numba import jit

from typing import Union

def rhs_density_matrix(rho: ArrayLike, evals: ArrayLike, vdotd: ArrayLike, k_rho:Union[ArrayLike, None]=None):
    if k_rho is None:
        k_rho = np.zeros_like(rho)
    return _rhs_density_matrix(rho=rho, evals=evals, vdotd=vdotd, k_rho=k_rho)

def rhs_wavefunction(c: ArrayLike, evals:ArrayLike, vdotd:ArrayLike, k_c: Union[ArrayLike, None]=None):
    if k_c is None:
        k_c = np.zeros_like(c)
    return _rhs_wavefunction(c=c, evals=evals, vdotd=vdotd, k_c=k_c)

def v_dot_d(
    v: ArrayLike,
    d: ArrayLike,
) -> ArrayLike:
    return np.tensordot(v, d, axes=(0, 0))

def commutator(Op1: ArrayLike, Op2: ArrayLike) -> ArrayLike:
    return np.dot(Op1, Op2) - np.dot(Op2, Op1)
 
def _expected_value_dm(
    rho: ArrayLike, # density matrix
    O: ArrayLike,   # operator (matrix and/or eigenvalues)
    is_diagonal: bool=False,
):
    if is_diagonal:
        return rho.diagonal().real.dot(O)
    elif O.ndim == 1:
        return np.dot(np.diagonal(rho).real, O)
    elif (O.ndim == 2) or ((O.ndim == 3) and (O.shape[0] == O.shape[1])):
        return np.trace(np.dot(rho, O)).real 
    else:
        raise ValueError("Invalid shape for operator O: {}".format(O.shape))
    
def _expected_value_wf(
    c: ArrayLike, # state coefficients
    O: ArrayLike, # operator (matrix and/or eigenvalues)
    is_diagonal: bool=False,
):
    if (is_diagonal) or (O.ndim == 1):
        return np.dot(c.conj(), O * c).real
    elif O.ndim == 2:
        return np.dot(c.conj(), np.dot(O, c)).real
    else:
        raise ValueError("Invalid shape for operator O: {}".format(O.shape))
    
def expected_value(qm: ArrayLike, O: ArrayLike, is_diagonal: bool=False):
    if qm.ndim == 2:
        return _expected_value_dm(rho=qm, O=O, is_diagonal=is_diagonal)
    else:
        return _expected_value_wf(c=qm, O=O, is_diagonal=is_diagonal)

# Equations of motion
@jit(nopython=True)
def _rhs_density_matrix(
    rho: ArrayLike,    # density matrix
    evals: ArrayLike,  # eigenvalues
    vdotd: ArrayLike,  # velocity dot derivative coupling
    k_rho: ArrayLike,  # rhs
) -> ArrayLike:
    for kk in range(rho.shape[0]):
        for jj in range(rho.shape[0]):
            k_rho[kk, jj] += -1.0j * rho[kk, jj] * (evals[kk] - evals[jj])
            for ll in range(rho.shape[0]):
                k_rho[kk, jj] += (-rho[ll, jj] * vdotd[kk, ll] + rho[kk, ll] * vdotd[ll, jj])
    return k_rho
                
@jit(nopython=True)
def _rhs_wavefunction(
    c: ArrayLike,      # state coefficients
    evals: ArrayLike,  # eigenvalues
    vdotd: ArrayLike,  # velocity dot derivative coupling
    k_c: ArrayLike,    # rhs
) -> ArrayLike:
    for kk in range(c.shape[0]):
        k_c[kk] += -1.0j * c[kk] * evals[kk]
        for ll in range(c.shape[0]):
            k_c[kk] += -c[ll] * vdotd[kk, ll] 
    return k_c
