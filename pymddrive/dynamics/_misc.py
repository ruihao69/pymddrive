import numpy as np
from numba import jit   
from typing import Union, Tuple
from numpy.typing import ArrayLike  
from pymddrive.models.scatter import NonadiabaticHamiltonian

def eval_nonadiabatic_hamiltonian(t, R: ArrayLike, model: NonadiabaticHamiltonian) -> Tuple:
    if R.shape[0] == 1:
        _, evals, evecs, d, F = model(t, R[0])
        d = d[np.newaxis, :, :]
        F = F[:, np.newaxis]
    else:
        _, evals, evecs, d, F = model(t, R)
    return evals, evecs, d, F

def rhs_density_matrix(rho, evals, vdotd, k_rho=None):
    if k_rho is None:
        k_rho = np.zeros_like(rho)
    return _rhs_density_matrix(rho=rho, evals=evals, vdotd=vdotd, k_rho=k_rho)

def rhs_wavefunction(c, evals, vdotd, k_c=None):
    if k_c is None:
        k_c = np.zeros_like(c)
    return _rhs_wavefunction(c=c, evals=evals, vdotd=vdotd, k_c=k_c)

def v_dot_d(
     v: ArrayLike,
     d: ArrayLike,
) -> ArrayLike:
     return np.tensordot(v, d, axes=(0, 0))
 
def _expected_value_dm(
    rho: ArrayLike, # density matrix
    O: ArrayLike,   # operator (matrix and/or eigenvalues)
):
    if O.ndim == 1:
        return np.dot(np.diagonal(rho).real, O)
    elif O.ndim == 2:
        return np.trace(np.dot(rho, O)).real
    else:
        raise ValueError("Invalid shape for operator O: {}".format(O.shape))
    
def _expected_value_wf(
    c: ArrayLike, # state coefficients
    O: ArrayLike, # operator (matrix and/or eigenvalues)
):
    if O.ndim == 1:
        return np.dot(c.conj(), O * c).real
    elif O.ndim == 2:
        return np.dot(c.conj(), np.dot(O, c)).real
    else:
        raise ValueError("Invalid shape for operator O: {}".format(O.shape))
    
def expected_value(qm, O):
    if qm.ndim == 2:
        return _expected_value_dm(rho=qm, O=O)
    else:
        return _expected_value_wf(c=qm, O=O)

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