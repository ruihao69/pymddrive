import numpy as np
import numpy.linalg as LA
from numpy.typing import ArrayLike
from numba import jit

from typing import Union

def rhs_density_matrix(rho: ArrayLike, evals: ArrayLike, vdotd: ArrayLike, k_rho:Union[ArrayLike, None]=None):
    # if k_rho is None:
    #     k_rho = np.zeros_like(rho)
    # return _rhs_density_matrix(rho=rho, evals=evals, vdotd=vdotd, k_rho=k_rho)
    if k_rho is None:
        k_rho = np.zeros_like(rho)
    H_eff = np.diag(evals) - 1.0j * vdotd
    k_rho[:] = -1.0j * commutator(H_eff, rho)
    return k_rho

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
    # return Op1 @ Op2 - Op2 @ Op1
 
def _expected_value_dm(
    rho: ArrayLike, # density matrix
    O: ArrayLike,   # operator (matrix and/or eigenvalues)
    is_diagonal: bool=False,
):
    if O.ndim == 2:
        return np.trace(np.dot(rho, O)).real
    elif O.ndim == 3:
        return [np.trace(np.dot(rho, O[i, :, :])).real for i in range(O.shape[0])]
    elif is_diagonal:
        return np.dot(rho.diagonal().real, O)
    else:
        raise ValueError("Invalid shape for operator O: {}".format(O.shape))
    
def _expected_value_wf(
    c: ArrayLike, # state coefficients
    O: ArrayLike, # operator (matrix and/or eigenvalues)
    is_diagonal: bool=False,
):
    if O.ndim == 2:
        return np.dot(c.conj().T, np.dot(O, c)).real
    elif O.ndim == 3:
        return [np.dot(c.conj().T, np.dot(O[i, :, :], c)).real for i in range(O.shape[0])]
    elif is_diagonal:
        return np.dot(c.conj(), O * c).real
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

# fix the phase factor for the non-adiabatic couplings
def nac_phase_following(prev_d, curr_d): # the wrapper function
    assert prev_d.ndim == curr_d.ndim
    assert prev_d.shape[0] == curr_d.shape[0], f"The classical dimension of the previous and current derivatives are different: {prev_d.shape[0]} != {curr_d.shape[0]}"
    assert prev_d.shape[1] == prev_d.shape[2] == curr_d.shape[1] == curr_d.shape[2], f"The quantum dimension of the previous and current derivatives are different: {prev_d.shape[1:]} != {curr_d.shape[1:]}"
    corrected_d = np.zeros_like(prev_d)
    return _nac_phase_following(prev_d, curr_d, corrected_d)

@jit(nopython=True)
def _nac_phase_following(prev_d, curr_d, corrected_d): # heavy lifting using numba
    # dim_cl = prev_d.shape[0]
    dim_elec = prev_d.shape[1]
    for jj in range(dim_elec):
        for kk in range(dim_elec):
            if jj == kk:
                continue
            prev_nac_norm = LA.norm(prev_d[:, jj, kk])
            curr_nac_norm = LA.norm(curr_d[:, jj, kk])
            nac_dot_product = np.dot(prev_d[:, jj, kk], curr_d[:, jj, kk])
            if (prev_nac_norm == 0) or (curr_nac_norm == 0):
                phase_factor = 1.0
            else:
                phase_factor = nac_dot_product / (prev_nac_norm * curr_nac_norm)
            corrected_d[:, jj, kk] = np.multiply(curr_d[:, jj, kk], phase_factor)
    return corrected_d