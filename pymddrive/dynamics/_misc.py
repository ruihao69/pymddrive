# %%
import numpy as np
from numba import jit   

from collections import namedtuple

from typing import Union, Tuple
from numbers import Real
from numpy.typing import ArrayLike  

from pymddrive.models.nonadiabatic_hamiltonian import NonadiabaticHamiltonianBase, evaluate_hamiltonian, evaluate_nonadiabatic_couplings
from pymddrive.dynamics.options import (
    BasisRepresentation, QunatumRepresentation, 
    NonadiabaticDynamicsMethods, NumericalIntegrators
)

HamiltonianRetureType = namedtuple('HamiltonianRetureType', 'H, dHdR, evals, evecs, d, F')

def eval_nonadiabatic_hamiltonian(
    t: float, R: ArrayLike, hamiltonian: NonadiabaticHamiltonianBase, 
    basis_rep: BasisRepresentation=BasisRepresentation.Adiabatic,
    eval_deriv_cp: bool=False,
) -> HamiltonianRetureType: 
    flag_reshape = False
    if R.shape[0] == 1:
        H, dHdR, evals, evecs = evaluate_hamiltonian(t, R[0], hamiltonian)
        flag_reshape = True
    else:
        H, dHdR, evals, evecs = evaluate_hamiltonian(t, R, hamiltonian)
        
    if basis_rep == BasisRepresentation.Adiabatic or eval_deriv_cp:
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
    else:
        d, F = None, None
        
    if flag_reshape:
        dHdR = dHdR[:, :, np.newaxis]
        d = d[np.newaxis, :, :] if d is not None else None
        # d = d[np.newaxis, :, :] if d is not None
        F = F[:, np.newaxis] if F is not None else None
        
    return HamiltonianRetureType(H=H, dHdR=dHdR, evals=evals, evecs=evecs, d=d, F=F)

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
# %%
