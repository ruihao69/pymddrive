import numpy as np
from numba import njit
from nptyping import NDArray, Shape
# import scipy.sparse as sp

from pymddrive.my_types import RealVector, GenericOperator, GenericVectorOperator, RealVectorOperator, RealOperator
from pymddrive.models.nonadiabatic_hamiltonian.hamiltonian_base import HamiltonianBase as Hamiltonian
from pymddrive.models.nonadiabatic_hamiltonian.td_hamiltonian_base import TD_HamiltonianBase
from pymddrive.models.nonadiabatic_hamiltonian.quasi_floquet_hamiltonian_base import QuasiFloquetHamiltonianBase
from pymddrive.models.floquet.floquet import get_grad_HF_Et_upper

from typing import Tuple, Any, Union


def evaluate_hamiltonian(
    t: float, 
    R: RealVector, 
    hamiltonian: Hamiltonian
) -> Tuple[GenericOperator, GenericVectorOperator]:
    H = hamiltonian.H(t, R)
    dHdR = hamiltonian.dHdR(t, R)
    return H, dHdR

def evaluate_pulse_NAC(
    t: float,
    R: RealVector,
    hamiltonian: Union[TD_HamiltonianBase, QuasiFloquetHamiltonianBase],
    evals: RealVector,
    evecs: GenericOperator,
) -> GenericOperator:
    H1 = hamiltonian.H1(t, R)
    if isinstance(hamiltonian, TD_HamiltonianBase):
        pulse_value = hamiltonian.pulse(t)
        pulse_gradient = hamiltonian.pulse.gradient(t)
        grad_H = H1 / pulse_value
        return evaluate_pulse_NAC_TD(pulse_gradient, grad_H, evals, evecs)
    elif isinstance(hamiltonian, QuasiFloquetHamiltonianBase):
        pulse_value = hamiltonian.envelope_pulse(t)
        pulse_gradient = hamiltonian.envelope_pulse.gradient(t)
        grad_H1 = H1 / pulse_value
        grad_H_upper = get_grad_HF_Et_upper(hamiltonian.floquet_type, grad_H1, hamiltonian.NF)
        return evaluate_pulse_NAC_TD(pulse_gradient, grad_H_upper, evals, evecs) + evaluate_pulse_NAC_TD(np.conjugate(pulse_gradient), grad_H_upper.T.conjugate(), evals, evecs)
        
        
def evaluate_pulse_NAC_TD(
    pulse_gradient: float, # note: the current implementation don't support complex pulse value, yet
    grad_H: RealOperator,
    evals: RealVector,
    evecs: RealOperator,
) -> GenericOperator:
    if np.iscomplexobj(grad_H) or np.iscomplexobj(evecs):
        return evaluate_pulse_NAC_TD_complex(pulse_gradient, grad_H, evals, evecs)
    else:
        return evaluate_pulse_NAC_TD_real(pulse_gradient, grad_H, evals, evecs)

@njit
def evaluate_pulse_NAC_TD_real(
    pulse_gradient: float, # note: the current implementation don't support complex pulse value, yet
    grad_H: RealOperator,
    evals: RealVector,
    evecs: RealOperator,
) -> RealOperator:
    nac_pulse = np.dot(evecs.T, np.dot(grad_H, evecs))
    for ii in range(evals.shape[0]):
        nac_pulse[ii, ii] = 0.0
        for jj in range(ii+1, evals.shape[0]):
            nac_pulse[ii, jj] = grad_H[ii, jj] / (evals[ii] - evals[jj]) * pulse_gradient
            nac_pulse[jj, ii] = -nac_pulse[ii, jj]
    return nac_pulse

@njit
def evaluate_pulse_NAC_TD_complex(
    pulse_gradient: float, # note: the current implementation don't support complex pulse value, yet
    grad_H: GenericOperator,
    evals: RealVector,
    evecs: GenericOperator,
) -> RealOperator:
    nac_pulse = np.zeros((evecs.shape[0], evecs.shape[0]), dtype=np.complex128)
    nac_pulse[:] = np.dot(evecs.T.conjugate(), np.dot(grad_H, evecs))
    for ii in range(evals.shape[0]):
        nac_pulse[ii, ii] = 0.0
        for jj in range(ii+1, evals.shape[0]):
            nac_pulse[ii, jj] = grad_H[ii, jj] / (evals[ii] - evals[jj]) * pulse_gradient
            nac_pulse[jj, ii] = -nac_pulse[ii, jj].conjugate()

def evaluate_nonadiabatic_couplings(
    dHdR: GenericVectorOperator,
    evals: RealVector,
    evecs: GenericOperator,
) -> Tuple[NDArray[Shape['A, A, B'], Any], NDArray[Shape['A, B'], Any], NDArray[Shape['A, A, B'], Any]]:
    if np.iscomplexobj(dHdR) or np.iscomplexobj(evecs):
        return evaluate_nonadiabatic_couplings_complex(dHdR, evals, evecs)
    else:
        return evaluate_nonadiabatic_couplings_real(dHdR, evals, evecs)

@njit
def evaluate_nonadiabatic_couplings_real(
    dHdR: RealVectorOperator,
    evals: RealVector,
    evecs: RealOperator,
) -> Tuple[NDArray[Shape['A, A, B'], Any], NDArray[Shape['A, B'], Any], NDArray[Shape['A, A, B'], Any]]:
    n_elec = dHdR.shape[0]
    n_nucl = dHdR.shape[-1]
    F = np.zeros((n_elec, n_nucl), dtype=np.float64)
    d = np.zeros((n_elec, n_elec, n_nucl), dtype=np.float64)
    F_hellmann_feynman = np.zeros((n_elec, n_elec, n_nucl), dtype=np.float64)
    
    for kk in range(dHdR.shape[-1]):
        dHdR_kk = np.ascontiguousarray(dHdR[:, :, kk])
        F_hellmann_feynman[:, :, kk] = -np.dot(evecs.T.conjugate(), np.dot(dHdR_kk, evecs))
    
    for ii in range(dHdR.shape[0]):
        F[ii, :] = np.real(F_hellmann_feynman[ii, ii, :])
        for jj in range(ii+1, dHdR.shape[0]):
            d[ii, jj, :] = F_hellmann_feynman[ii, jj, :] / (evals[ii] - evals[jj])
            d[jj, ii, :] = -d[ii, jj, :].conjugate()
            
    return d, F, F_hellmann_feynman

@njit
def evaluate_nonadiabatic_couplings_complex(
    dHdR: GenericVectorOperator,
    evals: RealVector,
    evecs: GenericOperator,
) -> Tuple[NDArray[Shape['A, A, B'], Any], NDArray[Shape['A, B'], Any], NDArray[Shape['A, A, B'], Any]]:
    F_hellmann_feynman = np.zeros((dHdR.shape[0], dHdR.shape[0], dHdR.shape[-1]), dtype=np.complex128)
    F = np.zeros((dHdR.shape[0], dHdR.shape[-1]), dtype=np.complex128)
    d = np.zeros((dHdR.shape[0], dHdR.shape[0], dHdR.shape[-1]), dtype=np.complex128)
    
    _op = np.zeros((dHdR.shape[0], dHdR.shape[0]), dtype=np.complex128) 
    evecs = evecs.astype(np.complex128)
    for kk in range(dHdR.shape[-1]):
        _op[:] = np.ascontiguousarray(dHdR[:, :, kk])
        F_hellmann_feynman[:, :, kk] = -np.dot(evecs.T.conjugate(), np.dot(_op, evecs))
        
    for ii in range(dHdR.shape[0]):
        F[ii, :] = np.real(F_hellmann_feynman[ii, ii, :])
        for jj in range(ii+1, dHdR.shape[0]):
            d[ii, jj, :] = F_hellmann_feynman[ii, jj, :] / (evals[ii] - evals[jj])
            d[jj, ii, :] = -d[ii, jj, :].conjugate()
            
    return d, F, F_hellmann_feynman
    

def vectorized_diagonalization(
    Hv: GenericVectorOperator,
) -> Tuple[RealVector, GenericOperator]:
    evals_tmp, evecs_tmp = np.linalg.eigh(Hv.transpose(2, 0, 1))
    return evals_tmp.T, evecs_tmp.transpose(1, 2, 0)