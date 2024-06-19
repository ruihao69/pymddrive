import numpy as np
from numba import njit
from nptyping import NDArray, Shape
# import scipy.sparse as sp

from pymddrive.my_types import RealVector, GenericOperator, GenericVectorOperator, RealVectorOperator, RealOperator, ComplexVector
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

def evaluate_pulse_gradient(
    t: float, 
    R: RealVector,
    hamiltonian: Union[TD_HamiltonianBase, QuasiFloquetHamiltonianBase],
) -> Tuple[float, GenericOperator]:
    if isinstance(hamiltonian, TD_HamiltonianBase):
        H1 = hamiltonian.H1(t, R)
        pulse_value = hamiltonian.pulse(t)
        pulse_gradient = hamiltonian.pulse.gradient(t)
        return pulse_gradient, H1 / pulse_value
    elif isinstance(hamiltonian, QuasiFloquetHamiltonianBase):
        H1 = hamiltonian.H1(t, R)
        pulse_value = hamiltonian.envelope_pulse(t)
        pulse_gradient = hamiltonian.envelope_pulse.gradient(t)
        return pulse_gradient, get_grad_HF_Et_upper(hamiltonian.floquet_type, H1 / pulse_value, hamiltonian.NF)
    else:
        return 0, 0

def evaluate_pulse_NAC(
    pulse_gradient: float, # note: the current implementation don't support complex pulse value, yet
    grad_H: RealOperator,
    evals: RealVector,
    evecs: GenericOperator,
    is_floquet: bool = False,
) -> GenericOperator:
    if pulse_gradient == 0:
        return 0
    elif is_floquet:
        grad_H_upper = -grad_H
        grad_H_lower = -grad_H.conjugate().T
        # return evaluate_pulse_NAC_TD(pulse_gradient, grad_H_upper, evals, evecs) + evaluate_pulse_NAC_TD(np.conjugate(pulse_gradient), grad_H_lower, evals, evecs)
        return evaluate_pulse_NAC_TD(pulse_gradient, grad_H_lower, evals, evecs) + evaluate_pulse_NAC_TD(np.conjugate(pulse_gradient), grad_H_upper, evals, evecs)
        # grad_H = grad_H + grad_H.conjugate().T
        # print(f"{pulse_gradient=}")
        # return evaluate_pulse_NAC_TD(pulse_gradient, grad_H, evals, evecs)
    else:
        return evaluate_pulse_NAC_TD(pulse_gradient, grad_H, evals, evecs)
    
def evaluate_pulse_NAC2(
    pulse_gradient: float, # note: the current implementation don't support complex pulse value, yet
    grad_H: RealOperator,
    evals: RealVector,
    evecs: GenericOperator,
    phase_correction: ComplexVector,
    is_floquet: bool = False,
) -> GenericOperator:
    if pulse_gradient == 0:
        return 0
    elif is_floquet:
        grad_H_upper = -grad_H
        grad_H_lower = -grad_H.conjugate().T
        # return evaluate_pulse_NAC_TD(pulse_gradient, grad_H_upper, evals, evecs) + evaluate_pulse_NAC_TD(np.conjugate(pulse_gradient), grad_H_lower, evals, evecs)
        return evaluate_pulse_NAC_TD2(pulse_gradient, grad_H_lower, evals, evecs, phase_correction) + evaluate_pulse_NAC_TD2(np.conjugate(pulse_gradient), grad_H_upper, evals, evecs, phase_correction)
        # grad_H = grad_H + grad_H.conjugate().T
        # print(f"{pulse_gradient=}")
        # return evaluate_pulse_NAC_TD(pulse_gradient, grad_H, evals, evecs)
    else:
        return evaluate_pulse_NAC_TD2(pulse_gradient, grad_H, evals, evecs, phase_correction) 
        
        
def evaluate_pulse_NAC_TD(
    pulse_gradient: float, # note: the current implementation don't support complex pulse value, yet
    grad_H: RealOperator,
    evals: RealVector,
    evecs: RealOperator,
) -> GenericOperator:
    if np.iscomplexobj(grad_H) or np.iscomplexobj(evecs):
        return evaluate_pulse_NAC_TD_complex(pulse_gradient, grad_H.astype(np.complex128), evals, evecs.astype(np.complex128))
    else:
        return evaluate_pulse_NAC_TD_real(pulse_gradient, grad_H, evals, evecs)
    
def evaluate_pulse_NAC_TD2(
    pulse_gradient: float, # note: the current implementation don't support complex pulse value, yet
    grad_H: RealOperator,
    evals: RealVector,
    evecs: RealOperator,
    phase_correction: ComplexVector,
) -> GenericOperator:
    if np.iscomplexobj(grad_H) or np.iscomplexobj(evecs):
        return evaluate_pulse_NAC_TD_complex2(pulse_gradient, grad_H.astype(np.complex128), evals, evecs.astype(np.complex128), phase_correction)
    else:
        if np.iscomplexobj(phase_correction):
            raise ValueError("phase_correction should be real for a real Hamiltonian")
        return evaluate_pulse_NAC_TD_real2(pulse_gradient, grad_H, evals, evecs, phase_correction)

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
            nac_pulse[ii, jj] = grad_H[ii, jj] / (evals[ii] - evals[jj]) # * pulse_gradient
            nac_pulse[jj, ii] = -nac_pulse[ii, jj]
    return nac_pulse * pulse_gradient

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
            nac_pulse[ii, jj] = grad_H[ii, jj] / (evals[ii] - evals[jj]) # * pulse_gradient
            nac_pulse[jj, ii] = -nac_pulse[ii, jj].conjugate()
    return nac_pulse * pulse_gradient

@njit
def evaluate_pulse_NAC_TD_real2(
    pulse_gradient: float, # note: the current implementation don't support complex pulse value, yet
    grad_H: RealOperator,
    evals: RealVector,
    evecs: RealOperator,
    phase_correction: RealVector,
) -> RealOperator:
    nac_pulse = np.dot(evecs.T, np.dot(grad_H, evecs))
    for ii in range(evals.shape[0]):
        nac_pulse[ii, ii] = 0.0
        for jj in range(ii+1, evals.shape[0]):
            fi_fjconj = phase_correction[ii] * phase_correction[jj] 
            nac_pulse[ii, jj] = grad_H[ii, jj] / (evals[ii] - evals[jj]) * fi_fjconj# * pulse_gradient
            nac_pulse[jj, ii] = -nac_pulse[ii, jj]
    return nac_pulse * pulse_gradient

@njit
def evaluate_pulse_NAC_TD_complex2(
    pulse_gradient: float, # note: the current implementation don't support complex pulse value, yet
    grad_H: GenericOperator,
    evals: RealVector,
    evecs: GenericOperator,
    phase_correction: ComplexVector,
) -> RealOperator:
    nac_pulse = np.zeros((evecs.shape[0], evecs.shape[0]), dtype=np.complex128)
    nac_pulse[:] = np.dot(evecs.T.conjugate(), np.dot(grad_H, evecs))
    for ii in range(evals.shape[0]):
        nac_pulse[ii, ii] = 0.0
        for jj in range(ii+1, evals.shape[0]):
            fi_fjconj = phase_correction[ii] * np.conjugate(phase_correction[jj])
            nac_pulse[ii, jj] = grad_H[ii, jj] / (evals[ii] - evals[jj]) * fi_fjconj # * pulse_gradient
            nac_pulse[jj, ii] = -nac_pulse[ii, jj].conjugate()
    return nac_pulse * pulse_gradient

def evaluate_nonadiabatic_couplings(
    dHdR: GenericVectorOperator,
    evals: RealVector,
    evecs: GenericOperator,
) -> Tuple[NDArray[Shape['A, A, B'], Any], NDArray[Shape['A, B'], Any], NDArray[Shape['A, A, B'], Any]]:
    if np.iscomplexobj(dHdR) or np.iscomplexobj(evecs):
        try:
            return evaluate_nonadiabatic_couplings_complex(dHdR, evals, evecs)
        except ZeroDivisionError:
            raise ZeroDivisionError(f"Conical intersection detected. Please check the Hamiltonian matrix. Print the eigenvalues {evals}")
    else:
        return evaluate_nonadiabatic_couplings_real(dHdR, evals, evecs)
    
def evaluate_nonadiabatic_couplings2(
    dHdR: GenericVectorOperator,
    evals: RealVector,
    evecs: GenericOperator,
    phase_correction: ComplexVector
) -> Tuple[NDArray[Shape['A, A, B'], Any], NDArray[Shape['A, B'], Any], NDArray[Shape['A, A, B'], Any]]:
    if np.iscomplexobj(dHdR) or np.iscomplexobj(evecs):
        try:
            return evaluate_nonadiabatic_couplings_complex2(dHdR, evals, evecs, phase_correction)
        except ZeroDivisionError:
            raise ZeroDivisionError(f"Conical intersection detected. Please check the Hamiltonian matrix. Print the eigenvalues {evals}")
    else:
        if np.iscomplexobj(phase_correction):
            raise ValueError("phase_correction should be real for a real Hamiltonian")
        return evaluate_nonadiabatic_couplings_real2(dHdR, evals, evecs, phase_correction)

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

@njit
def evaluate_nonadiabatic_couplings_real2(
    dHdR: RealVectorOperator,
    evals: RealVector,
    evecs: RealOperator,
    phase_correction: RealVector
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
            fi_fjconj = phase_correction[ii] * phase_correction[jj]  
            F_hellmann_feynman[ii, jj, :] = F_hellmann_feynman[ii, jj, :] * fi_fjconj            
            d[ii, jj, :] = F_hellmann_feynman[ii, jj, :] / (evals[ii] - evals[jj])
            d[jj, ii, :] = -d[ii, jj, :].conjugate()
            
    return d, F, F_hellmann_feynman

@njit
def evaluate_nonadiabatic_couplings_complex2(
    dHdR: GenericVectorOperator,
    evals: RealVector,
    evecs: GenericOperator,
    phase_correction: ComplexVector
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
            fi_fjconj = phase_correction[ii] * np.conjugate(phase_correction[jj])
            F_hellmann_feynman[ii, jj, :] = F_hellmann_feynman[ii, jj, :] * fi_fjconj       
            d[ii, jj, :] = F_hellmann_feynman[ii, jj, :] / (evals[ii] - evals[jj])
            d[jj, ii, :] = -d[ii, jj, :].conjugate()
            
    return d, F, F_hellmann_feynman
    
    

def vectorized_diagonalization(
    Hv: GenericVectorOperator,
) -> Tuple[RealVector, GenericOperator]:
    evals_tmp, evecs_tmp = np.linalg.eigh(Hv.transpose(2, 0, 1))
    return evals_tmp.T, evecs_tmp.transpose(1, 2, 0)