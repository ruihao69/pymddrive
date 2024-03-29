# %%
import numpy as np
from numba import njit

from pymddrive.my_types import RealVector, ComplexVector, ComplexOperator, GenericVectorOperator, GenericDiagonalVectorOperator

from typing import Union

@njit
def mean_force_adiabatic_representation_wavefunction(
    F: GenericDiagonalVectorOperator,
    evals: RealVector,
    dc: GenericVectorOperator, 
    psi: ComplexVector 
) -> RealVector:
    meanF = np.zeros((F.shape[-1], ), dtype=np.float64)
    for kk in range(meanF.shape[0]):
        for ii in range(F.shape[0]):
            rho_ii: np.complex128 = psi[ii] * np.conjugate(psi[ii])
            meanF[kk] += np.real(F[ii, kk] * rho_ii)
            for jj in range(ii+1, F.shape[0]):
                dE: float = evals[jj] - evals[ii]
                rho_ij_dc_ji: np.complex128 = psi[ii] * np.conjugate(psi[jj]) * dc[jj, ii, kk] 
                meanF[kk] += 2.0 * dE * np.real(rho_ij_dc_ji)
    return meanF

@njit
def mean_force_adiabatic_representation_density_matrix(
    F: GenericDiagonalVectorOperator,
    evals: RealVector,
    dc: GenericVectorOperator, 
    rho: ComplexOperator 
) -> RealVector:
    meanF = np.zeros((F.shape[-1], ), dtype=np.float64)
    for kk in range(meanF.shape[0]):
        for ii in range(F.shape[0]):
            meanF[kk] += np.real(F[ii, kk] * rho[ii, ii])
            for jj in range(ii+1, F.shape[0]):
                dE: float = evals[jj] - evals[ii]
                rho_ij_dc_ji: np.complex128 = rho[ii, jj] * dc[jj, ii, kk] 
                meanF[kk] += 2.0 * dE * np.real(rho_ij_dc_ji)
    return meanF

def mean_force_adiabatic_representation(
    F: GenericDiagonalVectorOperator,
    evals: RealVector,
    dc: GenericVectorOperator,
    quantum_state: Union[ComplexVector, ComplexOperator]
) -> RealVector:
    if quantum_state.ndim == 1:
        return mean_force_adiabatic_representation_wavefunction(F, evals, dc, quantum_state)
    elif quantum_state.ndim == 2:
        return mean_force_adiabatic_representation_density_matrix(F, evals, dc, quantum_state)
    else:
        raise ValueError("Invalid quantum state dimension")
    
# %%
