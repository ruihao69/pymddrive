# %%
import numpy as np
from numba import njit
from pymddrive.my_types import GenericOperator, GenericVectorOperator

from typing import Union

def get_sigma_z() -> GenericOperator:
    """Return the Pauli Z matrix for a given dimension."""
    return np.diag([1, -1])

def get_sigma_x() -> GenericOperator:
    """Return the Pauli X matrix for a given dimension."""
    return np.array([[0, 1], [1, 0]])

def get_sigma_y() -> GenericOperator:
    """Return the Pauli Y matrix for a given dimension."""
    return np.array([[0, -1j], [1j, 0]])

def mu_Et(mu: GenericOperator, Et: Union[float, complex]) -> GenericOperator:
    if not np.iscomplexobj(Et):
        return mu * Et
    else:
        return _mu_Et(mu, Et)

@njit
def _mu_Et(mu: GenericOperator, Et: complex) -> GenericOperator:
    dim = mu.shape[0]
    mu_Et = np.zeros(mu.shape, dtype=np.complex128)
    for i in range(dim):
        mu_Et[i, i] = mu[i, i] * Et
        for j in range(i+1, dim):
            mu_Et[i, j] = mu[i, j] * Et
            mu_Et[j, i] = mu[j, i] * np.conj(Et)
    return mu_Et

def dmu_dR_Et(dmu_dR: GenericVectorOperator, Et: Union[float, complex]) -> GenericVectorOperator:
    if not np.iscomplexobj(Et):
        return dmu_dR * Et
    else:
        return _dmu_dR_Et(dmu_dR, Et)
    
@njit
def _dmu_dR_Et(dmu_dR: GenericVectorOperator, Et: Union[float, complex]) -> GenericVectorOperator:
    dim = dmu_dR.shape[0]
    dmu_dR_Et = np.zeros(dmu_dR.shape, dtype=np.complex128)
    for i in range(dim):
        dmu_dR_Et[i, i, ...] = dmu_dR[i, i, ...] * Et
        for j in range(i+1, dim):
            dmu_dR_Et[i, j, ...] = dmu_dR[i, j, ...] * Et
            dmu_dR_Et[j, i, ...] = dmu_dR[j, i, ...] * np.conj(Et)
    return dmu_dR_Et