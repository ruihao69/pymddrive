import numpy as np
from numba import njit
import scipy.linalg as LA

from pymddrive.my_types import GenericOperator, GenericVectorOperator
from pymddrive.low_level.floquet import get_HF_cos, get_dHF_dR_cos, get_HF_sin, get_dHF_dR_sin
from pymddrive.models.floquet.floquet_types import FloquetType

def get_grad_HF_Et_upper_cosine(floquet_type: FloquetType, grad_H1_Et: GenericOperator, NF: int):
    dim: int = grad_H1_Et.shape[0]
    dtype = grad_H1_Et.dtype
    
    dimF = (2 * NF + 1) * dim
    grad_HF_Et_upper = np.zeros((dimF, dimF), dtype=dtype)
    data = (grad_H1_Et * 0.5 for _ in range(2 * NF))
    # use broadcasting to fill the upper 
    grad_HF_Et_upper[:(dimF-dim), dim:] = LA.block_diag(*data)
    return grad_HF_Et_upper

def get_grad_HF_Et_upper_sine(floquet_type: FloquetType, grad_H1_Et: GenericOperator, NF: int):
    dim: int = grad_H1_Et.shape[0]
    dtype = grad_H1_Et.dtype
    
    dimF = (2 * NF + 1) * dim
    grad_HF_Et_upper = np.zeros((dimF, dimF), dtype=dtype)
    data = (grad_H1_Et / 1.j * 0.5 for _ in range(2 * NF))
    # use broadcasting to fill the upper 
    grad_HF_Et_upper[dim:, :(dimF-dim)] = LA.block_diag(*data)
    return grad_HF_Et_upper

HF_function_table = {
    FloquetType.COSINE: get_HF_cos,
    FloquetType.SINE: get_HF_sin,
}

dHF_dR_function_table = {
    FloquetType.COSINE: get_dHF_dR_cos,
    FloquetType.SINE: get_dHF_dR_sin,
}

grad_HF_Et_upper_table = {
    FloquetType.COSINE: get_grad_HF_Et_upper_cosine,
    FloquetType.SINE: get_grad_HF_Et_upper_sine,
}

def get_HF(floquet_type: FloquetType, H0: GenericOperator, H1: GenericOperator, Omega: float, NF: int):
    try:
        return HF_function_table[floquet_type](H0, H1, Omega, NF)
    except KeyError:
        raise NotImplementedError(f"The Floquet type {floquet_type} is not implemented yet.")
    
def get_dHF_dR(floquet_type: FloquetType, dH0dR: GenericVectorOperator, dH1dR: GenericVectorOperator, NF: int):
    try:
        return dHF_dR_function_table[floquet_type](dH0dR, dH1dR, NF)
    except KeyError:
        raise NotImplementedError(f"The Floquet type {floquet_type} is not implemented yet.")
    
def get_grad_HF_Et_upper(floquet_type: FloquetType, grad_H1_Et: GenericOperator, NF: int):
    """get the upper part of the gradient of the Floquet Hamiltonian with respect to the pulse envelope.

    Args:
        floquet_type (FloquetType): the Floquet type
        grad_H1_Et (GenericOperator): the gradient of the Hamiltonian with respect to the pulse envelope
    """
    try:
        return grad_HF_Et_upper_table[floquet_type](floquet_type, grad_H1_Et, NF)
    except KeyError:
        raise NotImplementedError(f"The Floquet type {floquet_type} is not implemented yet.") 
   
