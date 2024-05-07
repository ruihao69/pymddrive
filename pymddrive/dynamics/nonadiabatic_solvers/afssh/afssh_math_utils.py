import numpy as np
from numba import njit

from pymddrive.my_types import RealVector, ComplexVector, ComplexOperator, GenericOperator, GenericVectorOperator, ActiveSurface, GenericDiagonalVectorOperator

def evaluate_delta_F(
    F: GenericVectorOperator, # force operator
    active_surface: ActiveSurface, # the active surface
) -> GenericDiagonalVectorOperator:
    lambd: int = active_surface[0]
    return F - F[lambd, :]

@njit
def compute_delta_op_tilde(
    delta_op: GenericDiagonalVectorOperator,
    evecs: GenericOperator
) -> GenericDiagonalVectorOperator:
    dim: int = evecs.shape[0]
    delta_op_tilde = np.zeros_like(delta_op)
    
    for jj in range(dim):
        for kk in range(dim):
            delta_op_tilde[jj] = np.abs(evecs[jj, kk])**2 * delta_op[kk]
            
    return delta_op_tilde
