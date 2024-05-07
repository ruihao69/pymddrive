# Velocity verlet integration of the auxiliary variables for the AFSSH method
import numpy as np

from pymddrive.my_types import RealDiagonalVectorOperator, ComplexOperator, RealVector, GenericDiagonalVectorOperator

from typing import Union

def delta_R_dot(
    delta_R: RealDiagonalVectorOperator, # aux position displacement
    delta_P: RealDiagonalVectorOperator, # aux momentum displacement
    delta_F_prev: GenericDiagonalVectorOperator, # aux force displacement
    mass: Union[float, RealVector], # mass of the system
    dt: float, # time step
    rho: ComplexOperator # density matrix
) -> RealDiagonalVectorOperator:
    delta_R_tilde = delta_R.copy()
    delta_R_tilde[:] += delta_P * dt / mass
    delta_R_tilde[:] += np.real(delta_F_prev * rho.diagonal()[:, np.newaxis]) * dt**2 / (2.0 * mass)
    return delta_R_tilde
    
def delta_P_dot(
    delta_P: RealDiagonalVectorOperator, # aux momentum displacement
    delta_F_prev: RealDiagonalVectorOperator, # aux force displacement at t
    delta_F: GenericDiagonalVectorOperator, # aux force displacement at t+dt
    dt: float, # time step
    rho: ComplexOperator # density matrix
) -> RealDiagonalVectorOperator:
    delta_P_tilde = delta_P.copy()
    delta_P_tilde[:] += 0.5 * np.real((delta_F_prev + delta_F) * rho.diagonal()[:, np.newaxis]) * dt
    return delta_P_tilde
