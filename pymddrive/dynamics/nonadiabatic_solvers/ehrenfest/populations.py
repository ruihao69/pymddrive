import numpy as np
from numba import njit

from pymddrive.my_types import RealVector, GenericOperator, GenericVector, BlockFloquetOperator
from pymddrive.models.nonadiabatic_hamiltonian import diabatic_to_adiabatic, adiabatic_to_diabatic
from pymddrive.dynamics.options import BasisRepresentation
from pymddrive.models.floquet import get_rhoF

from typing import Optional, Any, Union


def compute_populations_from_rho(rho: GenericOperator, evecs: Optional[GenericOperator]=None) -> RealVector:
    return np.diag(rho).real

def compute_populations_from_psi(psi: GenericVector, evecs: Optional[GenericOperator]=None) -> RealVector:
    return np.abs(psi)**2

def compute_diabatic_populations_from_adiabatic_rho(rho_adiabatic: GenericOperator, evecs: GenericOperator) -> RealVector:
    # return diabatic_to_adiabatic(rho_adiabatic, evecs).diagonal().real
    return adiabatic_to_diabatic(rho_adiabatic, evecs).diagonal().real

def compute_diabatic_populations_from_adiabatic_psi(psi_adiabatic: GenericVector, evecs: GenericOperator) -> RealVector:
    psi_diabatic = np.dot(evecs, psi_adiabatic)
    return np.abs(psi_diabatic)**2

def compute_adiabatic_populations_from_diabatic_rho(rho: GenericOperator, evecs: GenericOperator) -> RealVector:
    # return adiabatic_to_diabatic(rho, evecs).diagonal().real
    return diabatic_to_adiabatic(rho, evecs).diagonal().real

def compute_adiabatic_populations_from_diabatic_psi(psi: GenericVector, evecs: GenericOperator) -> RealVector:
    psi_adiabatic = np.dot(evecs.T.conjugate(), psi)
    return np.abs(psi_adiabatic)**2

# FUNCTION TABLE for computing the populations
# tuple(ndim, state_basis_representation, target_basis_representation) -> function
POPULATION_FUNCTIONS = {
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_populations_from_psi,
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_adiabatic_populations_from_diabatic_psi,
    (1, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_diabatic_populations_from_adiabatic_psi,
    (1, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC): compute_populations_from_psi,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_populations_from_rho,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_adiabatic_populations_from_diabatic_rho,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_diabatic_populations_from_adiabatic_rho,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC): compute_populations_from_rho,
}

def compute_populations(
    state: Union[GenericOperator, GenericVector],
    basis_representation: BasisRepresentation,
    target_basis_representation: BasisRepresentation,
    evecs: Optional[GenericOperator]
) -> RealVector:
    return POPULATION_FUNCTIONS[(state.ndim, basis_representation, target_basis_representation)](state, evecs)

# Floquet theory version of the population calculators
@njit
def compute_populations_from_rhoF_ddd_impl(
    rhoF: BlockFloquetOperator, 
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
) -> RealVector:
    populations = np.zeros(dim, dtype=np.float64)
    for ii in range(dim):
        accum: complex = 0.0 + 0.0j
        for mm in range(-NF, NF+1):
            for nn in range(-NF, NF+1):
                accum += rhoF[mm+NF, nn+NF, ii, ii] * np.exp(1j * (mm - nn) * Omega * t)
        populations[ii] = accum.real
    return populations

@njit
def compute_rho_from_rhoF_ddd_impl(
    rhoF: BlockFloquetOperator,
    Omega: float,
    t: float,
    NF: int,
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
) -> BlockFloquetOperator:
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for ii in range(dim):
        for jj in range(dim):
            for mm in range(-NF, NF+1):
                for nn in range(-NF, NF+1):
                    rho[ii, jj] += rhoF[mm+NF, nn+NF, ii, jj] * np.exp(1j * (mm - nn) * Omega * t)
    return rho

def compute_floquet_populations_from_rho_ddd(
    rho: GenericOperator,
    Omega: float,
    t: float,
    NF: int,
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
) -> RealVector:
    rhoF = get_rhoF(rho, NF, dim)
    return compute_populations_from_rhoF_ddd_impl(rhoF, Omega, t, NF, dim, evecs_0, evecs_F)

def compute_floquet_populations_from_rho_dda(
    rho: GenericOperator,
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
) -> RealVector:
    # populations = compute_floquet_populations_from_rho_ddd(rho, Omega, t, NF, dim, evecs_0, evecs_F)
    rhoF = get_rhoF(rho, NF, dim)
    rho_diabatic = compute_rho_from_rhoF_ddd_impl(rhoF, Omega, t, NF, dim, evecs_0, evecs_F)
    rho_adiabatic = diabatic_to_adiabatic(rho_diabatic, evecs_0)
    return np.real(np.diagonal(rho_adiabatic))
    

def compute_floquet_populations_from_rho_add(
    rho: GenericOperator,
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
) -> RealVector:
    rho_F_diab = adiabatic_to_diabatic(rho, evecs_F)
    return compute_floquet_populations_from_rho_ddd(rho_F_diab, Omega, t, NF, dim, evecs_0, evecs_F)

def compute_floquet_populations_from_rho_ada(
    rho: GenericOperator,
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
) -> RealVector:
    rho_F_diab = adiabatic_to_diabatic(rho, evecs_F)
    return compute_floquet_populations_from_rho_dda(rho_F_diab, Omega, t, NF, dim, evecs_0, evecs_F)

def compute_floquet_populations_from_rho_dad(rho: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:    
    raise NotImplementedError("Floquet populations are not implemented yet for dad.")

def compute_floquet_populations_from_rho_daa(rho: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet for daa.")

def compute_floquet_populations_from_rho_aad(rho: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet for aad.")

def compute_floquet_populations_from_rho_aaa(rho: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet for aaa.")

@njit
def compute_floquet_populations_from_psi_ddd(
    psi: GenericVector, 
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
) -> RealVector:
    populations = np.zeros(dim, dtype=np.float64)
    for ii in range(dim):
        accum: complex = 0.0 + 0.0j
        for mm in range(-NF, NF+1):
            accum += psi[(mm+NF) * dim + ii] * np.exp(1j * mm * Omega * t)
        populations[ii] = np.abs(accum)**2
    return populations

def compute_floquet_populations_from_psi_dda(
    psi: GenericVector,
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
) -> RealVector:
    populations = compute_floquet_populations_from_psi_ddd(psi, Omega, t, NF, dim, evecs_0, evecs_F)
    return np.dot(evecs_0, populations)

def compute_floquet_populations_from_psi_add(
    psi: GenericVector, 
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
) -> RealVector:
    psi_F_diab = adiabatic_to_diabatic(psi, evecs_F)
    return compute_floquet_populations_from_psi_ddd(psi_F_diab, Omega, t, NF, dim, evecs_0, evecs_F)

def compute_floquet_populations_from_psi_ada(
    psi: GenericVector,
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
) -> RealVector:
    psi_F_diab = adiabatic_to_diabatic(psi, evecs_F)
    return compute_floquet_populations_from_psi_dda(psi_F_diab, Omega, t, NF, dim, evecs_0, evecs_F)

def compute_floquet_populations_from_psi_dad(psi: GenericVector, Omega: float, t: float, NF: int, dim: int, evecs_0: GenericOperator, evecs_F: GenericOperator,) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet for dad.")

def compute_floquet_populations_from_psi_daa(psi: GenericVector, Omega: float, t: float, NF: int, dim: int, evecs_0: GenericOperator, evecs_F: GenericOperator,) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet for daa.")

def compute_floquet_populations_from_psi_aad(psi: GenericVector, Omega: float, t: float, NF: int, dim: int, evecs_0: GenericOperator, evecs_F: GenericOperator,) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet for aad.")

def compute_floquet_populations_from_psi_aaa(psi: GenericVector, Omega: float, t: float, NF: int, dim: int, evecs_0: GenericOperator, evecs_F: GenericOperator,) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet for aaa.")

# FUNCTION TABLE for computing the populations
FLOQUET_POPULATION_FUNCTIONS = {
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_psi_ddd,
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_psi_dda,
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_psi_dad,
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_psi_daa,
    (1, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_psi_add,
    (1, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_psi_ada,
    (1, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_psi_aad,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_rho_ddd,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_rho_dda,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_rho_dad,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_rho_daa,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_rho_add,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_rho_ada,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_rho_aad,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_rho_aaa,
}

def compute_floquet_populations(
    state: Union[GenericOperator, GenericVector], 
    dynamics_basis: BasisRepresentation,
    floquet_basis: BasisRepresentation,
    target_state_basis: BasisRepresentation,
    Omega: float,
    t: float,
    NF: int, 
    dim: int,
    evecs_0: GenericOperator, 
    evecs_F: GenericOperator,
) -> RealVector:
    return FLOQUET_POPULATION_FUNCTIONS[(state.ndim, dynamics_basis, floquet_basis, target_state_basis)](state, Omega, t, NF, dim, evecs_0, evecs_F)
