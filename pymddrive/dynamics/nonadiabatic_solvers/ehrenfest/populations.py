import numpy as np
from numba import njit

from pymddrive.my_types import RealVector, GenericOperator, GenericVectorOperator
from pymddrive.models.nonadiabatic_hamiltonian import diabatic_to_adiabatic, adiabatic_to_diabatic
from pymddrive.dynamics.options import BasisRepresentation

from multiprocessing import Manager
from typing import Optional, Any

manager = Manager()

def compute_populations_from_rho(rho: GenericOperator, evecs: Optional[GenericOperator]=None) -> RealVector:
    return np.diag(rho).real

def compute_populations_from_psi(psi: GenericOperator, evecs: Optional[GenericOperator]=None) -> RealVector:
    return np.abs(psi)**2

def compute_diabatic_populations_from_adiabatic_rho(rho_adiabatic: GenericOperator, evecs: GenericOperator) -> RealVector:
    return diabatic_to_adiabatic(rho_adiabatic, evecs).diagonal().real

def compute_diabatic_populations_from_adiabatic_psi(psi_adiabatic: GenericOperator, evecs: GenericOperator) -> RealVector:
    psi_diabatic = np.dot(evecs, psi_adiabatic)
    return np.abs(psi_diabatic)**2

def compute_adibatic_populations_from_diabatic_rho(rho: GenericOperator, evecs: GenericOperator) -> RealVector:
    return adiabatic_to_diabatic(rho, evecs).diagonal().real

def compute_adibatic_populations_from_diabatic_psi(psi: GenericOperator, evecs: GenericOperator) -> RealVector:
    psi_adiabatic = np.dot(evecs.T.conjugate(), psi)
    return np.abs(psi_adiabatic)**2

# FUNCTION TABLE for computing the populations
# tuple(ndim, state_basis_representation, target_basis_representation) -> function
POPULATION_FUNCTIONS = manager.dict({
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_populations_from_psi,
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_diabatic_populations_from_adiabatic_psi,
    (1, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_adibatic_populations_from_diabatic_psi,
    (1, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC): compute_populations_from_psi,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_populations_from_rho,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_diabatic_populations_from_adiabatic_rho,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_adibatic_populations_from_diabatic_rho,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC): compute_populations_from_rho,
})

def compute_populations(
    state: GenericOperator,
    basis_representation: BasisRepresentation,
    target_basis_representation: BasisRepresentation,
    evecs: Optional[GenericOperator]
) -> RealVector:
    return POPULATION_FUNCTIONS[(state.ndim, basis_representation, target_basis_representation)](state, evecs)

# Floquet theory version of the population calculators
@njit
def compute_floquet_populations_from_rho_ddd(
    rho: GenericOperator, 
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    *args,
    # evecs_F: Optional[GenericOperator] = None,
    # evecs_0: Optional[GenericOperator] = None
) -> RealVector:
    populations = np.zeros(dim, dtype=np.float64)
    for ii in range(dim):
        accum: complex = 0.0 + 0.0j
        for mm in range(-NF, NF+1):
            for nn in range(-NF, NF+1):
                accum += rho[(mm+NF) * dim + ii, (nn+NF) * dim + ii] * np.exp(1j * (mm - nn) * Omega * t)
        populations[ii] = accum.real
    return populations

def compute_floquet_populations_from_rho_dda(
    rho: GenericOperator, 
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    *args: Any,
    evecs_0: GenericOperator
) -> RealVector:
    populations = compute_floquet_populations_from_rho_ddd(rho, Omega, t, NF, dim, evecs_0)
    return np.dot(evecs_0, populations)

def compute_floquet_populations_from_rho_add(
    rho: GenericOperator, 
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    evecs_F: GenericOperator,
    evecs_0: Optional[GenericOperator]=None
) -> RealVector:
    rho_F_diab = adiabatic_to_diabatic(rho, evecs_F)
    return compute_floquet_populations_from_rho_ddd(rho_F_diab, Omega, t, NF, dim)

def compute_floquet_populations_from_rho_ada(
    rho: GenericOperator, 
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    evecs_F: GenericOperator,
    evecs_0: GenericOperator
) -> RealVector:
    rho_F_diab = adiabatic_to_diabatic(rho, evecs_F)
    return compute_floquet_populations_from_rho_dda(rho_F_diab, Omega, t, NF, dim, evecs_0=evecs_0)

def compute_floquet_populations_from_rho_dad(rho: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:    
    raise NotImplementedError("Floquet populations are not implemented yet.")

def compute_floquet_populations_from_rho_daa(rho: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet.")

def compute_floquet_populations_from_rho_aad(rho: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet.")

def compute_floquet_populations_from_rho_aaa(rho: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet.")

@njit
def compute_floquet_populations_from_psi_ddd(
    psi: GenericOperator, 
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    *args: Any,
    # evecs_F: Optional[GenericOperator] = None,
    # evecs_0: Optional[GenericOperator] = None
) -> RealVector:
    populations = np.zeros(dim, dtype=np.float64)
    for ii in range(dim):
        accum: complex = 0.0 + 0.0j
        for mm in range(-NF, NF+1):
            accum += psi[(mm+NF) * dim + ii] * np.exp(1j * mm * Omega * t)
        populations[ii] = np.abs(accum)**2
    return populations

def compute_floquet_populations_from_psi_dda(
    psi: GenericOperator, 
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    *args: Any,
    evecs_0: GenericOperator
) -> RealVector:
    populations = compute_floquet_populations_from_psi_ddd(psi, Omega, t, NF, dim, evecs_0)
    return np.dot(evecs_0, populations)

def compute_floquet_populations_from_psi_add(
    psi: GenericOperator, 
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    evecs_F: GenericOperator,
    evecs_0: Optional[GenericOperator]=None
) -> RealVector:
    psi_F_diab = adiabatic_to_diabatic(psi, evecs_F)
    return compute_floquet_populations_from_psi_ddd(psi_F_diab, Omega, t, NF, dim)

def compute_floquet_populations_from_psi_ada(
    psi: GenericOperator, 
    Omega: float, 
    t: float, 
    NF: int, 
    dim: int,
    evecs_F: GenericOperator,
    evecs_0: GenericOperator
) -> RealVector:
    psi_F_diab = adiabatic_to_diabatic(psi, evecs_F)
    return compute_floquet_populations_from_psi_dda(psi_F_diab, Omega, t, NF, dim, evecs_0=evecs_0)

def compute_floquet_populations_from_psi_dad(psi: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet.")

def compute_floquet_populations_from_psi_daa(psi: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet.")

def compute_floquet_populations_from_psi_aad(psi: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet.")

def compute_floquet_populations_from_psi_aaa(psi: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet.")

# FUNCTION TABLE for computing the populations
FLOQUET_POPULATION_FUNCTIONS = manager.dict({
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_psi_ddd,
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_psi_dda,
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_psi_add,
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_psi_ada,
    (1, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_psi_dad,
    (1, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_psi_daa,
    (1, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_psi_aad,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_rho_ddd,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_rho_dda,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_rho_add,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_rho_ada,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_rho_dad,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_rho_daa,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_floquet_populations_from_rho_aad,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC): compute_floquet_populations_from_rho_aaa,
})

def compute_floquet_populations(
    state: GenericOperator,
    dynamics_basis: BasisRepresentation,
    floquet_basis: BasisRepresentation,
    target_state_basis: BasisRepresentation,
    Omega: float,
    t: float,
    evecs_F: Optional[GenericOperator],
    evecs_0: Optional[GenericOperator],
) -> RealVector:
    return FLOQUET_POPULATION_FUNCTIONS[(state.ndim, dynamics_basis, floquet_basis, target_state_basis)](state, Omega, t, evecs_F, evecs_0)
