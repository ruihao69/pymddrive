import numpy as np
from numba import njit

from pymddrive.my_types import RealVector, GenericOperator, GenericVector, ActiveSurface
from pymddrive.models.nonadiabatic_hamiltonian import diabatic_to_adiabatic, adiabatic_to_diabatic
from pymddrive.dynamics.options import BasisRepresentation
from pymddrive.models.floquet import get_rhoF

from typing import Union, Any

@njit
def compute_diabatic_populations_from_adiabatic_rho(
    rho: GenericOperator,
    evecs: GenericOperator,
    active_surface: ActiveSurface,
) -> RealVector:
    dim: int = evecs.shape[0]
    active_state: int = active_surface[0]
    populations = np.zeros(dim, dtype=np.float64)
    for istate in range(dim):
        populations[istate] += np.abs(evecs[istate, active_state])**2
        for jj in range(dim):
            for kk in range(jj+1, dim):
                populations[istate] += 2.0 * np.real(evecs[istate, jj] * rho[jj, kk] * np.conjugate(evecs[istate, kk]))
    return populations

# def compute_diabatic_populations_from_adiabatic_rho(
#     rho: GenericOperator,
#     evecs: GenericOperator,
#     active_surface: ActiveSurface,
# ) -> RealVector:
#     np.fill_diagonal(rho, 0.0)
#     rho[active_surface[0], active_surface[0]] = 1.0
#     return np.real(adiabatic_to_diabatic(rho, evecs).diagonal())

def compute_diabatic_populations_from_adiabatic_psi(
    psi: GenericVector,
    evecs: GenericOperator,
    active_surface: ActiveSurface
) -> RealVector:
    rho = np.outer(psi, psi.conjugate())
    # np.fill_diagonal(rho, 0.0) 
    # rho[active_surface[0], active_surface[0]] = 1.0
    # return np.real(adiabatic_to_diabatic(rho, evecs).diagonal())
    return compute_diabatic_populations_from_adiabatic_rho(rho, evecs, active_surface)

def compute_populations_from_active_surface(
    rho_or_psi: Union[GenericOperator, GenericVector],
    evecs: GenericOperator,
    active_surface: ActiveSurface,
) -> RealVector:
    dim: int = rho_or_psi.shape[0]
    populations = np.zeros(dim, dtype=np.float64)
    populations[active_surface[0]] = 1.0
    return populations

def compute_adiabatic_populations_from_diabatic_rho(rho: GenericOperator, evecs: GenericOperator, active_surface: ActiveSurface) -> RealVector:
    raise NotImplementedError("We don't provide adiabatic populations from diabatic rho, because we don't encourage using diabatic basis for surface hopping algorithms.")

def compute_adiabatic_populations_from_diabatic_psi(psi: GenericOperator, evecs: GenericOperator, active_surface: ActiveSurface) -> RealVector:
    raise NotImplementedError("We don't provide adiabatic populations from diabatic psi, because we don't encourage using diabatic basis for surface hopping algorithms.")

def compute_diabatic_populations_from_diabatic_rho(rho: GenericOperator, evecs: GenericOperator, active_surface: ActiveSurface) -> RealVector:
    raise NotImplementedError("We don't provide adiabatic populations from diabatic rho, because we don't encourage using diabatic basis for surface hopping algorithms.")

def compute_diabatic_populations_from_diabatic_psi(psi: GenericOperator, evecs: GenericOperator, active_surface: ActiveSurface) -> RealVector:
    raise NotImplementedError("We don't provide adiabatic populations from diabatic psi, because we don't encourage using diabatic basis for surface hopping algorithms.")

POPULATION_FUNCTIONS = {
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_diabatic_populations_from_diabatic_psi,
    (1, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_adiabatic_populations_from_diabatic_psi,
    (1, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_diabatic_populations_from_adiabatic_psi,
    (1, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC): compute_populations_from_active_surface,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.DIABATIC): compute_diabatic_populations_from_diabatic_rho,
    (2, BasisRepresentation.DIABATIC, BasisRepresentation.ADIABATIC): compute_adiabatic_populations_from_diabatic_rho,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.DIABATIC): compute_diabatic_populations_from_adiabatic_rho,
    (2, BasisRepresentation.ADIABATIC, BasisRepresentation.ADIABATIC): compute_populations_from_active_surface,
}

def compute_populations(
    state: Union[GenericOperator, GenericVector],
    basis_representation: BasisRepresentation,
    target_basis_representation: BasisRepresentation,
    evecs: GenericOperator,
    active_surface: ActiveSurface
) -> RealVector:
    return POPULATION_FUNCTIONS[(state.ndim, basis_representation, target_basis_representation)](state, evecs, active_surface)


# Floquet version of the populations calculation routines
from pymddrive.dynamics.nonadiabatic_solvers.ehrenfest.populations import compute_floquet_populations_from_rho_ddd as ehrenfest_compute_floquet_populations_from_rho_ddd
from pymddrive.dynamics.nonadiabatic_solvers.ehrenfest.populations import compute_floquet_populations_from_psi_ddd as ehrenfest_compute_floquet_populations_from_psi_ddd
from pymddrive.dynamics.nonadiabatic_solvers.ehrenfest.populations import compute_rho_from_rhoF_ddd_impl

@njit
def fill_coherence(rho: GenericOperator, rho_coarse_grain: GenericOperator) -> None:
    dim: int = rho.shape[0]
    for ii in range(dim):
        for jj in range(ii+1, dim):
            rho_coarse_grain[ii, jj] = rho[ii, jj]
            rho_coarse_grain[jj, ii] = rho[jj, ii]

@njit
def fill_coherence_2(rho: GenericOperator, rho_coarse_grain: GenericOperator, active_surface: int) -> None:
    dim: int = rho.shape[0]
    for jj in range(dim):
        if jj == active_surface:
            continue
        rho_coarse_grain[active_surface, jj] += 0.5 * rho[active_surface, jj] / rho[active_surface, active_surface]
        rho_coarse_grain[jj, active_surface] += 0.5 * rho[jj, active_surface] / rho[active_surface, active_surface]

def get_coarse_grained_diabatic_rhoF(
    rho: GenericOperator,
    evecs_F: GenericOperator,
    active_surface: ActiveSurface
):
    rho_coarse_grain = np.zeros((evecs_F.shape[0], evecs_F.shape[0]), dtype=np.complex128)
    rho_coarse_grain[active_surface[0], active_surface[0]] = 1.0
    fill_coherence(rho, rho_coarse_grain)
    # fill_coherence_2(rho, rho_coarse_grain, active_surface[0])
    # active_state_adiabatic = np.zeros(rho.shape[0], dtype=np.float64)
    # active_state_adiabatic[active_surface[0]] = 1.0

    # Adiabatic to diabatic
    coarse_grain_rho_diabatic = adiabatic_to_diabatic(rho_coarse_grain, evecs_F)
    # coarse_grain_rho_diabatic = diabatic_to_adiabatic(rho_coarse_grain, evecs_F)
    return coarse_grain_rho_diabatic

def compute_floquet_populations_from_rho_ddd(
    rho: GenericOperator,
    Omega: float,
    t: float,
    NF: int,
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
    active_surface: ActiveSurface
) -> RealVector:
    raise NotImplementedError("We don't provide populations calculators from diabatic rho/psi, because we don't encourage using diabatic basis for surface hopping algorithms. (Floquet ddd)")

def compute_floquet_populations_from_rho_dda(
    rho: GenericOperator,
    Omega: float,
    t: float,
    NF: int,
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
    active_surface: ActiveSurface
) -> RealVector:
    raise NotImplementedError("We don't provide populations calculators from diabatic rho/psi, because we don't encourage using diabatic basis for surface hopping algorithms. (Floquet dda)")

def compute_floquet_populations_from_rho_add(
    rho: GenericOperator,
    Omega: float,
    t: float,
    NF: int,
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
    active_surface: ActiveSurface
) -> RealVector:
    rho_F_diab = get_coarse_grained_diabatic_rhoF(rho, evecs_F, active_surface)
    return ehrenfest_compute_floquet_populations_from_rho_ddd(rho_F_diab, Omega, t, NF, dim, evecs_0, evecs_F)


def compute_floquet_populations_from_rho_ada(
    rho: GenericOperator,
    Omega: float,
    t: float,
    NF: int,
    dim: int,
    evecs_0: GenericOperator,
    evecs_F: GenericOperator,
    active_surface: ActiveSurface
) -> RealVector:
    rho_diab = get_coarse_grained_diabatic_rhoF(rho, evecs_F, active_surface)
    rho_F_diab = get_rhoF(rho_diab, NF, dim)
    rho_diab = compute_rho_from_rhoF_ddd_impl(rho_F_diab, Omega, t, NF, dim, evecs_0, evecs_F)
    rho_adiabatic = diabatic_to_adiabatic(rho_diab, evecs_0)
    return np.real(np.diagonal(rho_adiabatic))

def compute_floquet_populations_from_rho_dad(rho: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet for dad.")

def compute_floquet_populations_from_rho_daa(rho: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet for daa.")

def compute_floquet_populations_from_rho_aad(rho: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet for aad.")

def compute_floquet_populations_from_rho_aaa(rho: GenericOperator, Omega: float, t: float, NF: int, dim: int, *args: Any) -> RealVector:
    raise NotImplementedError("Floquet populations are not implemented yet for aaa.")

def compute_floquet_populations_from_psi_ddd(psi: GenericVector, Omega: float, t: float, NF: int, dim: int, evecs_0: GenericOperator, evecs_F: GenericOperator, active_surface: ActiveSurface) -> RealVector:
    raise NotImplementedError("We don't provide populations calculators from diabatic rho/psi, because we don't encourage using diabatic basis for surface hopping algorithms. (Floquet ddd)")

def compute_floquet_populations_from_psi_dda(psi: GenericVector, Omega: float, t: float, NF: int, dim: int, evecs_0: GenericOperator, evecs_F: GenericOperator, active_surface: ActiveSurface) -> RealVector:
    raise NotImplementedError("We don't provide populations calculators from diabatic rho/psi, because we don't encourage using diabatic basis for surface hopping algorithms. (Floquet dda)")

def compute_floquet_populations_from_psi_add(psi: GenericVector, Omega: float, t: float, NF: int, dim: int, evecs_0: GenericOperator, evecs_F: GenericOperator, active_surface: ActiveSurface) -> RealVector:
    raise NotImplementedError("We don't provide populations calculators from diabatic rho/psi, because we don't encourage using diabatic basis for surface hopping algorithms. (Floquet add)")

def compute_floquet_populations_from_psi_ada(psi: GenericVector, Omega: float, t: float, NF: int, dim: int, evecs_0: GenericOperator, evecs_F: GenericOperator, active_surface: ActiveSurface) -> RealVector:
    raise NotImplementedError("We don't provide populations calculators from diabatic rho/psi, because we don't encourage using diabatic basis for surface hopping algorithms. (Floquet ada)")

def compute_floquet_populations_from_psi_dad(psi: GenericVector, Omega: float, t: float, NF: int, dim: int, evecs_0: GenericOperator, evecs_F: GenericOperator, active_surface: ActiveSurface) -> RealVector:
    raise NotImplementedError("We don't provide populations calculators from diabatic rho/psi, because we don't encourage using diabatic basis for surface hopping algorithms. (Floquet dad)")

def compute_floquet_populations_from_psi_daa(psi: GenericVector, Omega: float, t: float, NF: int, dim: int, evecs_0: GenericOperator, evecs_F: GenericOperator, active_surface: ActiveSurface) -> RealVector:
    raise NotImplementedError("We don't provide populations calculators from diabatic rho/psi, because we don't encourage using diabatic basis for surface hopping algorithms. (Floquet daa)")

def compute_floquet_populations_from_psi_aad(psi: GenericVector, Omega: float, t: float, NF: int, dim: int, evecs_0: GenericOperator, evecs_F: GenericOperator, active_surface: ActiveSurface) -> RealVector:
    raise NotImplementedError("We don't provide populations calculators from diabatic rho/psi, because we don't encourage using diabatic basis for surface hopping algorithms. (Floquet aad)")

def compute_floquet_populations_from_psi_aaa(psi: GenericVector, Omega: float, t: float, NF: int, dim: int, evecs_0: GenericOperator, evecs_F: GenericOperator, active_surface: ActiveSurface) -> RealVector:
    raise NotImplementedError("We don't provide populations calculators from diabatic rho/psi, because we don't encourage using diabatic basis for surface hopping algorithms. (Floquet aaa)")

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
    dim: int,
    NF: int,
    evecs_F: GenericOperator,
    evecs_0: GenericOperator,
    active_surface: ActiveSurface
) -> RealVector:
    return FLOQUET_POPULATION_FUNCTIONS[(state.ndim, dynamics_basis, floquet_basis, target_state_basis)](state, Omega, t, NF, dim, evecs_0, evecs_F, active_surface)



