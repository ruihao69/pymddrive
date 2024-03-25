# %%
import numpy as np
from numpy.typing import NDArray
from numba import jit

from pymddrive.low_level.states import State
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, QuasiFloquetHamiltonianBase
from pymddrive.integrators.rk4 import rk4
from pymddrive.integrators.state_rk4 import state_rk4
from pymddrive.dynamics.options import BasisRepresentation, QunatumRepresentation, NumericalIntegrators
from pymddrive.dynamics.misc_utils import eval_nonadiabatic_hamiltonian, HamiltonianRetureType
from pymddrive.dynamics.math_utils import rhs_density_matrix, v_dot_d
from pymddrive.dynamics.floquet.fssh import get_rho_and_populations
from pymddrive.dynamics.cache import Cache
from pymddrive.dynamics.langevin import LangevinBase

from typing import Tuple, Callable, Optional, Union
from collections import namedtuple 
from dataclasses import dataclass
from functools import partial

# FSSHCache = namedtuple('FSSHCache', 'active_surf, evals, evecs, H_diab, hamiltonian')
FSSHProperties = namedtuple('FSSHProperties', 'KE, PE, diab_populations, adiab_populations, active_surf')

@dataclass(frozen=True, order=True, unsafe_hash=True)
class ActiveSurface:
    index: int
    
def choose_fssh_stepper(numerical_integrator: NumericalIntegrators) -> Callable:
    if numerical_integrator == NumericalIntegrators.RK4:
        return step_rk
    else:
        raise NotImplemented(f"FSSH is not implemented for {numerical_integrator=} at this time.")

def choose_fssh_deriv(
    quantum_representation: QunatumRepresentation
) -> Callable:
    if quantum_representation == QunatumRepresentation.DensityMatrix:
        return _deriv_fssh_dm
    elif quantum_representation == QunatumRepresentation.Wavefunction:
        raise NotImplementedError(f"Quantum representation Wavefunction is not implemented for Ehrenfest dynamics.")
    else:
        raise NotImplementedError(f"Quantum representation {quantum_representation} is not implemented for Ehrenfest dynamics.")

def _rho_dot(
    rho: NDArray[np.complex128],
    v: NDArray[np.float64],
    hami_return: HamiltonianRetureType,
    basis_rep: BasisRepresentation
) -> NDArray[np.complex128]:
    if basis_rep == BasisRepresentation.Adiabatic:
        return _rho_dot_adiab(rho, hami_return.evals, v, hami_return.d)
    elif basis_rep == BasisRepresentation.Diabatic:
        return _rho_dot_diab(rho, hami_return.H)
    
def _rho_dot_adiab(
    rho: NDArray[np.complex128], 
    evals: NDArray[np.float64], 
    v: NDArray[np.float64],
    d: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    vdotd = v_dot_d(v, d)
    # if not np.allclose(vdotd + vdotd.T.conj(), 0):
    #     print("The symmetry of vdotd is not preserved.")
    # else:
    #     print("The symmetry of vdotd is preserved.")
    # print(vdotd.dtype)
    return rhs_density_matrix(rho=rho, evals=evals, vdotd=vdotd)

def _rho_dot_diab(rho: NDArray[np.complex128], H: NDArray[np.complex128]) -> NDArray[np.complex128]:
    raise NotImplementedError(f"the surface hopping algorithm for diabatic representation is not implemented yet.")

def _deriv_fssh_dm(
    t: float,
    s: State, 
    cache: Cache,
    hamiltonian: HamiltonianBase,
    basis_rep: BasisRepresentation,
) -> State:
    R, P, rho = s.get_variables()
    dsdt = s.zeros_like()
    # i_active_surf = active_surface.index
    i_active_surf = cache.active_surface
    
    # integrate R
    v = s.get_v()
    dsdt.set_R(v)
    
    # integrate P
    hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
    F_active_surf = hami_return.F[i_active_surf]
    F_langevin = cache.F_langevin
    dsdt.set_P(F_active_surf + F_langevin)
    
    # integrate the density matrix
    rho_dot = _rho_dot(rho, v, hami_return, basis_rep)
    dsdt.set_rho(rho_dot)
    
    return dsdt

def step_rk(
    t: float,
    s: State,
    cache: Cache,
    dt: float,
    hamiltonian: HamiltonianBase,
    langevin: LangevinBase,
    basis_rep: BasisRepresentation,
) -> Tuple[float, State, Cache]:
    # The numerical integration 
    
    deriv_wrapper = partial(_deriv_fssh_dm, cache=cache, hamiltonian=hamiltonian, basis_rep=basis_rep)
                            
    t, s = state_rk4(t, s, deriv_wrapper, dt=dt)
    
    # the callback functions: surface hopping
    new_s, new_cache = callback(t, s, cache, langevin, hamiltonian, basis_rep, dt)
    return t, new_s, new_cache
   
def hopping(
    dt: float,
    rho: NDArray[np.complex128],
    hami_return: HamiltonianRetureType,
    P: NDArray[np.float64],
    mass: Optional[NDArray[np.float64]],
    active_surf: ActiveSurface,
) -> Tuple[bool, ActiveSurface, NDArray[np.float64]]:
    ############################
    # The surface hopping algorithm
    ############################
    
    # compute the hopping probabilities
    v = P / mass
    vdotd = v_dot_d(v, hami_return.d)
    prob_vec = np.zeros(rho.shape[0])
    prob_vec[active_surf.index] = 1.0
    prob_vec = _evaluate_hopping_prob(dt, active_surf.index, rho, vdotd, prob_vec)
    
    # use random number to determine the hopping 
    to = _hopping(prob_vec)
    
    # if hopping happens, update the active surface, rescale the momentum
    if to == active_surf.index:
        return False, active_surf, P
    else:
        evals, d = hami_return.evals, hami_return.d
        allowed_hopping, P_rescaled = momentum_rescale(active_surf.index, to, evals, d, P, mass)
        return allowed_hopping, ActiveSurface(index=to), P_rescaled

@jit(nopython=True)
def _hopping_prob(
    from_: int,
    to_: int, 
    dt: float,
    vdotd: NDArray[np.complex128],
    rho: NDArray[np.complex128],
) -> float:
    # prob: float = -2.0 * dt * np.real(rho[to_, from_] * vdotd[from_, to_] / rho[from_, from_])
    prob: float = 2.0 * dt * np.real(rho[to_, from_] * vdotd[from_, to_] / rho[from_, from_])
    # prob: float = 2.0 * dt * np.real(rho[to_, from_] * vdotd[to_, from_] / rho[from_, from_])
    return prob if prob > 0 else 0.0

@jit(nopython=True)
def _evaluate_hopping_prob(
    dt: float, 
    from_: int,
    rho: NDArray[np.complex128],
    v_dot_d: NDArray[np.complex128],
    prob_vec: NDArray[np.float64],
) -> NDArray[np.float64]:
    # tr_rho: float = np.trace(rho).real
    for to_ in range(prob_vec.shape[0]): 
        if from_ == to_:
            continue
        # prob_vec[to_] = _hopping_prob(from_, to_, dt, v_dot_d, rho) / tr_rho
        prob_vec[to_] = _hopping_prob(from_, to_, dt, v_dot_d, rho)
        prob_vec[from_] -= prob_vec[to_]
    return prob_vec

@jit(nopython=True)
def _hopping(
    prob_vec: NDArray[np.float64]
) -> int:
    # prepare the variables
    accumulative_prob: float = 0.0
    to: int = 0
    
    # generate a random number
    random_number = np.random.rand()
    while (to < prob_vec.shape[0]):
        accumulative_prob += prob_vec[to]
        if accumulative_prob > random_number:
            break
        to += 1
    return to

@jit(nopython=True)
def _calculate_diabatic_populations(
    active_surf: int,
    another_surf: int,
    evecs: NDArray[np.complex128],
    rho: NDArray[np.complex128],
) -> float:
    """Calculate the diabatic population of a state in surface hopping algorithm. Here active_surf is the current active surface, another_surf is the target surface, evecs is the eigenvectors, rho is the density matrix, and diab_populations is the diabatic populations.

    Args:
        active_surf (int): the index of the current active surface
        another_surf (int): the index of the target surface
        evecs (ArrayLike): the eigenvectors
        rho (ArrayLike): the density matrix

    Returns:
        float: the diabatic population when active_surf is the current active surface and another_surf is the target surface
    """
    ret_val: float = np.abs(evecs[another_surf, active_surf])**2 
    for ii in range(rho.shape[0]):
        for jj in range(ii+1, rho.shape[0]):
            ret_val += 2.0 * evecs[another_surf, ii] * evecs[another_surf, jj] * np.real(rho[ii, jj])
    return ret_val

@jit(nopython=True)
def calculate_adiabatic_populations(
    active_surf: int,
    evecs: NDArray[np.complex128],
    rho: NDArray[np.complex128],
) -> NDArray[np.float64]:
    population = np.zeros(rho.shape[0])
    for ii in range(rho.shape[0]):
        population[ii] = _calculate_diabatic_populations(active_surf, ii, evecs, rho)
    return population
        

def momentum_rescale(
    from_: int,
    to_: int,
    evals: NDArray[np.float64],
    d: NDArray[np.complex128],
    P_current: NDArray[np.float64],
    mass: Optional[NDArray[np.float64]],
) -> Tuple[bool, NDArray[np.float64]]:
    # determine the rescaling direction
    d_component = d[:, to_, from_] # I use the convention (i, j, k) for the index
                                   # i: the index of the classical degree of freedom
                                   # j, k: the index of the electronic states
                                   
    normalized_direction = d_component 
    M_inv = 1.0 / mass 
    dE = evals[to_] - evals[from_]
    
    # solve the quadratic equation
    a = 0.5 * np.sum(M_inv * normalized_direction**2)
    b = np.vdot(P_current * M_inv, normalized_direction)
    c = dE
    b2_4ac = b**2 - 4 * a * c
    if b2_4ac < 0:
        return False, P_current
    elif b < 0:
        gamma: float = (b + np.sqrt(b2_4ac)) / (2 * a)
        P_rescaled = P_current - gamma * normalized_direction
        # dPE = np.sum((P_rescaled**2 - P_current**2) * M_inv) / 2
        # print(f"{dE=}, {dPE=}, {mass=}")
        return True, P_rescaled
    elif b >= 0:
        # print(f"the gamma is {gamma}")
        gamma: float = (b - np.sqrt(b2_4ac)) / (2 * a)
        P_rescaled = P_current - gamma * normalized_direction
        # dPE = np.sum((P_rescaled**2 - P_current**2) * M_inv) / 2
        # print(f"{dE=}, {dPE=}, {mass=}")
        return True, P_rescaled
    else:
        return False, P_current

def calculate_properties(
    t: float, 
    s: State, 
    cache: Cache, 
    hamiltonian: HamiltonianBase,
    basis_rep: BasisRepresentation
) -> FSSHProperties:
    # unpack the variables 
    R, P, rho = s.get_variables()
    active_surf, hami_return = cache.active_surface, cache.hami_return
    
    # type annotations
    active_surf: int
    hami_return: HamiltonianRetureType
    
    H_diab, evals, evecs = hami_return.H, hami_return.evals, hami_return.evecs
    
    # calculate the kinetic energy and potential energy 
    KE = 0.5 * np.sum(P**2 / s.get_mass())
    PE = evals[active_surf]
    if isinstance(hamiltonian, QuasiFloquetHamiltonianBase):
        NF: int = hamiltonian.NF
        Omega: float = hamiltonian.get_carrier_frequency()
        _, pop_adiab = get_rho_and_populations(t, active_surf, H_diab, evecs, NF, Omega, F_basis=BasisRepresentation.Adiabatic, target_basis=BasisRepresentation.Adiabatic)
        _, pop_diab = get_rho_and_populations(t, active_surf, H_diab, evecs, NF, Omega, F_basis=BasisRepresentation.Adiabatic, target_basis=BasisRepresentation.Diabatic)
    else:
        # calculate the adiabatic populations
        pop_adiab = np.zeros_like(evals)
        pop_adiab[active_surf] = 1.0
        # calculate the diabatic populations
        pop_diab = calculate_adiabatic_populations(active_surf, evecs, rho)
    
    return FSSHProperties(KE=KE, PE=PE, diab_populations=pop_diab, adiab_populations=pop_adiab, active_surf=active_surf)

@jit(nopython=True)
def _initialize_adiabatic_active_surf(
    rho_adiab: NDArray[np.complex128]
) -> int: 
    random_number: float = np.random.rand()
    init_state: int = 0
    accumulative_prob: float = 0.0
    tr_rho: float = np.trace(rho_adiab).real # normalize the probability
    while (init_state < rho_adiab.shape[0]):
        accumulative_prob += rho_adiab[init_state, init_state].real / tr_rho
        if accumulative_prob > random_number:
            break
        init_state += 1
    return init_state

def initialize_active_surf(
    rho: NDArray[np.complex128]
) -> ActiveSurface:
    active_surf = _initialize_adiabatic_active_surf(rho)
    return ActiveSurface(index=active_surf)

# def initialize_cache(t: float, s: State, hamiltonian: HamiltonianBase, basis_rep: BasisRepresentation) -> FSSHCache:
#     R, _, rho = s.get_variables()
#     hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
#     return FSSHCache(active_surf=initialize_active_surf(rho), evals=hami_return.evals, evecs=hami_return.evecs, H_diab=hami_return.H, hamiltonian=hamiltonian)

def initialize_cache(
    t: float, 
    s: State, 
    langevin_force: NDArray[np.float64],
    hamiltonian: HamiltonianBase, 
    basis_rep: BasisRepresentation, 
) -> Cache:
    R, P, rho = s.get_variables()
    hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
    langevin_force = langevin_force if langevin_force is not None else np.zeros_like(P)
    i_active_surface = initialize_active_surf(rho).index
    return Cache(hami_return=hami_return, active_surface=i_active_surface, F_langevin=langevin_force)

def callback(
    t: float, 
    s: State, 
    cache: Cache,
    langevin: LangevinBase,
    hamiltonian: HamiltonianBase, 
    basis_rep: BasisRepresentation, 
    dt: float,
) -> Tuple[State, Cache]:
    R, P, rho = s.get_variables()
    hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
    hamiltonian.update_last_evecs(hami_return.evecs) 
   
    # the hopping algorithm 
    prev_active_surf = ActiveSurface(index=cache.active_surface)
    mass = s.get_mass()
    has_hopped, new_active_surf, P_rescaled = hopping(dt, rho, hami_return, P, mass, prev_active_surf)
    if has_hopped:
        s.set_P(P_rescaled)
        if langevin is not None:
            F_langevin = langevin.evaluate_langevin(t, R, P, dt)
        else:
            F_langevin = np.zeros_like(P)
        return s, Cache(hami_return=hami_return, active_surface=new_active_surf.index, F_langevin=F_langevin)
    else:
        if langevin is not None:
            F_langevin = langevin.evaluate_langevin(t, R, P, dt)
        else:
            F_langevin = np.zeros_like(P)
        return s, Cache(hami_return=hami_return, active_surface=new_active_surf.index, F_langevin=F_langevin)
    


# %%
