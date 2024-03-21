# %%
import numpy as np
import numpy.linalg as LA
from numpy.typing import ArrayLike
from numba import jit
from scipy.integrate import ode

from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, QuasiFloquetHamiltonianBase
from pymddrive.integrators.state import State, zeros_like
from pymddrive.integrators.rk4 import rk4
from pymddrive.dynamics.options import BasisRepresentation, QunatumRepresentation, NumericalIntegrators
from pymddrive.dynamics.misc_utils import eval_nonadiabatic_hamiltonian, HamiltonianRetureType
from pymddrive.dynamics.math_utils import rhs_density_matrix, v_dot_d
from pymddrive.dynamics.floquet.fssh import get_rho_and_populations

from typing import Tuple, List, Callable
from collections import namedtuple 
from dataclasses import dataclass, field

FSSHCache = namedtuple('FSSHCache', 'active_surf, evals, evecs, H_diab, hamiltonian')
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

def _R_dot(v: ArrayLike) -> ArrayLike:
    return v

def _P_dot(F_active_surf: ArrayLike) -> ArrayLike:
    return F_active_surf

def _rho_dot_adiab(
    rho: ArrayLike, evals: ArrayLike, v: ArrayLike, d: ArrayLike
) -> ArrayLike:
    return rhs_density_matrix(rho=rho, evals=evals, vdotd=v_dot_d(v,d))

def _rho_dot_diab(
    rho: ArrayLike, v: ArrayLike, d: ArrayLike, evecs: ArrayLike
) -> ArrayLike:
    raise NotImplementedError(f"the surface hopping algorithm for diabatic representation is not implemented yet.")

def _deriv_fssh_dm(
    t: float,
    s: State, 
    hamiltonian: HamiltonianBase,
    mass: ArrayLike,
    basis_rep: BasisRepresentation,
    active_surface: ActiveSurface,
) -> State:
    R, P, rho = s.get_variables()
    out = zeros_like(s)
    kR, kP, krho = out.get_variables()
    i_active_surf = active_surface.index
    
    # integrate R
    v = P / mass
    kR[:] = _R_dot(v)
    
    # integrate P
    hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
    F_active_surf = hami_return.F[i_active_surf]
    kP[:] = _P_dot(F_active_surf)
    
    # integrate the density matrix
    if basis_rep is BasisRepresentation.Adiabatic:
        krho[:] = _rho_dot_adiab(rho, hami_return.evals, v, hami_return.d)
    else:
        krho[:] = _rho_dot_diab(rho, v, hami_return.d, hami_return.evecs)
    return out

def step_rk(
    t: float,
    s: State,
    cache: FSSHCache,
    dt: float,
    hamiltonian: HamiltonianBase,
    mass: ArrayLike,
    basis_rep: BasisRepresentation,
) -> Tuple[float, State, FSSHCache]:
    # The numerical integration 
    deriv_options = {
        'hamiltonian': hamiltonian,
        'mass': mass,
        'basis_rep': basis_rep,
        'active_surface': cache.active_surf
    }
    t, s = rk4(t, s, _deriv_fssh_dm, deriv_options, dt)
    # The callback functions: Updating cache 
    R, P, rho = s.get_variables()
    hami_return: HamiltonianRetureType = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
    hamiltonian.update_last_evecs(hami_return.evecs)
    hamiltonian.update_last_deriv_couplings(hami_return.d)
    
    # the callback functions: surface hopping
    _, new_active_surf, P_rescaled = hopping(dt, rho, hami_return, P, mass, cache.active_surf)
    P[:] = P_rescaled
    return t, s, FSSHCache(active_surf=new_active_surf, evals=hami_return.evals, evecs=hami_return.evecs, H_diab=hami_return.H, hamiltonian=hamiltonian)
   
def hopping(
    dt: float,
    rho: ArrayLike,
    hami_return: HamiltonianRetureType,
    P: ArrayLike,
    mass: ArrayLike,
    active_surf: ActiveSurface,
) -> Tuple[bool, ActiveSurface, ArrayLike]:
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
    vdotd: ArrayLike,
    rho: ArrayLike,
) -> float:
    # prob: float = -2.0 * dt * np.real(rho[to_, from_] * vdotd[from_, to_] / rho[from_, from_])
    prob: float = 2.0 * dt * np.real(rho[to_, from_] * vdotd[from_, to_] / rho[from_, from_])
    # prob: float = 2.0 * dt * np.real(rho[to_, from_] * vdotd[to_, from_] / rho[from_, from_])
    return prob if prob > 0 else 0.0

@jit(nopython=True)
def _evaluate_hopping_prob(
    dt: float, 
    from_: int,
    rho: ArrayLike,
    v_dot_d: ArrayLike,
    prob_vec: ArrayLike,
) -> ArrayLike:
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
    prob_vec: ArrayLike,
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
    evecs: ArrayLike,
    rho: ArrayLike,
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
    evecs: ArrayLike,
    rho: ArrayLike,
) -> ArrayLike:
    population = np.zeros(rho.shape[0])
    for ii in range(rho.shape[0]):
        population[ii] = _calculate_diabatic_populations(active_surf, ii, evecs, rho)
    return population
        

def momentum_rescale(
    from_: int,
    to_: int,
    evals: ArrayLike,
    d: ArrayLike,
    P_current: ArrayLike,
    mass: ArrayLike,
) -> Tuple[bool, ArrayLike]:
    # determine the rescaling direction
    d_component = d[:, to_, from_] # I use the convention (i, j, k) for the index
                                   # i: the index of the classical degree of freedom
                                   # j, k: the index of the electronic states
                                   
    # d_component_norm = LA.norm(d_component)
    # normalized_direction = d_component / d_component_norm 
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

    # roots = np.roots([a, b, c])
    # print(f"the roots are {roots}")
    # if np.iscomplexobj(roots):
    #     return False, P_current
    # else:
    #     gamma = np.min(roots)
    #     P_rescaled = P_current - gamma * normalized_direction 
    #     return True, P_rescaled
    
def calculate_properties(t: float, s: State, cache: FSSHCache, mass: ArrayLike) -> FSSHProperties:
    R, P, rho = s.get_variables()
    active_surf, evals, evecs, H_diab, hamiltonian = cache
    KE = 0.5 * np.sum(P**2 / mass)
    PE = evals[active_surf.index]
    if isinstance(hamiltonian, QuasiFloquetHamiltonianBase):
        NF: int = hamiltonian.NF
        Omega: float = hamiltonian.get_carrier_frequency()
        active_surf: ActiveSurface
        _, pop_adiab = get_rho_and_populations(t, active_surf.index, H_diab, evecs, NF, Omega, F_basis=BasisRepresentation.Adiabatic, target_basis=BasisRepresentation.Adiabatic)
        _, pop_diab = get_rho_and_populations(t, active_surf.index, H_diab, evecs, NF, Omega, F_basis=BasisRepresentation.Adiabatic, target_basis=BasisRepresentation.Diabatic)
    else:
        # calculate the adiabatic populations
        pop_adiab = np.zeros_like(evals)
        pop_adiab[active_surf.index] = 1.0
        # calculate the diabatic populations
        pop_diab = calculate_adiabatic_populations(active_surf.index, evecs, rho)
    
    return FSSHProperties(KE=KE, PE=PE, diab_populations=pop_diab, adiab_populations=pop_adiab, active_surf=active_surf.index)

@jit(nopython=True)
def _initialize_adiabatic_active_surf(
    rho_adiab: ArrayLike,
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
    rho: ArrayLike,
) -> ActiveSurface:
    active_surf = _initialize_adiabatic_active_surf(rho)
    return ActiveSurface(index=active_surf)

def initialize_cache(t: float, s: State, hamiltonian: HamiltonianBase, basis_rep: BasisRepresentation) -> FSSHCache:
    R, _, rho = s.get_variables()
    hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
    return FSSHCache(active_surf=initialize_active_surf(rho), evals=hami_return.evals, evecs=hami_return.evecs, H_diab=hami_return.H, hamiltonian=hamiltonian)

# def calculate_cache(t: float, s: State, cache: FSSHCache, hamiltonian: HamiltonianBase, basis_rep: BasisRepresentation) -> FSSHCache:
#     R, P, rho = s.get_variables()
#     hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
#     return FSSHCache(active_surf=cache.active_surf, evals=hami_return.evals, evecs=hami_return.evecs)

# %%

def estimate_delay_time(r0, p0, mass: float=2000.0):
    from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes
    from pymddrive.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics, NonadiabaticDynamicsMethods
    hamiltonian = get_tullyone(
        pulse_type=TullyOnePulseTypes.NO_PULSE
    )
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    dyn = NonadiabaticDynamics(
        hamiltonian=hamiltonian,
        t0=0.0,
        s0=s0,
        mass=mass,
        basis_rep=BasisRepresentation.Diabatic,
        qm_rep=QunatumRepresentation.DensityMatrix,
        solver=NonadiabaticDynamicsMethods.EHRENFEST,
        numerical_integrator=NumericalIntegrators.ZVODE,
        dt=1,
        save_every=1
    )
    def stop_condition(t, s, states):
        r, p, _ = s.get_variables()
        return (r>0.0) or (p<0.0)
    break_condition = lambda t, s, states: False
    res = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    return res['time'][-1]

def get_floquet_rho0(rho0: np.ndarray, NF: int):
    import scipy.sparse as sp
    data = [rho0]
    indptr = np.concatenate((np.zeros(NF+1), np.ones(NF+1))).astype(int)
    indicies = np.array([NF])
    dimF = (2*NF+1) * rho0.shape[0]
    rho0_floquet_bsr = sp.bsr_matrix((data, indicies, indptr), shape=(dimF, dimF), dtype=np.complex128)
    return rho0_floquet_bsr.toarray()

def wrap_deriv_fssh_dm(
    t: float, 
    y: ArrayLike,
    hamiltonian: HamiltonianBase, 
    mass: ArrayLike, 
    basis_rep: BasisRepresentation,
    active_surface: ActiveSurface,
    stype,
    dtype,
) -> ArrayLike:
    s = State.from_unstructured(y, dtype=dtype, stype=stype)
    dsdt = _deriv_fssh_dm(t, s, hamiltonian, mass, basis_rep, active_surface)
    return dsdt.flatten()

#FSSH_Dynamics_data = namedtuple('FSSH_Dynamics_data', 'time, state, hamiltonian, cache')
@dataclass
class FSSH_Dynamics_data:
    time: float
    state: State
    hamiltonian: HamiltonianBase
    cache: FSSHCache
    
    def __iter__(self):
        yield self.time
        yield self.state
        yield self.hamiltonian
        yield self.cache
        
    def __getitem__(self, index: int):
        if index == 0:
            return self.time
        elif index == 1:
            return self.state
        elif index == 2:
            return self.hamiltonian
        elif index == 3:
            return self.cache
        else:
            raise IndexError(f"Index {index} is out of range.")
        
    def __tuple__(self):
        return (self.time, self.state, self.hamiltonian, self.cache)
    
def _ehrenfest_benchmark():
    from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes, TD_Methods
    from pymddrive.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics, NonadiabaticDynamicsMethods
    t0 = 0.0    
    r0 = -5.0; p0 = 30.0; rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    _delay_time = estimate_delay_time(r0, p0)
    Omega = 0.3; tau = 100
    mass = 2000.0
    
    hamiltonian = get_tullyone(
        # t0=_delay_time, Omega=Omega, tau=tau,
        pulse_type=TullyOnePulseTypes.NO_PULSE
        # td_method=TD_Methods.BRUTE_FORCE
    )
    dyn = NonadiabaticDynamics(
        hamiltonian=hamiltonian,
        t0=t0,
        s0=s0,
        mass=mass,
        basis_rep=BasisRepresentation.Adiabatic,
        qm_rep=QunatumRepresentation.DensityMatrix,
        solver=NonadiabaticDynamicsMethods.EHRENFEST,
        numerical_integrator=NumericalIntegrators.ZVODE,
        dt=0.03,
        save_every=30
    )
    
    def stop_condition(t, s, states):
        r, p, _ = s.get_variables()
        return (r>5.0) or (r<-5.0)
    break_condition = lambda t, s, states: False
    return run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)

def _step_zvode(t: float, s: State, hamiltonian: HamiltonianBase, mass: ArrayLike, basis_rep: BasisRepresentation, cache: FSSHCache, dt: float, ode_solver: ode) -> Tuple[float, State, FSSHCache]:
    if not ode_solver.successful():
        raise RuntimeError("The ode solver is not successful.")
    if t != ode_solver.t:
        raise ValueError(f"The time {t} is not the same as the solver time {ode_solver.t}.")
    ode_solver.set_f_params(hamiltonian, mass, basis_rep, cache.active_surf, s.stype, s.data.dtype)
    ode_solver.integrate(ode_solver.t + dt)
    state = State.from_unstructured(ode_solver.y, dtype=s.data.dtype, stype=s.stype)
    R, P, rho = state.get_variables()
    hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
    hamiltonian.update_last_evecs(hami_return.evecs)
    # hamiltonian.update_last_deriv_couplings(hami_return.d)
    has_hopped, new_active_surf, P_rescaled = hopping(dt, rho, hami_return, P, mass, cache.active_surf)
    if has_hopped:
        P[:] = P_rescaled
        ode_solver.set_initial_value(state.flatten(), t)
    cache = FSSHCache(active_surf=new_active_surf, evals=hami_return.evals, evecs=hami_return.evecs, H_diab=hami_return.H, hamiltonian=hamiltonian)
    return ode_solver.t, state, cache

def _debug_test_run_one_fssh():
    from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes, TD_Methods
    NF: int = 1
    t0 = 0.0
    r0 = -5.0; p0 = 30.0; rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    # rho0_floquet = get_floquet_rho0(rho0, NF=1)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    
    _delay_time = estimate_delay_time(r0, p0)
    Omega = 0.3; tau = 100
    mass = 2000.0  
    
    hamiltonian = get_tullyone(
        # t0=_delay_time, Omega=Omega, tau=tau,
        pulse_type=TullyOnePulseTypes.NO_PULSE
        # td_method=TD_Methods.FLOQUET, NF=NF
    )
    R0, P0, rho0 = s0.get_variables()   
    hami_return = eval_nonadiabatic_hamiltonian(t0, R0, hamiltonian, BasisRepresentation.Adiabatic)
    active_index = np.diag(rho0).real.argmax()
    cache = FSSHCache(active_surf=ActiveSurface(index=active_index), evals=hami_return.evals, evecs=hami_return.evecs, H_diab=hami_return.H, hamiltonian=hamiltonian)
    
    time_out = np.array([])
    traj_out = None
    property_dict = {field: [] for field in FSSHProperties._fields}
    
    def _append_properties(properties_dict: dict, properties: FSSHProperties) -> dict:
        for (field, value) in zip(FSSHProperties._fields, properties):
            properties_dict[field].append(value)
        return properties_dict
            
    def stop_condition(t, s, states):
        r, p, _ = s.get_variables()
        return (r>5.0) or (r<-5.0)
    
    dt = 0.1
    t, s = t0, s0
    
    
    ode_solver = ode(wrap_deriv_fssh_dm).set_integrator('zvode', method='bdf')
    ode_solver.set_initial_value(s.flatten(), t)
    
    for ns in range(int(1e6)):
        if ns % 10 == 0:
            properties = calculate_properties(t, s, cache, mass)
            time_out = np.append(time_out, t)   
            traj_out = np.array([s.data]) if traj_out is None else np.append(traj_out, s.data)
            property_dict = _append_properties(property_dict, properties)
            if stop_condition(t, s, traj_out):
                break
        t, s, cache = _step_zvode(t, s, hamiltonian, mass, BasisRepresentation.Adiabatic, cache, dt, ode_solver)
        
    for (field, value) in property_dict.items():
        property_dict[field] = np.array(value)
    return time_out, traj_out, property_dict 

def _para_run_fssh(ntraj: int):
    from joblib import Parallel, delayed
    ensemble_out = Parallel(n_jobs=-1,verbose=5)(
        delayed(_debug_test_run_one_fssh)() for _ in range(ntraj)
    )
    return reduce_ensemble(ensemble_out, ensemble_length=float(ntraj))

def get_shortest_time_length(ensemble_out: List[Tuple[np.ndarray, np.ndarray, dict]]):
    len_time = int(1e10)
    for time_out, _, _ in ensemble_out:
        if len(time_out) < len_time:
            len_time = len(time_out)
    return len_time
    
def reduce_ensemble(ensemble_out: List[Tuple[np.ndarray, np.ndarray, dict]], ensemble_length: int):
    len_time = get_shortest_time_length(ensemble_out)
    time_reduced = None
    traj_reduced = None
    property_reduced = {field: None for field in FSSHProperties._fields}
    for i, (time_out, traj_out, property_dict) in enumerate(ensemble_out):
        if time_reduced is None:
            time_reduced = time_out[:len_time]
        if traj_reduced is None:
            traj_reduced = traj_out[:len_time]
        else:
            for name in traj_out.dtype.names:
                traj_reduced[name] += traj_out[name][:len_time]
        for (field, value) in property_dict.items():  
            property_reduced[field] = value[:len_time] if property_reduced[field] is None else property_reduced[field] + value[:len_time]
    for name in traj_reduced.dtype.names:
        traj_reduced[name] /= ensemble_length
    for (field, value) in property_reduced.items():
        property_reduced[field] = np.array(value) / ensemble_length
    return time_reduced, traj_reduced, property_reduced
        
        

def _plot_all(time_out, traj_out, property_dict, ref_out, fixed_dt: float=None):
    import matplotlib.pyplot as plt
    if fixed_dt is not None: 
        time_out = np.arange(len(time_out)) / fixed_dt
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time_out, traj_out['R'], label='R')
    ax.plot(ref_out['time'], ref_out['states']['R'], label='R_ref')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('R (a.u.)')
    ax.legend()
    plt.show()

    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time_out, traj_out['P'], label='P')
    ax.plot(ref_out['time'], ref_out['states']['P'], label='P_ref')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('P (a.u.)')
    ax.legend()
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)   
    pop_out = property_dict['diab_populations']
    ax.plot(time_out, pop_out, label='pop0')
    ax.plot(time_out, pop_out, label='pop1')
    ax.plot(ref_out['time'], ref_out['diab_populations'][:, 0], label='pop0_ref')
    ax.plot(ref_out['time'], ref_out['diab_populations'][:, 1], label='pop1_ref')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Population')
    ax.legend()
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)   
    ax.plot(time_out, property_dict['PE'], label='PE')
    ax.plot(ref_out['time'], ref_out['PE'], label='PE_ref')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Potential Energy (a.u.)')
    ax.legend()
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)   
    ax.plot(time_out, property_dict['KE'], label='KE')
    ax.plot(ref_out['time'], ref_out['KE'], label='KE_ref')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Kinetic Energy (a.u.)')
    ax.legend()
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)   
    TE = property_dict['KE'] + property_dict['PE']
    TE_ref = ref_out['KE'] + ref_out['PE']
    ax.plot(time_out, TE, label='TE')
    ax.plot(ref_out['time'], TE_ref, label='TE_ref')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Total Energy (a.u.)')
    ax.legend()
    plt.show()

    
# %%
if __name__ == '__main__':
    import time
    start = time.perf_counter()
    # time_out, traj_out, property_dict = _debug_test_run_one_fssh()
    # time_out, traj_out, property_out = _debug_test_fssh(ntraj=2)
    ntraj = 1000
    time_out, traj_out, property_dict = _para_run_fssh(ntraj=ntraj)
    print(f"Time: {time.perf_counter()-start:.4f} s")
    ref_out = _ehrenfest_benchmark() 
    
# %%
if __name__ == '__main__':
    ref_out = _ehrenfest_benchmark() 
    print(ref_out['time'][-1])
   
# %%    
if __name__ == '__main__':
    _plot_all(time_out, traj_out, property_dict, ref_out)
    
# %%
