# %% 
import numpy as np
from numpy.typing import NDArray
from numba import jit

from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, QuasiFloquetHamiltonianBase
from pymddrive.models.nonadiabatic_hamiltonian import diabatic_to_adiabatic, adiabatic_to_diabatic
from pymddrive.low_level.states import State
from pymddrive.integrators.rk4 import rk4
from pymddrive.integrators.state_rk4 import state_rk4
from pymddrive.dynamics.options import BasisRepresentation, QunatumRepresentation, NumericalIntegrators
from pymddrive.dynamics.misc_utils import eval_nonadiabatic_hamiltonian, HamiltonianRetureType
from pymddrive.dynamics.math_utils import commutator, rhs_density_matrix, v_dot_d, expected_value
from pymddrive.dynamics.floquet.ehrenfest import get_rho_and_populations
from pymddrive.dynamics.cache import Cache
from pymddrive.dynamics.langevin import LangevinBase

from typing import Union, Callable, Optional, Tuple
from collections import namedtuple  
from functools import partial


# EhrenfestCache = namedtuple('EhrenfestCache', 'meanF, evals, evecs')
# for debugging
# EhrenfestProperties = namedtuple('EhrenfestProperties', 'KE, PE, populations, meanF, dij') 
EhrenfestProperties = namedtuple('EhrenfestProperties', 'KE, PE, diab_populations, adiab_populations, meanF, dij')
# for production
# EhrenfestProperties = namedtuple('EhrenfestProperties', 'KE, PE, diab_populations, adiab_populations')

def choose_ehrenfest_stepper(
    numerical_integrator: NumericalIntegrators
) -> Callable:
    if numerical_integrator == NumericalIntegrators.RK4:
        return step_rk
    elif numerical_integrator == NumericalIntegrators.VVRK4:
        # return step_vv_rk
        raise NotImplementedError
    elif numerical_integrator == NumericalIntegrators.VVRK4_GPAW:
        return step_vv_rk_gpaw
    else:
        # raise NotImplementedError(f"Numerical integrator {numerical_integrator} is not implemented for Ehrenfest dynamics.")
        raise NotImplementedError
    
def choose_ehrenfest_deriv(
    quantum_representation: QunatumRepresentation
) -> Callable:
    if quantum_representation == QunatumRepresentation.DensityMatrix:
        return _deriv_ehrenfest_dm
    elif quantum_representation == QunatumRepresentation.Wavefunction:
        raise NotImplementedError(f"Quantum representation Wavefunction is not implemented for Ehrenfest dynamics.")
    else:
        raise NotImplementedError(f"Quantum representation {quantum_representation} is not implemented for Ehrenfest dynamics.")


# def _rho_dot(t, rho, R, v, hamiltonian, basis_rep) -> NDArray[np.complex128]:
#     hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
#     if basis_rep == BasisRepresentation.Adiabatic:
#         return _rho_dot_adiab(rho, hami_return.evals, v, hami_return.d)
#     elif basis_rep == BasisRepresentation.Diabatic:
#         return _rho_dot_diab(rho, hami_return.H)

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
    return -1.j * commutator(H, rho)

def _deriv_ehrenfest_dm(
    t: float,
    s: State,
    cache: Cache,
    hamiltonian: HamiltonianBase,
    basis_rep: BasisRepresentation,
) -> State:
    dsdt = s.zeros_like()
    # kR, kP, k_rho = out.get_variables()
    
    # integrate the position
    v = s.get_v()
    dsdt.set_R(v)
    
    # integrate the momentum
    meanF, hami_return = compute_ehrenfest_meanF(t, s.get_R(), s.get_rho(), hamiltonian, basis_rep)
    F_langevin = cache.F_langevin
    dsdt.set_P(meanF + F_langevin)
    
    # integrate the density matrix
    krho = _rho_dot(s.get_rho(), v, hami_return, basis_rep)
    dsdt.set_rho(krho)
    
    return dsdt
    

def step_rk(
    t: float,
    s: State,
    cache: Cache,
    dt: float,
    hamiltonian: HamiltonianBase,
    langevin: LangevinBase,
    basis_rep: BasisRepresentation,
):
    # numerical Integration
    deriv_wrapper = partial(_deriv_ehrenfest_dm, cache=cache, hamiltonian=hamiltonian, basis_rep=basis_rep)
    t, s = state_rk4(t, s, deriv_wrapper, dt=dt)
    
    # callback after one step
    new_s, new_cache = callback(t, s, cache, langevin, hamiltonian, basis_rep, dt)
    return t, new_s, new_cache

def step_vv_rk(
    t: float,
    s: State,
    cache: Cache,
    dt: float,
    langevin: LangevinBase,
    hamiltonian: HamiltonianBase,
    basis_rep: BasisRepresentation,
):
    if cache is not None:
        meanF = cache.meanF
    else:
        meanF, hami_return = compute_ehrenfest_meanF(t, s.get_R(), s.get_rho(), hamiltonian, basis_rep)
        
    F_langevin = cache.F_langevin
        
    # First half of momentum integration
    s.set_P(s.get_P() + 0.5 * dt * (meanF + F_langevin))
        
    # Full position integration
    v = s.get_v()
    s.set_R(s.get_R() + dt * v)
    
    # for _ in range(n_qm_steps): 
    # rk4_options = {'R': R, 'v': v, "hamiltonian": hamiltonian, "basis_rep": basis_rep} 
    # _, rho[:] = rk4(t, rho, _rho_dot, rk4_options, dt=dt)
    def deriv_wrapper(t, rho) -> Tuple[float, NDArray[np.complex128]]:
        hami_return = eval_nonadiabatic_hamiltonian(t, s.get_R(), hamiltonian, basis_rep)
        return _rho_dot(rho, v, hami_return, basis_rep)
    
    _, new_rho = rk4(t, s.get_rho(), deriv_wrapper, dt=dt)
    s.set_rho(new_rho)
    
    meanF, hami_return = compute_ehrenfest_meanF(t+dt, s.get_R(), s.get_rho(), hamiltonian, basis_rep)
    
    # Second half of momentum integration
    s.set_P(s.get_P() + 0.5 * dt * (meanF + F_langevin))
       
    # update the time 
    t += dt     
    new_s, new_cache = callback(t, s, cache, langevin, hamiltonian, basis_rep, dt)
    return t, new_s, new_cache

def step_vv_rk_gpaw(
    t: float,
    s: State,
    cache: Cache,
    dt: float,
    hamiltonian: HamiltonianBase,
    basis_rep: BasisRepresentation,
    langevin: LangevinBase,
):
    if cache is not None:
        meanF = cache.meanF
    else:
        meanF, hami_return = compute_ehrenfest_meanF(t, s.get_R(), s.get_rho(), hamiltonian, basis_rep)
        
    mass: Union[float, NDArray[np.float64]] = s.get_mass()
    
    F_langvin = cache.F_langevin 
    
    # The First half nuclear step
    s.set_R(s.get_R() + 0.5 * dt * s.get_v() + 0.25 * dt**2 * meanF / mass)
    s.set_P(s.get_P() + 0.25 * dt * (meanF + F_langvin))
    
    meanF, hami_return = compute_ehrenfest_meanF(t+0.5*dt, s.get_R(), s.get_rho(), hamiltonian, basis_rep)
    s.set_P(s.get_P() + 0.25 * dt * (meanF + F_langvin))
 
    # The full electronic step
    v = s.get_v()
    
    def deriv_wrapper(t, rho) -> Tuple[float, NDArray[np.complex128]]:
        hami_return = eval_nonadiabatic_hamiltonian(t, s.get_R(), hamiltonian, basis_rep)
        return _rho_dot(rho, v, hami_return, basis_rep)
    # for _ in range(n_qm_steps): 
    _, new_rho = rk4(t, s.get_rho(), deriv_wrapper, dt=dt)
    s.set_rho(new_rho)
   
    meanF = _compute_ehrenfest_meanF(s.get_rho(), hami_return.dHdR, hami_return.evals, hami_return.evecs, hami_return.d, hami_return.F, basis_rep)
    
    # the second half nuclear step
    s.set_R(s.get_R() + 0.5 * dt * s.get_v() + 0.25 * dt**2 * meanF / mass) 
    s.set_P(s.get_P() + 0.25 * dt * (meanF + F_langvin))
    meanF, hami_return = compute_ehrenfest_meanF(t+dt, s.get_R(), s.get_rho(), hamiltonian, basis_rep)
    s.set_P(s.get_P() + 0.25 * dt * (meanF + F_langvin))
    
    # update the time
    t += dt
    
    new_s, new_cache = callback(t, s, cache, langevin, hamiltonian, basis_rep, dt)
    
    return t, new_s, new_cache

def calculate_properties(
    t: float,
    s: State, 
    cache: Cache, 
    hamiltonian: HamiltonianBase,
    basis_rep: BasisRepresentation
):
    _, P, rho = s.get_variables()
    mass = s.get_mass()
    meanF, hami_return = cache.meanF, cache.hami_return
    hami_return: HamiltonianRetureType
    
    KE = 0.5 * np.sum(P**2 / mass)
    if isinstance(hamiltonian, QuasiFloquetHamiltonianBase):
        # print(np.sum(rho.diagonal().real))
        NF: int = hamiltonian.NF
        Omega: float = hamiltonian.get_carrier_frequency()
        _, pop_adiab = get_rho_and_populations(t, rho, hami_return.H, hami_return.evecs, NF, Omega, F_basis=basis_rep, target_basis=BasisRepresentation.Adiabatic)
        _, pop_diab = get_rho_and_populations(t, rho, hami_return.H, hami_return.evecs, NF, Omega, F_basis=basis_rep, target_basis=BasisRepresentation.Diabatic)
    else:
        if basis_rep == BasisRepresentation.Adiabatic:
            # PE = expected_value(s.data['rho'], hami_return.evals)
            pop_adiab = rho.diagonal().real.copy()
            pop_diab = adiabatic_to_diabatic(rho, hami_return.evecs).diagonal().real
        elif basis_rep == BasisRepresentation.Diabatic:
            # PE = expected_value(s.data['rho'], hami_return.H)
            pop_adiab = diabatic_to_adiabatic(rho, hami_return.evecs).diagonal().real
            pop_diab = rho.diagonal().real.copy()
    if basis_rep == BasisRepresentation.Adiabatic:
        PE = expected_value(rho, hami_return.evals)
    elif basis_rep == BasisRepresentation.Diabatic:
        PE = expected_value(rho, hami_return.H)
    else:
        raise NotImplemented
    # debugging output
    # PE = PE / (2.0 * hamiltonian.NF + 1.0) if isinstance(hamiltonian, QuasiFloquetHamiltonianBase) else PE
    return EhrenfestProperties(KE=KE, PE=PE, diab_populations=pop_diab, adiab_populations=pop_adiab, meanF=meanF, dij=hami_return.d)
    # production output
    # return EhrenfestProperties(KE=KE, PE=PE, diab_populations=pop_diab, adiab_populations=pop_adiab)

# Helper functions

def compute_ehrenfest_meanF(
    t: float,
    R: NDArray[np.float64],
    rho_or_psi: NDArray[np.complex128],
    hamiltonian: HamiltonianBase, 
    basis_rep: BasisRepresentation
) -> NDArray[np.float64]:
    hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
    meanF = _compute_ehrenfest_meanF(rho_or_psi, hami_return.dHdR, hami_return.evals, hami_return.evecs, hami_return.d, hami_return.F, basis_rep)
    return meanF, hami_return
    
def _compute_ehrenfest_meanF(
    rho_or_psi: NDArray[np.complex128],
    dHdR: NDArray[np.complex128],
    evals: NDArray[np.float64],
    evecs: NDArray[np.complex128],
    d: NDArray[np.complex128],
    F: NDArray[np.float64],
    basis_rep: BasisRepresentation
) -> NDArray[np.float64]:
    if basis_rep == BasisRepresentation.Adiabatic:
        # qm_diab = adiabatic_to_diabatic(qm, evecs)
        # return -1 * expected_value(qm_diab, dHdR)
        return _eval_Ehrenfest_meanF(evals, d, rho_or_psi, F)
    elif basis_rep == BasisRepresentation.Diabatic:
        return -1 * expected_value(rho_or_psi, dHdR)
    else:
        raise NotImplemented


@jit(nopython=True)
def _second_term_meanF(
    rho: NDArray[np.complex128],
    evals: NDArray[np.float64],
    d: NDArray[np.complex128],
    meanF: NDArray[np.float64]
) -> NDArray[np.float64]:
    for i in range(evals.shape[0]):
        for j in range(i+1, evals.shape[0]):
            meanF += 2.0 * (evals[j] - evals[i]) * np.real(rho[i, j] * d[:, j, i])
    return meanF

def second_term_meanF(
    rho: NDArray[np.complex128],
    evals: NDArray[np.float64],
    d: NDArray[np.complex128],
    meanF: Optional[NDArray[np.float64]] = None
) -> NDArray[np.float64]:
    if meanF is None:
        meanF = np.zeros(d.shape[0], dtype=np.float64)
        
    return _second_term_meanF(rho, evals, d, meanF)

def _eval_Ehrenfest_meanF(
    evals: NDArray[np.float64],
    d: NDArray[np.complex128],
    rho: NDArray[np.complex128],
    F: NDArray[np.float64]
):
    """ Nonadiabatic Dynamics: Mean-Field and Surface Hopping. Nikos L. Doltsinis. """
    """ https://juser.fz-juelich.de/record/152530/files/FZJ-2014-02134.pdf """
    meanF_term1 = expected_value(rho, F, is_diagonal=True) # population-weighted average force
    meanF_term2 = second_term_meanF(rho, evals, d)         # nonadiabatic changes of the adiabatic state occupations
    return meanF_term1 + meanF_term2

def initialize_cache(
    t: float, 
    s: State, 
    F_Langevin: NDArray[np.float64],
    hamiltonian: HamiltonianBase, 
    basis_rep: BasisRepresentation, 
) -> Cache:
    R, _, rho = s.get_variables()
    meanF, hami_return = compute_ehrenfest_meanF(t, R, rho, hamiltonian, basis_rep)
    return Cache(hami_return=hami_return, meanF=meanF, F_langevin=F_Langevin)

def callback(
    t: float, 
    s: State, 
    cache: Cache,
    langevin: Optional[LangevinBase],
    hamiltonian: HamiltonianBase, 
    basis_rep: BasisRepresentation, 
    dt: float,
) -> Cache:
    R, P, rho = s.get_variables()
    hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
    meanF = _compute_ehrenfest_meanF(rho, hami_return.dHdR, hami_return.evals, hami_return.evecs, hami_return.d, hami_return.F, basis_rep)
    hamiltonian.update_last_evecs(hami_return.evecs)
    # hamiltonian.update_last_deriv_couplings(hami_return.d)
    if langevin is not None:
        F_langevin = langevin.evaluate_langevin(t, R, P, dt)
    else:
        F_langevin = np.zeros_like(P)
    return s, Cache(hami_return=hami_return, meanF=meanF, F_langevin=F_langevin)
    
# %% the testing/debugging code

def _run_dynamics(algorithm: str, basis_rep: BasisRepresentation):
    from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes
    from pymddrive.integrators.state import get_state
    
    # get a convenient model for testing 
    model = get_tullyone(pulse_type=TullyOnePulseTypes.NO_PULSE)
    mass = 2000.0 
    
    t0 = 0.0
    R0 = -5.0; P0 = 30.0; rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    s0 = get_state(mass, R0, P0, rho0)
    
    r0, p0, rho0 = s0.get_variables()
   
    time_out = np.array([])
    traj_out = None 
    properties_dict = {field: [] for field in EhrenfestProperties._fields}
    
    def _append_properties(properties_dict: dict, properties: EhrenfestProperties) -> dict:
        for (field, value) in zip(EhrenfestProperties._fields, properties):
            properties_dict[field].append(value)
        return properties_dict

    
    t, s = t0, s0
    cache = None
    
    def stop_condition(t, s, states):
        r, _, _ = s.get_variables()
        return (r>5.0) or (r<-5.0)
    
    if algorithm == "rk4":
        stepper = step_rk
    elif algorithm == "vv":
        stepper = step_vv_rk
    else:
        stepper = step_vv_rk_gpaw
        
    dt = 0.03
    
    cache = initialize_cache(t, s, np.zeros_like(s.get_R()), model, basis_rep)      
    R_out = np.zeros((0, s.get_R().shape[0]))
    P_out = np.zeros((0, s.get_P().shape[0]))   
    rho_out = np.zeros((0, s.get_rho().shape[0], s.get_rho().shape[1]), dtype=np.complex128)
    for ns in range(int(1e6)):
        if ns % 100 == 0:
            properties = calculate_properties(t=t, s=s, cache=cache, hamiltonian=model, basis_rep=basis_rep)
            time_out = np.append(time_out, t)
            R_out = np.append(R_out, s.get_R().reshape(1, -1), axis=0)
            P_out = np.append(P_out, s.get_P().reshape(1, -1), axis=0)
            rho_out = np.append(rho_out, s.get_rho().reshape(1, s.get_rho().shape[0], s.get_rho().shape[1]), axis=0)
            
            properties_dict = _append_properties(properties_dict, properties)
            
            if stop_condition(t, s, traj_out):
                break
        t, s, cache = stepper(
            t=t, s=s, cache=cache, dt=dt, hamiltonian=model, langevin=None, basis_rep=basis_rep
        )
    properties_dict = {field: np.array(value) for (field, value) in properties_dict.items()}
    traj_out = {'R': R_out, 'P': P_out, 'rho': rho_out}
    return time_out, traj_out, properties_dict['KE'], properties_dict['PE'], properties_dict['adiab_populations'] 

def _plot_debug(t_out, s_out, KE_out, PE_out, pop_out, label: str, fig=None, ls='-'):
    import matplotlib.pyplot as plt
    if fig is None:
        fig, axs = plt.subplots(3, 2, figsize=(5*2, 3*3))
        axs_dynamics = axs[:, 0]
        axs_energy = axs[:, 1]
    else:
        axs = np.array(fig.get_axes()).reshape(3, 2)
        axs_dynamics = axs[:, 0]
        axs_energy = axs[:, 1]
        
    axs_dynamics[0].plot(t_out, s_out['R'], label=f"{label}: R", linestyle=ls)
    axs_dynamics[1].plot(t_out, s_out['P'], label=f"{label}: P", linestyle=ls)
    axs_dynamics[2].plot(t_out, pop_out[:, 0], label=f"{label}: rho00", linestyle=ls)
    axs_dynamics[2].plot(t_out, pop_out[:, 1], label=f"{label}: rho11", linestyle=ls)
    for ax in axs_dynamics:
        ax.legend()
        ax.set_xlabel("Time")
    
    axs_energy[0].plot(t_out, KE_out, label=f"{label}: KE", linestyle=ls)
    axs_energy[1].plot(t_out, PE_out, label=f"{label}: PE", linestyle=ls)
    TE = np.array([ke + pe for (ke, pe) in zip(KE_out, PE_out)])
    axs_energy[2].plot(t_out, TE/TE[0], label=f"{label} E/E0", linestyle=ls)
    for ax in axs_energy:
        ax.legend()
        ax.set_xlabel("Time") 
        
    return fig

def _test_debug():
    import time
    start = time.perf_counter()
    # t_out, s_out, KE_out_rk, PE_out_rk, pop_out_rk= _run_dynamics("rk4", basis_rep=BasisRepresentation.Adiabatic)
    t_out, s_out, KE_out_adiab, PE_out_adiab, pop_out_adiab= _run_dynamics("rk4", basis_rep=BasisRepresentation.Adiabatic)
    fig = _plot_debug(t_out, s_out, KE_out_adiab, PE_out_adiab, pop_out_adiab, "RK4 adiabatic")
    print(f"The simulation time for dynamics with adiabatic rk4 is {time.perf_counter()-start}")
    start = time.perf_counter()
    t_out, s_out, KE_out_diab, PE_out_diab, pop_out_diab= _run_dynamics("rk4", basis_rep=BasisRepresentation.Diabatic)
    fig = _plot_debug(t_out, s_out, KE_out_diab, PE_out_diab, pop_out_diab, "RK4 diabatic", fig, ls='-.')
    print(f"The simulation time for dynamics with diabatic rk4 is {time.perf_counter()-start}")
    start = time.perf_counter()
    t_out, s_out, KE_out_vv, PE_out_vv, pop_out_vv= _run_dynamics("vv", basis_rep=BasisRepresentation.Diabatic)
    fig = _plot_debug(t_out, s_out, KE_out_vv, PE_out_vv, pop_out_vv, "VV", fig)
    print(f"The simulation time for dynamics with simple vv-rk4 is {time.perf_counter()-start}")
    start = time.perf_counter()
    t_out, s_out, KE_out_vv_GPAW, PE_out_vv_GPAW, pop_out_vv_GPAW = _run_dynamics("vv_gpaw", basis_rep=BasisRepresentation.Diabatic)
    fig = _plot_debug(t_out, s_out, KE_out_vv_GPAW, PE_out_vv_GPAW, pop_out_vv_GPAW, "VV GPAW", fig)
    print(f"The simulation time for dynamics with vv-rk4-GPAW is {time.perf_counter()-start}")
    fig.tight_layout()

    

# %% the __main__ code for testing the package
if __name__ == "__main__":
    _test_debug()
       
# %%