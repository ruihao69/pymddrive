# %% 
import numpy as np
from numpy.typing import ArrayLike
from numba import jit

from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, QuasiFloquetHamiltonianBase
from pymddrive.models.nonadiabatic_hamiltonian import diabatic_to_adiabatic, adiabatic_to_diabatic
from pymddrive.integrators.state import State, zeros_like
from pymddrive.integrators.rk4 import rk4
from pymddrive.dynamics.options import BasisRepresentation, QunatumRepresentation, NumericalIntegrators
from pymddrive.dynamics.misc_utils import eval_nonadiabatic_hamiltonian, HamiltonianRetureType
from pymddrive.dynamics.math_utils import commutator, rhs_density_matrix, v_dot_d, expected_value
from pymddrive.dynamics.floquet.ehrenfest import get_rho_and_populations
from pymddrive.dynamics.cache import Cache
from pymddrive.dynamics.langevin import LangevinBase

from numbers import Real
from typing import Union, Callable, Optional
from collections import namedtuple  


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

def _R_dot(v: ArrayLike) -> ArrayLike:
    return v

def _P_dot(meanF: ArrayLike) -> ArrayLike:
    return meanF

def _rho_dot(t, rho, R, v, hamiltonian, basis_rep) -> ArrayLike:
    hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
    if basis_rep == BasisRepresentation.Adiabatic:
        return _rho_dot_adiab(rho, hami_return.evals, v, hami_return.d)
    elif basis_rep == BasisRepresentation.Diabatic:
        return _rho_dot_diab(rho, hami_return.H)

def _rho_dot_adiab(
    rho: ArrayLike, evals: ArrayLike, v: ArrayLike, d: ArrayLike
) -> ArrayLike:
    vdotd = v_dot_d(v, d)
    # if not np.allclose(vdotd + vdotd.T.conj(), 0):
    #     print("The symmetry of vdotd is not preserved.")
    # else:
    #     print("The symmetry of vdotd is preserved.")
    # print(vdotd.dtype)
    return rhs_density_matrix(rho=rho, evals=evals, vdotd=vdotd)

def _rho_dot_diab(rho, H) -> ArrayLike:
    return -1.j * commutator(H, rho)

def _deriv_ehrenfest_dm(
    t: Real,
    s: State,
    cache: Cache,
    hamiltonian: HamiltonianBase,
    mass: Union[Real, np.ndarray],
    basis_rep: BasisRepresentation,
) -> State:
    R, P, rho = s.get_variables()
    out = zeros_like(s)
    kR, kP, k_rho = out.get_variables()
    
    # integrate the position
    v = P / mass
    kR[:] = _R_dot(v)
    
    # integrate the momentum
    meanF, hami_return = compute_ehrenfest_meanF(t, R, rho, hamiltonian, basis_rep)
    kP[:] = _P_dot(meanF + cache.F_langevin)
    
    
    # integrate the density matrix
    if basis_rep == BasisRepresentation.Adiabatic:
        k_rho[:] = _rho_dot_adiab(rho, hami_return.evals, v, hami_return.d)
    else:
        k_rho[:] = _rho_dot_diab(rho, hami_return.H)
    
    return out 

def step_rk(
    t: float,
    s: State,
    cache: Cache,
    dt: float,
    hamiltonian: HamiltonianBase,
    langevin: LangevinBase,
    mass: Union[float, np.ndarray],
    basis_rep: BasisRepresentation,
):
    # numerical Integration
    deriv_options = {
        "cache": cache,
        "hamiltonian": hamiltonian,
        "mass": mass,
        "basis_rep": basis_rep
    }
    t, s = rk4(t, s, _deriv_ehrenfest_dm, deriv_options, dt)
    # callback after one step
    R, P, _ = s.get_variables()
    F_Langevin = langevin.evaluate_langevin(t, R, P, dt)
    new_s, new_cache = callback(t, s, cache, F_Langevin, hamiltonian, basis_rep, dt, mass)
    return t, new_s, new_cache

def step_vv_rk(t, s, cache, dt, hamiltonian, mass, basis_rep):
    R, P, rho = s.get_variables()
    if cache is not None:
        meanF = cache.meanF
    else:
        meanF, hami_return = compute_ehrenfest_meanF(t, R, rho, hamiltonian, basis_rep)
        
    # First half of momentum integration
    P += 0.5 * dt * _P_dot(meanF)
        
    # Full position integration
    v = P / mass 
    R += dt * _R_dot(v)
    
    # for _ in range(n_qm_steps): 
    rk4_options = {'R': R, 'v': v, "hamiltonian": hamiltonian, "basis_rep": basis_rep} 
    _, rho[:] = rk4(t, rho, _rho_dot, rk4_options, dt=dt)
    
    meanF, hami_return = compute_ehrenfest_meanF(t+dt, R, rho, hamiltonian, basis_rep)
    
    P += 0.5 * dt * _P_dot(meanF)
        
    t += dt     
    cache = callback(t, s, hamiltonian, basis_rep)
    return t, s, cache

def step_vv_rk_gpaw(t, s, cache, dt, hamiltonian, mass, basis_rep):
    R, P, rho = s.get_variables()
    if cache is not None:
        meanF = cache.meanF
    else:
        meanF, hami_return = compute_ehrenfest_meanF(t, R, rho, hamiltonian, basis_rep)
    
    # The First half nuclear step
    R += 0.5 * dt * P / mass + 0.25 * dt**2 * meanF / mass
    P += 0.25 * dt * meanF
    meanF, hami_return = compute_ehrenfest_meanF(t+0.5*dt, R, rho, hamiltonian, basis_rep)
    P += 0.25 * dt * meanF 
 
    # The full electronic step
    v = P / mass
    # for _ in range(n_qm_steps): 
    rk4_options = {'R': R, 'v': v, "hamiltonian": hamiltonian, "basis_rep": basis_rep} 
    _, rho[:] = rk4(t, rho, _rho_dot, rk4_options, dt=dt)
   
    meanF = _compute_ehrenfest_meanF(rho, hami_return.dHdR, hami_return.evals, hami_return.evecs, hami_return.d, hami_return.F, basis_rep)
    
    # the second half nuclear step
    R += 0.5 * dt * P / mass + 0.25 * dt**2 * meanF / mass
    P += 0.25 * dt * meanF
    meanF, hami_return = compute_ehrenfest_meanF(t+dt, R, rho, hamiltonian, basis_rep)
    P += 0.25 * dt * meanF
    
    # update the time
    t += dt
    
    cache = callback(t, s, hamiltonian, basis_rep)
    
    return t, s, cache

def calculate_properties(t: Real, s: State, cache: Cache, hamiltonian: HamiltonianBase, mass: Union[Real, np.ndarray], basis_rep: BasisRepresentation):
    _, P, rho = s.get_variables()
    # meanF, hami_return = compute_ehrenfest_meanF(t, R, rho, hamiltonian, basis_rep)
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
    t: Real, R: ArrayLike, qm: ArrayLike, 
    hamiltonian: HamiltonianBase, basis_rep: BasisRepresentation
) -> ArrayLike:
    hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
    meanF = _compute_ehrenfest_meanF(qm, hami_return.dHdR, hami_return.evals, hami_return.evecs, hami_return.d, hami_return.F, basis_rep)
    # if isinstance(hamiltonian, QuasiFloquetHamiltonianBase):
    #     NF: int = hamiltonian.NF
    #     meanF = meanF / (2.0 * NF + 1.0)         
    return meanF, hami_return
    
def _compute_ehrenfest_meanF(qm, dHdR, evals, evecs, d, F, basis_rep: BasisRepresentation) -> ArrayLike:
    if basis_rep == BasisRepresentation.Adiabatic:
        # qm_diab = adiabatic_to_diabatic(qm, evecs)
        # return -1 * expected_value(qm_diab, dHdR)
        return _eval_Ehrenfest_meanF(evals, d, qm, F)
    elif basis_rep == BasisRepresentation.Diabatic:
        return -1 * expected_value(qm, dHdR)
    else:
        raise NotImplemented


@jit(nopython=True)
def _second_term_meanF(rho: ArrayLike, evals: ArrayLike, d: ArrayLike, meanF: Union[Real, ArrayLike]) -> ArrayLike:
    for i in range(evals.shape[0]):
        for j in range(i+1, evals.shape[0]):
            meanF += 2.0 * (evals[j] - evals[i]) * np.real(rho[i, j] * d[:, j, i])
    return meanF

def second_term_meanF(rho: ArrayLike, evals: ArrayLike, d: ArrayLike, meanF: Union[Real, ArrayLike, None]=None) -> ArrayLike:
    if meanF is None:
        meanF = np.zeros(d.shape[0], dtype=np.float64)
        
    return _second_term_meanF(rho, evals, d, meanF)

def _eval_Ehrenfest_meanF(evals, d, rho, F):
    """ Nonadiabatic Dynamics: Mean-Field and Surface Hopping. Nikos L. Doltsinis. """
    """ https://juser.fz-juelich.de/record/152530/files/FZJ-2014-02134.pdf """
    meanF_term1 = expected_value(rho, F, is_diagonal=True) # population-weighted average force
    meanF_term2 = second_term_meanF(rho, evals, d)         # nonadiabatic changes of the adiabatic state occupations
    return meanF_term1 + meanF_term2

def initialize_cache(
    t: float, 
    s: State, 
    F_Langevin: ArrayLike,
    hamiltonian: HamiltonianBase, 
    basis_rep: BasisRepresentation, 
) -> Cache:
    R, _, rho = s.get_variables()
    meanF, hami_return = compute_ehrenfest_meanF(t, R, rho, hamiltonian, basis_rep)
    return Cache(hamiltonian=hamiltonian, hami_return=hami_return, meanF=meanF, F_langevin=F_Langevin)

def callback(
    t: float, 
    s: State, 
    cache: Cache,
    F_langevin: Optional[ArrayLike],
    hamiltonian: HamiltonianBase, 
    basis_rep: BasisRepresentation, 
    dt: float,
    mass: Union[float, ArrayLike],
) -> Cache:
    R, _, rho = s.get_variables()
    hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
    meanF = _compute_ehrenfest_meanF(rho, hami_return.dHdR, hami_return.evals, hami_return.evecs, hami_return.d, hami_return.F, basis_rep)
    hamiltonian.update_last_evecs(hami_return.evecs)
    # hamiltonian.update_last_deriv_couplings(hami_return.d)
    F_langevin = F_langevin if F_langevin is not None else np.zeros_like(meanF)
    return s, Cache(hamiltonian=hamiltonian, hami_return=hami_return, meanF=meanF, F_langevin=F_langevin)
    
# %% the testing/debugging code

def _run_dynamics(algorithm: str, basis_rep: BasisRepresentation):
    from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes
    
    # get a convenient model for testing 
    model = get_tullyone(pulse_type=TullyOnePulseTypes.NO_PULSE)
    mass = 2000.0 
    
    t0 = 0.0
    R0 = -5.0; P0 = 30.0; rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=R0, P=P0, rho=rho0)
    
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
         
    for ns in range(int(1e6)):
        if ns % 100 == 0:
            properties = calculate_properties(t, s, model, mass, basis_rep)
            time_out = np.append(time_out, t)
            traj_out = np.array([s.data]) if traj_out is None else np.append(traj_out, s.data)
            properties_dict = _append_properties(properties_dict, properties)
            
            if stop_condition(t, s, traj_out):
                break
        t, s, cache = stepper(t, s, cache, dt, model, mass, basis_rep)
    properties_dict = {field: np.array(value) for (field, value) in properties_dict.items()}
    return time_out, traj_out, properties_dict['KE'], properties_dict['PE'], properties_dict['populations'] 

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
    t_out, s_out, KE_out_vv, PE_out_vv, pop_out_vv= _run_dynamics("vv", basis_rep=BasisRepresentation.Adiabatic)
    fig = _plot_debug(t_out, s_out, KE_out_vv, PE_out_vv, pop_out_vv, "VV", fig)
    print(f"The simulation time for dynamics with simple vv-rk4 is {time.perf_counter()-start}")
    start = time.perf_counter()
    t_out, s_out, KE_out_vv_GPAW, PE_out_vv_GPAW, pop_out_vv_GPAW = _run_dynamics("vv_gpaw", basis_rep=BasisRepresentation.Adiabatic)
    fig = _plot_debug(t_out, s_out, KE_out_vv_GPAW, PE_out_vv_GPAW, pop_out_vv_GPAW, "VV GPAW", fig)
    print(f"The simulation time for dynamics with vv-rk4-GPAW is {time.perf_counter()-start}")
    fig.tight_layout()

    

# %% the __main__ code for testing the package
if __name__ == "__main__":
    _test_debug()
       
# %%