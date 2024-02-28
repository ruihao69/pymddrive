# %% 
import numpy as np
from numba import jit
from dataclasses import dataclass

from numbers import Real
from typing import Union, Tuple
from numpy.typing import ArrayLike

from pymddrive.models.nonadiabatic_hamiltonian import NonadiabaticHamiltonianBase
from pymddrive.integrators.state import State, zeros_like

from pymddrive.integrators.rk4 import rk4

from pymddrive.dynamics._misc import (
    eval_nonadiabatic_hamiltonian,
    rhs_density_matrix,
    rhs_wavefunction,
    v_dot_d,
    expected_value
)

@dataclass(frozen=True)
class EhrenfestCache:
    meanF: Real
    evals: ArrayLike

@jit(nopython=True)
def _second_term_meanF(rho: ArrayLike, evals: ArrayLike, d: ArrayLike, meanF: Union[Real, ArrayLike]) -> ArrayLike:
    for i in range(evals.shape[0]):
        for j in range(i+1, evals.shape[0]):
            meanF += 2.0 * (evals[j] - evals[i]) * np.real(rho[j, i] * d[:, j, i])
    return meanF            

def second_term_meanF(rho: ArrayLike, evals: ArrayLike, d: ArrayLike, meanF: Union[Real, ArrayLike, None]=None) -> ArrayLike:
    if meanF is None:
        meanF = np.zeros(d.shape[0], dtype=np.float64)
        
    return _second_term_meanF(rho, evals, d, meanF)

def _eval_Ehrenfest_meanF(evals, d, rho, F):
    """ Nonadiabatic Dynamics: Mean-Field and Surface Hopping. Nikos L. Doltsinis. """
    """ https://juser.fz-juelich.de/record/152530/files/FZJ-2014-02134.pdf """
    meanF_term1 = expected_value(rho, F)           # population-weighted average force
    meanF_term2 = second_term_meanF(rho, evals, d) # nonadiabatic changes of the adiabatic state occupations
    # meanF_term2 = 0.0
    return meanF_term1 + meanF_term2
    

def compute_ehrenfest_meanF(t: Real, R: ArrayLike, qm: ArrayLike, model: NonadiabaticHamiltonianBase) -> ArrayLike:
    evals, evecs, d, F = eval_nonadiabatic_hamiltonian(t, R, model)
    return _eval_Ehrenfest_meanF(evals, d, qm, F), evals, evecs, d, F

def _R_dot(P: ArrayLike, mass: Union[Real, np.ndarray]) -> np.ndarray:
    return P / mass

def _P_dot(F: ArrayLike) -> np.ndarray:
    return F

def _rho_dot(t: Real, rho: ArrayLike, r: ArrayLike, v: ArrayLike, model: NonadiabaticHamiltonianBase) -> ArrayLike:
    evals, _, d, _ = eval_nonadiabatic_hamiltonian(t, r, model)
    vdotd = v_dot_d(v, d) # derivative coupling dot velocity
    # print(f"vdotd: {vdotd.shape}")
    return rhs_density_matrix(rho=rho, evals=evals, vdotd=vdotd)

def _psi_dot(t: Real, psi: ArrayLike, r: ArrayLike, v: ArrayLike, model: NonadiabaticHamiltonianBase) -> ArrayLike:
    evals, _, d, _ = eval_nonadiabatic_hamiltonian(t, r, model)
    vdotd = v_dot_d(v, d) # derivative coupling dot velocity
    return rhs_wavefunction(c=psi, evals=evals, vdotd=vdotd)

def _deriv_ehrenfest_dm(
    t: Real,
    s: State,
    model: NonadiabaticHamiltonianBase,
    mass: Union[Real, np.ndarray]
) -> Tuple[Real, State, EhrenfestCache]:
    r, p, rho = s.get_variables()
    out = zeros_like(s)
    kr, kp, k_rho = out.get_variables()
    
    # integrate the position
    kr[:] = _R_dot(P=p, mass=mass)
    
    # integrate the momentum
    # meanF = _ehrenfest_meanF(t, r, rho, model)
    meanF, _, _, _, _= compute_ehrenfest_meanF(t, r, rho, model)
    kp[:] = _P_dot(meanF)
    
    # integrate the density matrix
    k_rho[:] = _rho_dot(t, rho, r, p/mass, model)
    
    return out

def step_rk(t, s, cache, dt, model, mass):
    deriv_options = {
        "model": model,
        "mass": mass
    }
    t, s = rk4(t, s, _deriv_ehrenfest_dm, deriv_options, dt)
    # t, s = rkgill4(t, s, _deriv_ehrenfest_dm, deriv_options, dt)
    return t, s, EhrenfestCache(None, None)

def step_vv_rk(t, s, cache, dt, model, mass):
    r, p, rho = s.get_variables()
    if cache is not None:
        meanF = cache.meanF
    else:
        meanF, _, _, _, _= compute_ehrenfest_meanF(t, r, rho, model)
        
    p[:] += 0.5 * dt * meanF
        
    v = p / mass 
    r[:] += dt * v
        
    _, rho[:] = rk4(t, rho, _rho_dot, {"r": r, "v": v, "model": model}, dt=dt)
    # _, rho[:] = rkgill4(t, rho, _rho_dot, {"r": r, "v": v, "model": model}, dt=dt)
    meanF, evals, _, _, _ = compute_ehrenfest_meanF(t, r, rho, model)
        
    p[:] += 0.5 * dt * meanF 
        
    t += dt     
    return t, s, EhrenfestCache(meanF, evals)

def step_vv_rk_gpaw(t, s, cache, dt, model, mass):
    R, P, rho = s.get_variables()
    if cache is not None:
        meanF = cache.meanF 
    else:
        meanF, evals, _, _, _ = compute_ehrenfest_meanF(t, R, rho, model)
    
    # The First half nuclear step
    R += 0.5 * dt * P / mass + 0.25 * dt**2 * meanF / mass
    P += 0.25 * dt * meanF
    meanF, evals, _, d, F = compute_ehrenfest_meanF(t, R, rho, model)
    P += 0.25 * dt * meanF 
 
    # The full electronic step
    v = P / mass
    _, rho = rk4(t, rho, _rho_dot, {"r": R, "v": v, "model": model}, dt=dt)
    # _, rho = rkgill4(t, rho, _rho_dot, {"r": R, "v": v, "model": model}, dt=dt)
    meanF = _eval_Ehrenfest_meanF(evals, d, rho, F)
    
    # the second half nuclear step
    R += 0.5 * dt * P / mass + 0.25 * dt**2 * meanF / mass
    P += 0.25 * dt * meanF
    meanF, evals, _, d, F = compute_ehrenfest_meanF(t, R, rho, model) 
    P += 0.25 * dt * meanF
    
    # update the time
    t += dt
    
    return t, s, EhrenfestCache(meanF=meanF, evals=evals)

def calculate_properties(t: Real, s: State, model: NonadiabaticHamiltonianBase, mass: Union[Real, np.ndarray]):
    R, P, rho = s.get_variables()
    meanF, evals, _, d, F = compute_ehrenfest_meanF(t, R, rho, model)
    KE = 0.5 * np.sum(s.data['P']**2 / mass)
    PE = expected_value(s.data['rho'], evals)
    return {
        "meanF": meanF,
        "KE": KE,
        "PE": PE
    }

def calculate_properties_state(t: Real, s: State, model: NonadiabaticHamiltonianBase, mass: Union[Real, np.ndarray]):
    R, P, rho = s.get_variables()
    meanF, evals, _, d, F = compute_ehrenfest_meanF(t, R, rho, model) 
    KE = 0.5 * np.sum(s.data['P']**2 / mass)
    PE = expected_value(s.data['rho'], evals)
    return meanF, KE, PE
    
# %% the testing/debugging code

def _run_dynamics(algorithm):
    from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes
    
    # get a convenient model for testing 
    model = get_tullyone(pulse_type=TullyOnePulseTypes.NO_PULSE)
    mass = 2000.0 
    
    t0 = 0.0
    R0 = -5.0; P0 = 30.0; rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=R0, P=P0, rho=rho0)
    
    r0, p0, rho0 = s0.get_variables()
    output = {
        'time': [],
        'states': [],
        'KE': [],
        'PE': [],
    }
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
         
    for ns in range(int(1e6)):
        if ns % 100 == 0:
            properties = calculate_properties(t, s, model, mass)
            output['time'] = np.append(output['time'], t)
            output['states'] = np.append(output['states'], s.data) if len(output['states']) > 0 else np.array([s.data])
            output['KE'] = np.append(output['KE'], properties['KE'])
            output['PE'] = np.append(output['PE'], properties['PE'])
            if stop_condition(t, s, output['states']):
                break
        t, s, cache = stepper(t, s, cache, 0.03, model, mass)
    return output["time"], output["states"], output["KE"], output["PE"]

def _plot_debug(t_out, s_out, KE_out, PE_out, label: str, fig=None, ):
    import matplotlib.pyplot as plt
    if fig is None:
        fig, axs = plt.subplots(3, 2, figsize=(5*2, 3*3))
        print(axs.shape)
        axs_dynamics = axs[:, 0]
        axs_energy = axs[:, 1]
    else:
        axs = np.array(fig.get_axes()).reshape(3, 2)
        axs_dynamics = axs[:, 0]
        axs_energy = axs[:, 1]
        
    axs_dynamics[0].plot(t_out, s_out['R'], label=f"{label}: R")
    axs_dynamics[1].plot(t_out, s_out['P'], label=f"{label}: P")
    axs_dynamics[2].plot(t_out, s_out['rho'][:, 0, 0].real, label=f"{label}: rho00")
    axs_dynamics[2].plot(t_out, s_out['rho'][:, 1, 1].real, label=f"{label}: rho11")
    for ax in axs_dynamics:
        ax.legend()
        ax.set_xlabel("Time")
    
    axs_energy[0].plot(t_out, KE_out, label=f"{label}: KE")
    axs_energy[1].plot(t_out, PE_out, label=f"{label}: PE")
    axs_energy[2].plot(t_out, [ke + pe for (ke, pe) in zip(KE_out, PE_out)], label=f"{label} E")
    for ax in axs_energy:
        ax.legend()
        ax.set_xlabel("Time") 
        
    return fig

def _test_debug():
    import time
    start = time.perf_counter()
    t_out, s_out, KE_out_rk, PE_out_rk = _run_dynamics("rk4")
    fig = _plot_debug(t_out, s_out, KE_out_rk, PE_out_rk, "RK4")
    print(f"The simulation time for dynamics with rk4 is {time.perf_counter()-start}")
    start = time.perf_counter()
    t_out, s_out, KE_out_vv, PE_out_vv = _run_dynamics("vv")
    fig = _plot_debug(t_out, s_out, KE_out_vv, PE_out_vv, "VV", fig)
    print(f"The simulation time for dynamics with simple vv-rk4 is {time.perf_counter()-start}")
    start = time.perf_counter()
    t_out, s_out, KE_out_vv_GPAW, PE_out_vv_GPAW = _run_dynamics("vv_gpaw")
    fig = _plot_debug(t_out, s_out, KE_out_vv_GPAW, PE_out_vv_GPAW, "VV GPAW", fig)
    print(f"The simulation time for dynamics with vv-rk4-GPAW is {time.perf_counter()-start}")
    fig.tight_layout()

    

# %% the __main__ code for testing the package
if __name__ == "__main__":
    _test_debug()
       
# %%
