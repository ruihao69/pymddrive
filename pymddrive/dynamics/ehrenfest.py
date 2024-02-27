# %% 
import numpy as np
from numba import jit
from dataclasses import dataclass

from numbers import Real
from typing import Union, Tuple
from numpy.typing import ArrayLike

from pymddrive.models.scatter import NonadiabaticHamiltonian
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
    return meanF_term1 + meanF_term2
    

def compute_ehrenfest_meanF(t: Real, R: ArrayLike, qm: ArrayLike, model: NonadiabaticHamiltonian) -> ArrayLike:
    evals, evecs, d, F = eval_nonadiabatic_hamiltonian(t, R, model)
    return _eval_Ehrenfest_meanF(evals, d, qm, F), evals, evecs, d, F

def _R_dot(P: ArrayLike, mass: Union[Real, np.ndarray]) -> np.ndarray:
    return P / mass

def _P_dot(F: ArrayLike) -> np.ndarray:
    return F

def _rho_dot(t: Real, rho: ArrayLike, r: ArrayLike, v: ArrayLike, model: NonadiabaticHamiltonian) -> ArrayLike:
    evals, _, d, _ = eval_nonadiabatic_hamiltonian(t, r, model)
    vdotd = v_dot_d(v, d) # derivative coupling dot velocity
    # print(f"vdotd: {vdotd.shape}")
    return rhs_density_matrix(rho=rho, evals=evals, vdotd=vdotd)

def _psi_dot(t: Real, psi: ArrayLike, r: ArrayLike, v: ArrayLike, model: NonadiabaticHamiltonian) -> ArrayLike:
    evals, _, d, _ = eval_nonadiabatic_hamiltonian(t, r, model)
    vdotd = v_dot_d(v, d) # derivative coupling dot velocity
    return rhs_wavefunction(c=psi, evals=evals, vdotd=vdotd)

def _deriv_ehrenfest_dm(
    t: Real,
    s: State,
    model: NonadiabaticHamiltonian,
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

def calculate_properties(t: Real, s: State, model: NonadiabaticHamiltonian, mass: Union[Real, np.ndarray]):
    R, P, rho = s.get_variables()
    meanF, evals, _, d, F = compute_ehrenfest_meanF(t, R, rho, model)
    KE = 0.5 * np.sum(s.data['P']**2 / mass)
    PE = expected_value(s.data['rho'], evals)
    return {
        "meanF": meanF,
        "KE": KE,
        "PE": PE
    }

def calculate_properties_state(t: Real, s: State, model: NonadiabaticHamiltonian, mass: Union[Real, np.ndarray]):
    R, P, rho = s.get_variables()
    meanF, evals, _, d, F = compute_ehrenfest_meanF(t, R, rho, model) 
    KE = 0.5 * np.sum(s.data['P']**2 / mass)
    PE = expected_value(s.data['rho'], evals)
    return meanF, KE, PE


def run_rk4():
    mass = 2000.0 
    model = TullyOne()
    
    t0 = 0.0
    s0 = State(
        r=-5.0,
        p=30.0,
        rho=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    )
    r0, p0, rho0 = s0.get_variables()

    t, s = t0, s0
    dt = 0.03
    deriv_options = {
        "model": model,
        "mass": mass
    } 
    t_out = []
    s_out = []
    meanF_out = []
    KE_out = [] 
    PE_out = []
    ns = 0
    c = None
    while t < 700.0:
        if ns % 100 == 0:
            meanF, KE, PE = calculate_properties_state(t, s, model, mass)
            t_out.append(t)
            s_out.append(s.data.copy())
            meanF_out.append(meanF)
            KE_out.append(KE)
            PE_out.append(PE)
        t, s, c = step_rk(t, s, c, dt, model, mass)
        ns += 1
    s_out = np.array(s_out) 
    return t_out, s_out, meanF_out, KE_out, PE_out

def run_vv(gpaw=False):
    mass = 2000.0 
    model = TullyOne()
    
    t0 = 0.0
    s0 = State(
        r=-5.0,
        p=30.0,
        rho=np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    )
    r0, p0, rho0 = s0.get_variables()

    t, s = t0, s0
    dt = 0.03
    deriv_options = {
        "model": model,
        "mass": mass
    } 
    t_out = []
    s_out = []
    meanF_out = []
    KE_out = [] 
    PE_out = []
    ns = 0
    
    meanF, evals, _, d, F = compute_ehrenfest_meanF(t, r0, rho0, model)  
    c = EhrenfestCache(meanF, evals)
    
    stepper = step_vv_rk_gpaw if gpaw else step_vv_rk
    
    while t < 700.0:
        if ns % 100 == 0:
            meanF, KE, PE = calculate_properties_state(t, s, model, mass)
            t_out.append(t)
            s_out.append(s.data.copy())
            meanF_out.append(meanF)
            KE_out.append(KE)
            PE_out.append(PE)
        if c.meanF is not None:
            meanF = c.meanF 
        else:
            r, _, rho = s.get_variables()
            meanF, evals, _, d, F = compute_ehrenfest_meanF(t, r, rho, model)
        # t, s, c = step_vv_rk(
        t, s, c = stepper(
            t, s, c, dt, model, mass, 
        )
        ns += 1
        
    s_out = np.array(s_out) 
    return t_out, s_out, meanF_out, KE_out, PE_out
    
    

# %%
if __name__ == "__main__":
    from scipy.integrate import ode, complex_ode
    from pymddrive.models.tully import TullyOne
    import time
    start = time.perf_counter()
    t_out, s_out, meanF_out, KE_out_rk, PE_out_rk = run_rk4()
    print(f"The simulation time for dynamics with rk4 is {time.perf_counter()-start}")
    start = time.perf_counter()
    t_out, s_out, meanF_out, KE_out_vv, PE_out_vv = run_vv(gpaw=False)
    print(f"The simulation time for dynamics with simple vv-rk4 is {time.perf_counter()-start}")
    start = time.perf_counter()
    t_out, s_out, meanF_out, KE_out_vv_GPAW, PE_out_vv_GPAW = run_vv(gpaw=True)
    print(f"The simulation time for dynamics with vv-rk4-GPAW is {time.perf_counter()-start}")
   

# %%
if __name__ == "__main__":
    from pymddrive.models.tully import TullyOne
    import matplotlib.pyplot as plt
    from pymddrive.integrators.rk4 import rk4

    plt.plot(t_out, KE_out_rk, label="KE RK")
    plt.plot(t_out, KE_out_vv, label="KE VV")
    plt.plot(t_out, KE_out_vv_GPAW, label="KE VV GPAW")
    plt.legend()
    plt.show()

    plt.plot(t_out, PE_out_rk, label="PE RK")
    plt.plot(t_out, PE_out_vv, label="PE VV")
    plt.plot(t_out, PE_out_vv_GPAW, label="PE VV GPAW")
    plt.legend()
    plt.show()

    plt.plot(t_out, [pe + ke for (pe, ke) in zip(PE_out_rk, KE_out_rk)], label="E RK")
    plt.plot(t_out, [pe + ke for (pe, ke) in zip(PE_out_vv, KE_out_vv)], label="E VV")
    plt.plot(t_out, [pe + ke for (pe, ke) in zip(PE_out_vv_GPAW, KE_out_vv_GPAW)], label="E VV GPAW")
    plt.legend()
    plt.show()

    plt.plot(t_out, s_out['rho'][:, 0, 0].real, label="rho00")
    plt.plot(t_out, s_out['rho'][:, 1, 1].real, label="rho11")
    # plt.plot(t_out, s_out['rho'][:, 0, 0].real + s_out['rho'][:, 1, 1].real, label="rho11")
    plt.legend()
    plt.show()
# %%
