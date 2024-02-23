# %% 
import numpy as np
from typing import Union, Tuple
from dataclasses import dataclass

from numpy.typing import ArrayLike
from pymddrive.models.scatter import NonadiabaticHamiltonian
from pymddrive.integrators.state import State, zeros_like

from pymddrive.integrators.rk4 import rk4

from _misc import (
    eval_nonadiabatic_hamiltonian,
    rhs_density_matrix,
    rhs_wavefunction,
    v_dot_d,
    expected_value
)

@dataclass(frozen=True)
class EhrenfestCache:
    meanF: float
    evals: ArrayLike
    

def _ehrenfest_meanF(t: float, R: ArrayLike, qm: ArrayLike, model: NonadiabaticHamiltonian) -> ArrayLike:
    evals, evecs, d, F = eval_nonadiabatic_hamiltonian(t, R, model)
    return expected_value(qm, F), evals, evecs, d

def _R_dot(P: ArrayLike, mass: Union[float, np.ndarray]) -> np.ndarray:
    return P / mass

def _P_dot(F: ArrayLike) -> np.ndarray:
    return F

def _rho_dot(t: float, rho: ArrayLike, r: ArrayLike, v: ArrayLike, model: NonadiabaticHamiltonian) -> ArrayLike:
    evals, _, d, _ = eval_nonadiabatic_hamiltonian(t, r, model)
    vdotd = v_dot_d(v, d) # derivative coupling dot velocity
    return rhs_density_matrix(rho=rho, evals=evals, vdotd=vdotd)

def _psi_dot(t: float, psi: ArrayLike, r: ArrayLike, v: ArrayLike, model: NonadiabaticHamiltonian) -> ArrayLike:
    evals, _, d, _ = eval_nonadiabatic_hamiltonian(t, r, model)
    vdotd = v_dot_d(v, d) # derivative coupling dot velocity
    return rhs_wavefunction(c=psi, evals=evals, vdotd=vdotd)

def _deriv_ehrenfest_dm(
    t: float,
    s: State,
    model: NonadiabaticHamiltonian,
    mass: Union[float, np.ndarray]
) -> Tuple[float, State, EhrenfestCache]:
    r, p, rho = s.get_variables()
    out = zeros_like(s)
    kr, kp, k_rho = out.get_variables()
    
    # integrate the position
    kr[:] = _R_dot(P=p, mass=mass)
    
    # integrate the momentum
    # meanF = _ehrenfest_meanF(t, r, rho, model)
    meanF, _, _, _ = _ehrenfest_meanF(t, r, rho, model)
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
    return t, s, EhrenfestCache(None, None)

def step_vv_rk(t, s, cache, dt, model, mass):
    r, p, rho = s.get_variables()
    if cache is not None:
        meanF = cache.meanF
    else:
        meanF, _, _, _ = _ehrenfest_meanF(t, r, rho, model)
        
    p[:] += 0.5 * dt * meanF
        
    v = p / mass 
    r[:] += dt * v
        
    _, rho[:] = rk4(t, rho, _rho_dot, {"r": r, "v": v, "model": model}, dt=dt)
    meanF, evals, _, _ = _ehrenfest_meanF(t, r, rho, model)
        
    p[:] += 0.5 * dt * meanF 
        
    t += dt     
    return t, s, EhrenfestCache(meanF, evals)

def calculate_properties(t: float, s: State, model: NonadiabaticHamiltonian, mass: Union[float, np.ndarray]):
    meanF, evals, _, _ = _ehrenfest_meanF(t, s.data['R'], s.data['rho'], model)
    KE = 0.5 * np.sum(s.data['P']**2 / mass)
    PE = expected_value(s.data['rho'], evals)
    return {
        "meanF": meanF,
        "KE": KE,
        "PE": PE
    }


def run_rk4():
    def calculate_properties_state(t: float, s: State, model: NonadiabaticHamiltonian, mass: Union[float, np.ndarray]):
        meanF, evals, _, _ = _ehrenfest_meanF(t, s.data['R'], s.data['rho'], model)
        KE = 0.5 * np.sum(s.data['P']**2 / mass)
        PE = expected_value(s.data['rho'], evals)
        return meanF, KE, PE
        
    
    mass = 2000.0 
    model = TullyOne()
    
    t0 = 0.0
    s0 = State(
        r=-10.0,
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
    while t < 1000.0:
        if ns % 10 == 0:
            meanF, KE, PE = calculate_properties_state(t, s, model, mass)
            t_out.append(t)
            s_out.append(s.data.copy())
            meanF_out.append(meanF)
            KE_out.append(KE)
            PE_out.append(PE)
        t, s, c = step_rk(t, s, dt, model, mass)
        # t, s = rk4(
        #     t, s,
        #     _deriv_ehrenfest_dm, 
        #     deriv_options=deriv_options,
        #     dt=dt
        # )
        ns += 1
    s_out = np.array(s_out) 
    return t_out, s_out, meanF_out, KE_out, PE_out

def run_vv():
    def calculate_properties_state(t: float, s: State, model: NonadiabaticHamiltonian, mass: Union[float, np.ndarray]):
        meanF, evals, _, _ = _ehrenfest_meanF(t, s.data['R'], s.data['rho'], model)
        KE = 0.5 * np.sum(s.data['P']**2 / mass)
        PE = expected_value(s.data['rho'], evals)
        return meanF, KE, PE
        
    
    mass = 2000.0 
    model = TullyOne()
    
    t0 = 0.0
    s0 = State(
        r=-10.0,
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
    
    
    meanF, evals, _, _ = _ehrenfest_meanF(t0, r0, rho0, model)
    c = EhrenfestCache(meanF, evals)
    
    while t < 1000.0:
        if ns % 10 == 0:
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
            meanF, _, _, _ = _ehrenfest_meanF(t, r, rho, model) 
        t, s, c = step_vv_rk(
            t, s, dt, model, mass, meanF
        )
        ns += 1
        
    s_out = np.array(s_out) 
    return t_out, s_out, meanF_out, KE_out, PE_out
    
    

# %%
if __name__ == "__main__":
    from scipy.integrate import ode, complex_ode
    from pymddrive.models.tully import TullyOne
    from pymddrive.integrators.rk4 import rk4
    t_out, s_out, meanF_out, KE_out_rk, PE_out_rk = run_rk4()
    t_out, s_out, meanF_out, KE_out_vv, PE_out_vv = run_vv()
   

# %%
# from pymddrive.models.tully import TullyOne
# import matplotlib.pyplot as plt
# from pymddrive.integrators.rk4 import rk4
# 
# plt.plot(t_out, KE_out_rk, label="KE RK")
# plt.plot(t_out, KE_out_vv, label="KE VV")
# plt.legend()
# plt.show()
# 
# plt.plot(t_out, PE_out_rk, label="PE RK")
# plt.plot(t_out, PE_out_vv, label="PE VV")
# plt.legend()
# plt.show()
# 
# plt.plot(t_out, [pe + ke for (pe, ke) in zip(PE_out_rk, KE_out_rk)], label="E RK")
# plt.plot(t_out, [pe + ke for (pe, ke) in zip(PE_out_vv, KE_out_vv)], label="E VV")
# plt.legend()
# plt.show()
# 
# plt.plot(t_out, s_out['rho'][:, 0, 0].real, label="rho00")
# plt.plot(t_out, s_out['rho'][:, 1, 1].real, label="rho11")
# plt.legend()
# plt.show()
# %%
