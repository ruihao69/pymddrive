# %% The package code
import warnings

import numpy as np
from numba import jit

from typing import (
    Union, 
    Tuple
)
from numpy.typing import ArrayLike

from pymddrive.models.scatter import NonadiabaticHamiltonian
from pymddrive.integrators.state import (
    State, 
    zeros_like
)
from pymddrive.integrators.rk4 import rk4
from pymddrive.integrators.rungekutta import evaluate_initial_dt

def _process_mass(
    s0: State,
    mass: Union[float, None, ArrayLike],
) -> Union[float, ArrayLike]:
    if mass is None:
        return 1.0
    elif isinstance(mass, (float, int)):
        if mass < 0:
            raise ValueError("The mass should be positive.")
        return mass
    elif isinstance(mass, np.ndarray, list, tuple):
        r, _, _ = s0.get_variables()
        if len(mass) != len(r):
            raise ValueError("The length of mass should be the same as the length of r.")
        if all(m < 0 for m in mass):
            raise ValueError("The mass should be positive.")
        if all(m == mass[0] for m in mass):
            return mass[0]
        else:
            return mass
    raise ValueError("The mass should be a float, int, list, tuple or numpy.ndarray.") 

class Dynamics:
    def __init__(
        self,
        model: Union[NonadiabaticHamiltonian, None] = None,
        t0: float = 0.0,
        s0: Union[State, None] = None,
        mass: Union[float, None, ArrayLike] = None,
        dt: Union[float, None] = None,
        safty: float = 0.9,
    ) -> None:
        self.model = model
        self.t0 = t0
        self.s0 = s0
        self.mass = _process_mass(s0, mass)
        self.dt = dt
        self.safty = safty
        
    def deriv(
        self,
        t: float,
        state: State,
    ) -> State:
        pass
    
    def step(self, t: float, s: State, dt: float) -> State:
        pass
    
    def run(self, t: float, dt: float) -> State:
        pass
    
class NonadiabaticDynamics(Dynamics):
    def __init__(
        self,
        model: Union[NonadiabaticHamiltonian, None] = None,
        t0: float = 0.0,
        s0: Union[State, None] = None,
        mass: Union[float, None, ArrayLike] = None,
        solver: Union[str, None] = 'Ehrenfest',
        dt: Union[float, None] = None,
        save_every: int = 10,
        save_func: Union[None, callable] = None,
        safty: float = 0.9,
    ) -> None:
        super().__init__(model, t0, s0, mass, dt, safty)
        self.safty = safty
        self.solver = solver
        _dt = evaluate_initial_dt(self.deriv, t0, s0, order=4, atol=1e-8, rtol=1e-6,)
        _prop_dt = _dt * self.safty
        if dt is None:
            self.dt = _prop_dt
            print("The initial step size is set to ", _prop_dt)
        elif dt > _prop_dt:
            warnings.warn(f"The initial step size is set to {dt}, which is larger than the recommended step size {_prop_dt}. Please consider using a smaller step size.")
            self.dt = _prop_dt
        else:
            print("The initial step size is set to ", dt, ", which is smaller than the recommended step size ", _prop_dt)
            self.dt = _prop_dt
            
        self.save_every = save_every
        self.save_func = save_func
        self.nsteps = 0
        
    def deriv(
        self,
        t: float,
        state: State,
    ) -> State: 
        if self.solver == 'Ehrenfest':
            return _deriv_ehrenfest(self.model, t, state, mass=self.mass)
        else:
            raise ValueError("The solver should be either 'Ehrenfest', ...")
        
    def step(self, t: float, s: State, dt: float) -> Tuple[float, State]:
        return rk4(t, s, self.deriv, dt)

def _deriv_ehrenfest(
    model: NonadiabaticHamiltonian, 
    t: float,
    state: State,
    mass: Union[float, ArrayLike]=1.0,
) -> State:
    # get the states variables and output variables
    out = zeros_like(state)
    r, p, rho = state.get_variables()
    kr, kp, k_rho =  out.get_variables()
    
    # evaluate the nonadiabatic Hamiltonian
    if r.shape[0] == 1:
        _, evals, _, d, F = model(r=r[0], t=t)
        d = d[np.newaxis, :, :]
        F = F[:, np.newaxis]
    else:
        _, evals, _, d, F = model(r=r, t=t)
    
    # nonadiabatic dynamics: position
    kr[:] = p / mass
    
    # nonadiabatic dynamics: momentum
    meanF = _expected_value(rho=rho, O=F, is_O_diagonal_operator=True)
    kp[:] = meanF
    
    # nonadiabatic dynamics: density matrix
    vdotd = _v_dot_d(P=p, d=d, mass=mass)
    _rhs_density_matrix(rho=rho, evals=evals, vdotd=vdotd, k_rho=k_rho)
    
    return out

def _expected_value(
    rho: ArrayLike,
    O: ArrayLike,
    is_O_diagonal_operator: bool = False,
) -> float:
    if is_O_diagonal_operator:
        return rho.diagonal().dot(O).real
    else:
        return np.trace(np.dot(rho, O)).real
   
def _v_dot_d(
    P: ArrayLike,
    d: ArrayLike,
    mass: Union[float, ArrayLike]=1.0,
) -> ArrayLike:
    return np.tensordot(P/mass, d, axes=(0, 0)) 

@jit(nopython=True)
def _rhs_density_matrix(
    rho: ArrayLike,
    evals: ArrayLike,
    vdotd: ArrayLike,
    k_rho: ArrayLike,
) -> None:
    for kk in range(rho.shape[0]):
        for jj in range(rho.shape[0]):
            # k_rho[kk, jj] += -1.0j * (rho[kk, jj] * evals[kk] - rho[kk, jj] * evals[jj])
            k_rho[kk, jj] += -1.0j * rho[kk, jj] * (evals[kk] - evals[jj])
            for ll in range(rho.shape[0]):
                k_rho[kk, jj] += (-rho[ll, jj] * vdotd[kk, ll] + rho[kk, ll] * vdotd[ll, jj])
                
                
def _output_ehrenfest(
    t: float, 
    s: State, 
    model: NonadiabaticHamiltonian, 
    mass: Union[float, ArrayLike]
) -> None:
    r, p, rho = s.get_variables()
    _, evals, _, _, _ = model(r=r, t=t)
    PE = _expected_value(rho=rho, O=evals, is_O_diagonal_operator=True)
    KE = np.sum(p**2 / (2 * mass))
    return {'PE': PE, 'KE': KE, 'TE': PE + KE}
                
# %% The temporary test code
if __name__ == "__main__":
    from pymddrive.models.tully import TullyOne
    model = TullyOne()
    t0 = 0.0
    r0 = -10.0
    p0 = 30.0
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    
    s0 = State(r0, p0, rho0)
    
    mass = 2000.0
    dyn = NonadiabaticDynamics(model, t0, s0, mass, solver='Ehrenfest', dt=0.1)
    dyn.dt = 0.01
    
    t = t0
    s = s0
    out_t = np.array([t0])
    out_states = np.copy(s0.data)
    out_KE = np.array([0.5 * p0**2 / mass])
    out_PE = np.array([-model.A])
    out_TE = np.array([out_KE[0] + out_PE[0]])
    while (s.data['R'] < 10) or (s.data['R'] < -10):
        t, s = dyn.step(t, s, dyn.dt)
        dyn.nsteps += 1
        if dyn.nsteps % dyn.save_every == 0:
            out_t = np.append(out_t, t)
            out_states = np.append(out_states, s.data)
            out_props = _output_ehrenfest(t, s, dyn.model, dyn.mass)
            out_KE = np.append(out_KE, out_props['KE'])
            out_PE = np.append(out_PE, out_props['PE'])
            out_TE = np.append(out_TE, out_props['TE']) 
            
            # print(s.data['R']) 
        # out_t = np.append(out_t, t)
        # out_states = np.append(out_states, s.data)
        # print(s.data['R'])
    
# %%
r = out_states['R']
p = out_states['P']
rho = out_states['rho']
KE = p**2 / (2 * mass)
import matplotlib.pyplot as plt
fig, axs = plt.subplots(4, 1, figsize=(4, 8), dpi=300)
axs[0].plot(out_t, r)
axs[0].set_ylabel('r')
axs[1].plot(out_t, p)
axs[1].set_ylabel('p')
axs[2].plot(out_t, rho[:, 0, 0].real, label='rho11')
axs[2].plot(out_t, rho[:, 1, 1].real, label='rho22')
axs[2].legend()
# axs[3].plot(out_t, KE, label='KE')
# axs[3].plot(out_t, out_PE, label='PE')
axs[3].plot(out_t, out_TE, label='TE')
axs[3].axhline(y=out_TE[0], color='k', linestyle='--')
axs[3].set_ylabel('E')

axs[3].legend()
for ax in axs:
    ax.set_xlabel('t')

# %%
