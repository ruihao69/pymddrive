# %% The package code
import warnings

import numpy as np

from typing import (
    Union, 
    Tuple,
    Any
)
from numbers import Real
from numpy.typing import ArrayLike

# from pymddrive.models.scatter import NonadiabaticHamiltonianBase
from pymddrive.models.nonadiabatic_hamiltonian import NonadiabaticHamiltonianBase
from pymddrive.integrators.state import State
from pymddrive.integrators.rungekutta import evaluate_initial_dt

from pymddrive.dynamics import ehrenfest 

from functools import partial

SOLVERS = ('Ehrenfest',)
INTEGRATORS = ('rk4', 'vv_rk4', 'vv_rk4_gpaw')

def _process_mass(
    s0: State,
    mass: Union[Real, None, ArrayLike],
) -> Union[float, ArrayLike]:
    if mass is None:
        return 1.0
    elif isinstance(mass, Real):
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

def estimate_scatter_dt(deriv: callable, r_bounds: tuple, p0: float, model: NonadiabaticHamiltonianBase, mass: Real=2000, nsample: Real=30, t_bounds: Tuple[Real]=None) -> float:
    r_list = np.linspace(*r_bounds, nsample)
    if t_bounds is not None:
        t_list = np.random.uniform(*t_bounds, nsample)
    else:
        t_list = np.zeros(nsample)
    _dt = 99999999999
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    for i in range(nsample):
        s0 = State.from_variables(R=r_list[i], P=p0, rho=rho0)
        _dt = min(_dt, evaluate_initial_dt(deriv, t_list[i], s0, order=4, atol=1e-8, rtol=1e-6,))
    return _dt
 

class Dynamics:
    def __init__(
        self,
        model: Union[NonadiabaticHamiltonianBase, None] = None,
        t0: Real = 0.0,
        s0: Union[State, None] = None,
        mass: Union[Real, None, ArrayLike] = None,
        dt: Union[Real, None] = None,
        safty: float = 0.9,
    ) -> None:
        self.model = model
        self.t0 = t0
        self.s0 = s0
        self.mass = _process_mass(s0, mass)
        self.dt = dt
        self.safty = safty
        
    def step(self, t: Real, s: State, c: Any) -> Tuple[Real, State, Any]:
        pass
    
class NonadiabaticDynamics(Dynamics):
    def __init__(
        self,
        model: Union[NonadiabaticHamiltonianBase, None] = None,
        t0: Real= 0.0,
        s0: Union[State, None] = None,
        mass: Union[Real, None, ArrayLike] = None,
        solver: Union[str, None] = 'Ehrenfest',
        method: Union[str, None] = 'rk4',
        dt: Union[float, None] = None,
        r_bounds: Union[Tuple[Real], None] = None,
        t_bounds: Union[Tuple[Real], None] = None,
        save_every: int = 10,
        safty: float = 0.9,
    ) -> None:
        super().__init__(model, t0, s0, mass, dt, safty)
        self.safty = safty
        self.solver = solver
        if solver == 'Ehrenfest':
            self.deriv = partial(ehrenfest._deriv_ehrenfest_dm, model=model, mass=mass)
        if r_bounds is None:
            _dt = evaluate_initial_dt(self.deriv, t0, s0, order=4, atol=1e-8, rtol=1e-6,)
            _prop_dt = _dt * self.safty
        else:
            _dt = estimate_scatter_dt(self.deriv, r_bounds, s0.data['P'][0], model, mass, nsample=30, t_bounds=t_bounds)
            _prop_dt = _dt * self.safty
            
            
        self.dt = min(dt, _prop_dt) if dt is not None else _prop_dt
        print(f"The recommended step size is {_prop_dt}, the final decision is {self.dt}.", flush=True)
        
        self.save_every = save_every
        self.nsteps = 0
        
        if method not in INTEGRATORS:
            raise ValueError(f"The method {method} is not supported.")
        
        if solver not in SOLVERS:
            raise ValueError(f"The solver {solver} is not supported.")
        
        if solver == 'Ehrenfest': 
            if method == 'rk4':
                self.step = partial(ehrenfest.step_rk, dt=self.dt, model=model, mass=mass)
            elif method == 'vv_rk4':
                self.step = partial(ehrenfest.step_vv_rk, dt=self.dt, model=model, mass=mass)
            elif method == 'vv_rk4_gpaw':
                self.step = partial(ehrenfest.step_vv_rk_gpaw, dt=self.dt, model=model, mass=mass)
            else:
                raise ValueError(f"The method {method} is not supported for the {solver} solver.")
            self.calculate_properties = ehrenfest.calculate_properties
            self.deriv = ehrenfest._deriv_ehrenfest_dm
        else:
            raise ValueError(f"The solver {solver} is not supported.")

def run_nonadiabatic_dynamics(
    dyn: NonadiabaticDynamics,
    stop_condition: callable,
    break_condition: callable,
    max_iters: int=int(1e8),
):
    check_stop_every = dyn.save_every * 30
    check_break_every = dyn.save_every * 30
    output = {
        'time': [],
        'states': [],
        'KE': [],
        'PE': [],
    }
    
    t, s = dyn.t0, dyn.s0
    cache = None
    
    for istep in range(max_iters):
        if istep % dyn.save_every == 0:
            properties = dyn.calculate_properties(t, s, dyn.model, dyn.mass)
            output['time'] = np.append(output['time'], t)
            output['states'] = np.append(output['states'], s.data) if len(output['states']) > 0 else np.array([s.data])
            output['KE'] = np.append(output['KE'], properties['KE'])
            output['PE'] = np.append(output['PE'], properties['PE'])
            if istep % check_stop_every == 0:
                if stop_condition(t, s, output['states']):
                    break
            if istep % check_break_every == 0:
                if break_condition(t, s, output['states']):
                    warnings.warn("The break condition is met.")
                    break
        t, s, cache = dyn.step(t, s, cache)
    output['states'] = np.array(output['states'])
        
    return output

# %% The temporary test code
def _debug_test():
    import time
    from pymddrive.models.tullyone import TullyOnePulseTypes, TD_Methods, get_tullyone
    import matplotlib.pyplot as plt
    
    # get a convenient model for testing
    model = get_tullyone(
        pulse_type=TullyOnePulseTypes.NO_PULSE, 
    )
    mass = 2000.0
    
    # initial conditions
    t0 = 0.0; r0 = -10.0; p0 = 30.0
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    
    # prepare the dynamics object
    dyn = NonadiabaticDynamics( 
        model=model,
        t0=t0, s0=s0, mass=mass,
        solver='Ehrenfest', method='vv_rk4',
        r_bounds=(-10.0, 10.0)
    )
    
    def stop_condition(t, s, states):
        r, _, _ = s.get_variables()
        return (r>10.0) or (r<-10.0)
    
    def break_condition(t, s, states):
        r = np.array(states['R'])
        def count_re_crossings(r, r_TST=0.0):
            r_sign = np.sign(r - r_TST)
            r_sign_diff = np.diff(r_sign)
            n = np.sum(r_sign_diff != 0) - 1
            n_re_crossings = 0 if n < 0 else n
            return n_re_crossings
        return (count_re_crossings(r) > 10)
    
    start = time.perf_counter() 
    output = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    print(f"The time for the simulation is {time.perf_counter()-start} s.")
    
    t = output['time']
    r = output['states']['R']
    p = output['states']['P']
    rho = output['states']['rho']
    
    plt.plot(t, r)  
    plt.xlabel('Time')
    plt.ylabel('R')
    plt.title('Position') 
    plt.show()
    plt.plot(t, p)
    plt.xlabel('Time')
    plt.ylabel('P')
    plt.title('Momentum')
    plt.show()
    
    plt.plot(t, rho[:, 0, 0].real, label='rho00')
    plt.plot(t, rho[:, 1, 1].real, label='rho11')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()
     
    
# %% the __main__ code
if __name__ == "__main__":
    _debug_test()
    
# %%