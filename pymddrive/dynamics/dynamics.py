# %% The package code
import warnings
import numpy as np
from scipy.integrate import ode

from typing import Union, Tuple, Any, Type, Callable, NamedTuple
    
from numbers import Real
from numpy.typing import ArrayLike

from pymddrive.models.nonadiabatic_hamiltonian import NonadiabaticHamiltonianBase
from pymddrive.integrators.state import State
from pymddrive.integrators.rungekutta import evaluate_initial_dt

from pymddrive.dynamics import ehrenfest 

from functools import partial
from abc import ABC

from pymddrive.dynamics.options import (
    BasisRepresentation, QunatumRepresentation, 
    NonadiabaticDynamicsMethods, NumericalIntegrators
)
    
class Dynamics(ABC):
    def __init__(
        self,
        hamiltonian: NonadiabaticHamiltonianBase,
        t0: float, s0: State, mass: Union[float, ArrayLike], 
        dt: Union[float, None]=None, 
        atol: float=1e-8, rtol: float=1e-8, safety: float=0.9, save_every: int=10,
        numerical_integrator: NumericalIntegrators=NumericalIntegrators.VVRK4
    ) -> None:
        self.hamiltonian = hamiltonian
        self.t0 = t0    
        self.s0 = s0    
        self.mass = self._process_mass(s0, mass)
        # self.dt = dt
        self.safty = safety
        self.save_every = save_every
        self.atol = atol
        self.rtol = rtol
        
        self.ode_solver = None
        self.dtype = s0.data.dtype
        self.stype = s0.stype
        
        # the stepper design: takes time, state, and cache, returns time, state, and cache
        self.step: Callable[[float, State, Any], Tuple[float, State, Any]] = None
        # the derivative function: takes time and state, returns the derivative of the state
        self.deriv: Callable[[float, State], State] = None
        # the properties calculator: takes the current time and state, returns the properties as a namedtuple
        self.calculate_properties: Callable[[float, State], NamedTuple] = None
        self.properties_type: Type = None 
        
        self.numerical_integrator: NumericalIntegrators = numerical_integrator
            
    @staticmethod 
    def _process_mass(
        s0: State, mass: Union[Real, None, ArrayLike],
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
    
class NonadiabaticDynamics(Dynamics):
    def __init__(
        self,
        hamiltonian: Union[NonadiabaticHamiltonianBase, None] = None,
        t0: Real= 0.0, s0: Union[State, None] = None, mass: Union[Real, None, ArrayLike] = None,
        dt: Union[float, None] = None, atol: float=1e-8, rtol: float=1e-6, safety: float=0.9, save_every: int = 10,
        qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
        basis_rep: BasisRepresentation = BasisRepresentation.Adiabatic,
        solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST, 
        numerical_integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
        r_bounds: Union[Tuple[Real], None] = None,
        t_bounds: Union[Tuple[Real], None] = None,
    ) -> None:
        super().__init__(
            hamiltonian, t0, s0, mass, dt, atol, rtol, safety, save_every, numerical_integrator
        ) 
        
        if qm_rep != QunatumRepresentation.DensityMatrix:
            raise NotImplementedError(f"At this time, NonadiabaticDynamics class only supports the Density Matrix representation for quantum system.")
        
        self.qm_rep = qm_rep
        self.basis_rep = basis_rep
       
        self.deriv = self.get_deriv(qm_rep, solver, basis_rep, numerical_integrator)
        self.calculate_properties, self.properties_type = self.get_properties_calculator(solver)
        
        self.dt = dt if dt is not None else 0.03
        if numerical_integrator != NumericalIntegrators.ZVODE:
            if r_bounds is not None:
                prop_dt_scatter = estimate_scatter_dt(self.deriv, r_bounds, s0, nsample=100, t_bounds=t_bounds)
                if self.dt > prop_dt_scatter:
                    warnings.warn(f"The intial dt {self.dt} is larger than the estimated scatter dt {prop_dt_scatter}. Changing to the scatter dt.")
                    self.dt = prop_dt_scatter
            print(f"Using the {numerical_integrator} solver, where {self.dt=} is used for the fixed dt.", flush=True)
        else:
            print(f"Using the zvode solver, where {self.dt=} is used for dense output.", flush=True)
                
        self.step = self.get_stepper(solver, numerical_integrator)
        
    def get_stepper(
        self, 
        method: NonadiabaticDynamicsMethods,
        numerical_integrator: NumericalIntegrators,
    ) -> Callable[[float, State, Any], Tuple[float, State, Any]]:
        if method == NonadiabaticDynamicsMethods.EHRENFEST:
            if numerical_integrator != NumericalIntegrators.ZVODE:
                raw_stepper = ehrenfest.choose_ehrenfest_stepper(numerical_integrator)
                return partial(raw_stepper, dt=self.dt, hamiltonian=self.hamiltonian, mass=self.mass, basis_rep=self.basis_rep)
            else:
                self.ode_solver = ode(self.deriv).set_integrator(
                    'zvode',
                    method='bdf',
                    atol=self.atol,
                    rtol=self.rtol,
                )
                self.ode_solver.set_initial_value(self.s0.flatten(), self.t0)
                return self._step_zvode 
        elif method == NonadiabaticDynamicsMethods.FSSH:
            raise NotImplemented("FSSH is not implemented at this time.")
        else:
            raise NotImplemented(f"Unrecogonized nonadiabatic {method=}. Not implemented in pymddrive.")
        
    def _step_zvode(self, t: float, s: State, cache: Any) -> Tuple[float, State, Any]:
        if not self.ode_solver.successful():
            raise RuntimeError("The ode solver is not successful.")
        if t != self.ode_solver.t:
            raise ValueError(f"The time {t} is not the same as the solver time {self.ode_solver.t}.")
        self.ode_solver.integrate(self.ode_solver.t + self.dt)
        return self.ode_solver.t, State.from_unstructured(self.ode_solver.y, dtype=self.dtype, stype=self.stype, copy=False), cache
    
    def get_deriv(
        self, 
        quatum_representation: QunatumRepresentation,
        method: NonadiabaticDynamicsMethods,
        basis_representation: BasisRepresentation,
        numerical_integrator: NumericalIntegrators,
    ) -> Callable[[float, State], State]:
        if method == NonadiabaticDynamicsMethods.EHRENFEST:
            raw_deriv = ehrenfest.choose_ehrenfest_deriv(quatum_representation)
            if numerical_integrator != NumericalIntegrators.ZVODE:
                return partial(raw_deriv, hamiltonian=self.hamiltonian, mass=self.mass, basis_rep=basis_representation)
            else:
                def deriv_wrapper(t, y: ArrayLike, copy: bool=False)->ArrayLike:
                    s = State.from_unstructured(y, dtype=self.dtype, stype=self.stype, copy=copy)
                    dsdt = raw_deriv(t, s, hamiltonian=self.hamiltonian, mass=self.mass, basis_rep=basis_representation)
                    return dsdt.flatten(copy=copy)
                return deriv_wrapper
                
        elif method == NonadiabaticDynamicsMethods.FSSH:
            raise NotImplemented("FSSH is not implemented at this time.")
        else:
            raise NotImplemented(f"Unrecogonized nonadiabatic {method=}. Not implemented in pymddrive.")
        
    def get_properties_calculator(
        self, method: NonadiabaticDynamicsMethods, 
    ) -> Tuple[Callable, Type]:
        if method == NonadiabaticDynamicsMethods.EHRENFEST:
            raw_calculator = ehrenfest.calculate_properties
            calculator = partial(raw_calculator, hamiltonian=self.hamiltonian, mass=self.mass, basis_rep=self.basis_rep)
            return calculator, ehrenfest.EhrenfestProperties
        elif method == NonadiabaticDynamicsMethods.FSSH:
            raise NotImplemented("FSSH is not implemented at this time.")
        else:
            raise NotImplemented(f"Unrecogonized nonadiabatic {method=}. Not implemented in pymddrive.")
        
        
def run_nonadiabatic_dynamics(
    dyn: NonadiabaticDynamics,
    stop_condition: callable,
    break_condition: callable,
    max_iters: int=int(1e8),
    save_traj: bool=True,
):
    check_stop_every = dyn.save_every * 30
    check_break_every = dyn.save_every * 30
    time_array = np.array([])
    traj_array = None
    properties_dict = {field: [] for field in dyn.properties_type._fields}
    
    t, s = dyn.t0, dyn.s0
    cache = None
    
    for istep in range(max_iters):
        if istep % dyn.save_every == 0:
            properties = dyn.calculate_properties(t, s)
            properties_dict = _append_properties(properties_dict, properties)
            time_array = np.append(time_array, t)
            traj_array = np.array([s.data]) if traj_array is None else np.append(traj_array, s.data)
            if istep % check_stop_every == 0:
                if stop_condition(t, s, traj_array):
                    break
            if istep % check_break_every == 0:
                if break_condition(t, s, traj_array):
                    warnings.warn("The break condition is met.")
                    break
        t, s, cache = dyn.step(t, s, cache)
    properties_dict = {field: np.array(value) for field, value in properties_dict.items()}
    if save_traj:
        output = {'time': time_array, 'states': traj_array, **properties_dict}
    else:
        output = {'time': time_array, **properties_dict}
    return output

# %% The helper functions

def estimate_scatter_dt(deriv: callable, r_bounds: tuple, s0: State, nsample: Real=30, t_bounds: Tuple[Real]=None) -> float:
    _, p0, rho0 = s0.get_variables()
    r_list = np.linspace(*r_bounds, nsample)
    if t_bounds is not None:
        t_list = np.random.uniform(*t_bounds, nsample)
    else:
        t_list = np.zeros(nsample)
    _dt = 99999999999
    for i in range(nsample):
        s0 = State.from_variables(R=r_list[i], P=p0, rho=rho0)
        _dt = min(_dt, evaluate_initial_dt(deriv, t_list[i], s0, order=4, atol=1e-8, rtol=1e-6,))
    return _dt

def _append_properties(properties_dict: dict, properties: NamedTuple) -> dict:
    for (field, value) in zip(properties._fields, properties):
        properties_dict[field].append(value)
    return properties_dict
    

# %% The temporary test code
def _debug_test():
    import time
    from pymddrive.models.tullyone import TullyOnePulseTypes, TD_Methods, get_tullyone
    import matplotlib.pyplot as plt
    
    # get a convenient model for testing
    hamiltonian = get_tullyone(
        pulse_type=TullyOnePulseTypes.NO_PULSE, 
    )
    mass = 2000.0
    
    # initial conditions
    t0 = 0.0; r0 = -10.0; p0 = 30.0
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    
    # prepare the dynamics object
    dyn = NonadiabaticDynamics( 
        hamiltonian=hamiltonian,
        t0=t0, s0=s0, mass=mass,
        qm_rep=QunatumRepresentation.DensityMatrix,
        basis_rep=BasisRepresentation.Diabatic,
        solver=NonadiabaticDynamicsMethods.EHRENFEST,
        numerical_integrator=NumericalIntegrators.RK4,
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
    
    
    # initial conditions
    t0 = 0.0; r0 = -10.0; p0 = 30.0
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    
    # prepare the dynamics object
    dyn = NonadiabaticDynamics( 
        hamiltonian=hamiltonian,
        t0=t0, s0=s0, mass=mass,
        qm_rep=QunatumRepresentation.DensityMatrix,
        basis_rep=BasisRepresentation.Diabatic,
        solver=NonadiabaticDynamicsMethods.EHRENFEST,
        numerical_integrator=NumericalIntegrators.ZVODE,
        r_bounds=(-10.0, 10.0)
    ) 
    start = time.perf_counter() 
    output_zvode = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    print(f"The time for the simulation using scipy zode is {time.perf_counter()-start} s.")
    
    t0, t1 = output['time'], output_zvode['time']
    r0, r1 = output['states']['R'], output_zvode['states']['R']
    p0, p1 = output['states']['P'], output_zvode['states']['P']
    # rho0, rho1= output['states']['rho'], output_zvode['states']['rho']
    pop0, pop1 = output['populations'], output_zvode['populations']
    
    plt.plot(t0, r0, ls='-', label='home-made RK4')
    plt.plot(t1, r1, ls='-.', label='scipy zvode')
    plt.xlabel('Time')
    plt.ylabel('R')
    plt.legend()
    plt.title('Position') 
    plt.show()
    plt.plot(t0, p0, ls='-', label='home-made RK4')
    plt.plot(t1, p1, ls='-.', label='scipy zvode')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('P')
    plt.title('Momentum')
    plt.show()
    plt.plot(t0, pop0[:, 0].real, ls='-', label='pop0: home-made RK4') 
    plt.plot(t0, pop0[:, 1].real, ls='-', label='pop1: home-made RK4') 
    plt.plot(t1, pop1[:, 0].real, ls='-.', label='pop0: scipy zvode')
    plt.plot(t1, pop1[:, 1].real, ls='-.', label='pop1: scipy zvode')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()
    
     
    
# %% the __main__ code
if __name__ == "__main__":
    _debug_test()
    
# %%
