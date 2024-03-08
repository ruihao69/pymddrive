# %% The package code
import numpy as np

from typing import Union, Tuple, Any, Type, Callable, NamedTuple
    
from numbers import Real
from numpy.typing import ArrayLike

from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.integrators.state import State

from abc import ABC

from pymddrive.dynamics.options import NumericalIntegrators
    
class Dynamics(ABC):
    def __init__(
        self,
        hamiltonian: HamiltonianBase,
        t0: float, s0: State, mass: Union[float, ArrayLike], 
        dt: Union[float, None]=None, 
        atol: float=1e-8, rtol: float=1e-8, safety: float=0.9, save_every: int=10,
        numerical_integrator: NumericalIntegrators=NumericalIntegrators.ZVODE,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.t0 = t0    
        self.s0 = s0    
        self.mass = self._process_mass(s0, mass)
        self.dt = dt
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
        # the cache calculator: takes the current time, state, and cache, returns the updated cache
        # self.calculate_cache: Callable[[float, State, Any], Any] = None
        self.cache_initializer: Callable[[float, State], Any] = None
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