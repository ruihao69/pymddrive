from numpy.typing import ArrayLike
from scipy.integrate import ode

from pymddrive.integrators import State
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.dynamics import ehrenfest
from pymddrive.dynamics.dynamics import Dynamics
from pymddrive.dynamics.options import (
    BasisRepresentation, QunatumRepresentation,
    NonadiabaticDynamicsMethods, NumericalIntegrators
)
from pymddrive.dynamics.misc_utils import estimate_scatter_dt

import warnings
from typing import Union, Tuple, Any, Type, Callable
from numbers import Real
from functools import partial



class NonadiabaticDynamics(Dynamics):
    def __init__(
        self,
        hamiltonian: Union[HamiltonianBase, None] = None,
        t0: Real= 0.0, s0: Union[State, None] = None, mass: Union[Real, None, ArrayLike] = None,
        dt: Union[float, None] = None, atol: float=1e-8, rtol: float=1e-6, safety: float=0.9, save_every: int = 10,
        qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
        basis_rep: BasisRepresentation = BasisRepresentation.Adiabatic,
        solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST, 
        numerical_integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
        r_bounds: Union[Tuple[Real], None] = None,
        t_bounds: Union[Tuple[Real], None] = None,
        max_step: float = None,
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
                
        self.step = self.get_stepper(solver, numerical_integrator, max_step)
        
    def get_stepper(
        self, 
        method: NonadiabaticDynamicsMethods,
        numerical_integrator: NumericalIntegrators,
        max_step: float
    ) -> Callable[[float, State, Any], Tuple[float, State, Any]]:
        if method == NonadiabaticDynamicsMethods.EHRENFEST:
            if numerical_integrator != NumericalIntegrators.ZVODE:
                raw_stepper = ehrenfest.choose_ehrenfest_stepper(numerical_integrator)
                return partial(raw_stepper, dt=self.dt, hamiltonian=self.hamiltonian, mass=self.mass, basis_rep=self.basis_rep)
            else:
                if max_step is None:
                    self.ode_solver = ode(self.deriv).set_integrator(
                        'zvode',
                        method='bdf',
                        atol=self.atol,
                        rtol=self.rtol,
                    )
                elif isinstance(max_step, Real):
                    self.ode_solver = ode(self.deriv).set_integrator(
                        'zvode',
                        method='bdf',
                        atol=self.atol,
                        rtol=self.rtol,
                        max_step=max_step
                    )
                else:
                    raise ValueError(f"Invalid value for {max_step=}.")
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