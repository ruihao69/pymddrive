import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import ode

from pymddrive.integrators import State
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, adiabatic_to_diabatic, diabatic_to_adiabatic
from pymddrive.dynamics import ehrenfest
from pymddrive.dynamics import fssh
from pymddrive.dynamics.dynamics import Dynamics
from pymddrive.dynamics.options import (
    BasisRepresentation, QunatumRepresentation,
    NonadiabaticDynamicsMethods, NumericalIntegrators
)
from pymddrive.integrators.rk4 import rk4
from pymddrive.dynamics.misc_utils import estimate_scatter_dt, assert_valid_real_positive_value, eval_nonadiabatic_hamiltonian

import warnings
import logging
from typing import Union, Tuple, Any, Type, Callable, Optional, Dict
from numbers import Real
from functools import partial

MAX_STEPS: int = int(1e6)


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
        max_step: float = None, min_step: float = None,
    ) -> None:
        super().__init__(
            hamiltonian, t0, s0, mass, dt, atol, rtol, safety, save_every, numerical_integrator
        ) 
        
        self.do_hopping = False
        
        if qm_rep != QunatumRepresentation.DensityMatrix:
            raise NotImplementedError(f"At this time, NonadiabaticDynamics class only supports the Density Matrix representation for quantum system.")
        
        self.qm_rep = qm_rep
        self.basis_rep = basis_rep
        self.nonadiabatic_method = solver
        
        self.cache_initializer = self.get_cahce_initializer(solver) 
        self.deriv = self.get_deriv(qm_rep, solver, basis_rep, numerical_integrator)
        self.calculate_properties, self.properties_type = self.get_properties_calculator(solver)
        
        self.dt = dt if dt is not None else 0.03
        # Create a logger
        logger = logging.getLogger(__name__)

        if numerical_integrator != NumericalIntegrators.ZVODE:
            if r_bounds is not None:
                prop_dt_scatter = estimate_scatter_dt(self.deriv, r_bounds, s0, nsample=100, t_bounds=t_bounds)
                if self.dt > prop_dt_scatter:
                    warnings.warn(f"The intial dt {self.dt} is larger than the estimated scatter dt {prop_dt_scatter}. Changing to the scatter dt.")
                    self.dt = prop_dt_scatter
            logger.info(f"Using the {numerical_integrator} solver, where {self.dt=} is used for the fixed dt.")
        else:
            logger.info(f"Using the zvode solver, where {self.dt=} is used for dense output.")
                
        self.step = self.get_stepper(solver, numerical_integrator, max_step, min_step)
        
    def get_stepper(
        self, 
        method: NonadiabaticDynamicsMethods,
        numerical_integrator: NumericalIntegrators,
        max_step: float,
        min_step: float,
    ) -> Callable[[float, State, Any], Tuple[float, State, Any]]:
        if method == NonadiabaticDynamicsMethods.EHRENFEST:
            if numerical_integrator != NumericalIntegrators.ZVODE:
                raw_stepper = ehrenfest.choose_ehrenfest_stepper(numerical_integrator)
                return partial(raw_stepper, dt=self.dt, hamiltonian=self.hamiltonian, mass=self.mass, basis_rep=self.basis_rep)
            else:
                ode_integrator_options = self._get_ode_integrator_options(max_step, min_step)
                self.ode_solver = ode(self.deriv).set_integrator(**ode_integrator_options)
                self.ode_solver.set_initial_value(self.s0.flatten(), self.t0)
                # self.ode_solver.set_solout(solout)
                return self._step_zvode 
        elif method == NonadiabaticDynamicsMethods.FSSH:
            if numerical_integrator != NumericalIntegrators.ZVODE:
                if numerical_integrator == NumericalIntegrators.RK4:
                    raw_stepper = fssh.choose_fssh_stepper(numerical_integrator)
                    return partial(raw_stepper, dt=self.dt, hamiltonian=self.hamiltonian, mass=self.mass, basis_rep=self.basis_rep)
            elif numerical_integrator == NumericalIntegrators.ZVODE:
                ode_integrator_options = self._get_ode_integrator_options(max_step, min_step)
                self.ode_solver = ode(self.deriv).set_integrator(**ode_integrator_options)
                self.ode_solver.set_initial_value(self.s0.flatten(), self.t0)
                self.do_hopping = True
                return self._step_zvode
        else:
            raise NotImplemented(f"Unrecogonized nonadiabatic {method=}. Not implemented in pymddrive.")
        
    def get_cahce_initializer(
        self, 
        method: NonadiabaticDynamicsMethods
    ) -> Callable[[float, State, Any], Any]:
        if method == NonadiabaticDynamicsMethods.EHRENFEST:
            return partial(ehrenfest.initialize_cache, hamiltonian=self.hamiltonian, basis_rep=self.basis_rep)
        elif method == NonadiabaticDynamicsMethods.FSSH:
            return partial(fssh.initialize_cache, hamiltonian=self.hamiltonian, basis_rep=self.basis_rep)
        
    def _get_ode_integrator_options(
        self, 
        max_step: Optional[float]=None, 
        min_step: Optional[float]=None
    ) -> Dict[str, Any]:
        ode_integrator_options = {
            'name': 'zvode',
            'method': 'bdf',
            'atol': self.atol, 
            'rtol': self.rtol,
            'nsteps': MAX_STEPS
        }
        if max_step is not None:
            assert_valid_real_positive_value(max_step)
            ode_integrator_options['max_step'] = max_step
        if min_step is not None:
            assert_valid_real_positive_value(min_step)
            if max_step is not None and max_step < min_step:
                raise ValueError(f"The {max_step=} is smaller than the {min_step=}.")
            else:
                ode_integrator_options['min_step'] = min_step
        return ode_integrator_options
    
    def _diabatic_ehrenfest_in_conical_intersection(self, t: float, s: State, cache: Any) -> Tuple[float, State, Any]:
        TOL_CONOCAL_INTERSECTION: float = 3e-5
        # RTOL = 3e-3
        # R_CI = 0.0
        dE_min: float = -1
        deriv_diabatic_ehrenfest = self.get_deriv(
            quatum_representation=self.qm_rep,
            method=NonadiabaticDynamicsMethods.EHRENFEST,
            basis_representation=BasisRepresentation.Diabatic,
            numerical_integrator=NumericalIntegrators.RK4
        )
        state_during_conical_intersection = State.from_unstructured(s.flatten(), dtype=self.dtype, stype=self.stype, copy=True)
        R, _, rho = state_during_conical_intersection.get_variables()
        hami_return = eval_nonadiabatic_hamiltonian(t, R, self.hamiltonian, basis_rep=BasisRepresentation.Diabatic)
        rho[:] = adiabatic_to_diabatic(rho, hami_return.evecs)
        
        # while (dE_min < TOL_CONOCAL_INTERSECTION) and (R[0] < R_CI + RTOL):
        while (dE_min < TOL_CONOCAL_INTERSECTION):
            t, state_during_conical_intersection = rk4(
                t, state_during_conical_intersection, deriv_diabatic_ehrenfest, dt=self.dt
            )
            R, _, _ = state_during_conical_intersection.get_variables()
            hami_return = eval_nonadiabatic_hamiltonian(t, R, self.hamiltonian, basis_rep=BasisRepresentation.Diabatic)
            evals = hami_return.evals
            dE_min = np.min(evals[1:] - evals[:-1]) 
            
        R, P, rho = state_during_conical_intersection.get_variables()
        rho[:] = diabatic_to_adiabatic(rho, hami_return.evecs)
        state_after_conical_intersection = State.from_unstructured(state_during_conical_intersection.flatten(), dtype=self.dtype, stype=self.stype, copy=True)
        return t, state_after_conical_intersection, cache
        
        
    def _step_zvode(self, t: float, s: State, cache: Any) -> Tuple[float, State, Any]:
        if not self.ode_solver.successful():
            raise RuntimeError("The ode solver is not successful.")
        if t != self.ode_solver.t:
            raise ValueError(f"The time {t} is not the same as the solver time {self.ode_solver.t}.")
        self.ode_solver.set_f_params(cache)
        self.ode_solver.integrate(self.ode_solver.t + self.dt)
        t, y = self.ode_solver.t, self.ode_solver.y
        s_new = State.from_unstructured(y, dtype=self.dtype, stype=self.stype, copy=False)
        R_new, P_new, rho_new = s_new.get_variables()
        PTOL : float = 1e2
        hami_return = eval_nonadiabatic_hamiltonian(t, R_new, self.hamiltonian, basis_rep=self.basis_rep)
        self.hamiltonian.update_last_evecs(hami_return.evecs) 

        # if self.basis_rep == BasisRepresentation.Adiabatic:
        #     print(f'{R_new=}', f'{P_new=}', f'{np.isnan(rho_new).any()=}')
        # self.hamiltonian.update_last_deriv_couplings(hami_return.d)
        if self.do_hopping:
            has_hopped, new_active_surf, P_rescaled = fssh.hopping(self.dt, rho_new, hami_return, P_new, self.mass, cache.active_surf)
            if has_hopped:
                # dE = hami_return.evals[new_active_surf.index] - hami_return.evals[cache.active_surf.index]
                # dPE = np.sum(P_rescaled**2 - P_new**2) / (2 * self.mass)
                # print(f"Has hopped. P was {P_new=}, now {P_rescaled=}. {dE=}, {dPE=}.")
                P_new[:] = P_rescaled
                self.ode_solver.set_initial_value(s_new.flatten(), t)
                # print(f"Hopped. Checking the ode solver initial value status. {P_new=}, {self.ode_solver.y[1]=}")
                # print(f'Hopping at {t=}, {new_active_surf=}, prev_surf={cache.active_surf},')
                cache = fssh.FSSHCache(active_surf=new_active_surf, evals=hami_return.evals, evecs=hami_return.evecs, H_diab=hami_return.H, hamiltonian=self.hamiltonian)
            else:
                cache = fssh.FSSHCache(active_surf=cache.active_surf, evals=hami_return.evals, evecs=hami_return.evecs, H_diab=hami_return.H, hamiltonian=self.hamiltonian)
        evals = hami_return.evals
        dE_min :float = np.min(evals[1:] - evals[:-1])
        TOL_CONOCAL_INTERSECTION = 3e-5
        if (dE_min < TOL_CONOCAL_INTERSECTION) and (self.basis_rep==BasisRepresentation.Adiabatic):
            warnings.warn(f"Alert! The energy gap {dE_min} is smaller than the tolerance {TOL_CONOCAL_INTERSECTION}. We might be near a conical intersection. Switching to the diabatic representation for the dynamics.")
            t, s_new, cache = self._diabatic_ehrenfest_in_conical_intersection(t, s_new, cache)
            self.ode_solver.set_initial_value(s_new.flatten(), t)
        
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
            raw_deriv = fssh.choose_fssh_deriv(quatum_representation)
            if numerical_integrator != NumericalIntegrators.ZVODE:
                return partial(raw_deriv, hamiltonian=self.hamiltonian, mass=self.mass, basis_rep=basis_representation)
            else:
                def deriv_wrapper(t, y: ArrayLike, cache: fssh.FSSHCache, copy: bool=False)->ArrayLike:
                    s = State.from_unstructured(y, dtype=self.dtype, stype=self.stype, copy=copy)
                    dsdt = raw_deriv(t, s, hamiltonian=self.hamiltonian, mass=self.mass, basis_rep=basis_representation, active_surface=cache.active_surf)
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
            raw_calculator = fssh.calculate_properties
            calculator = partial(raw_calculator, mass=self.mass)
            return calculator, fssh.FSSHProperties
        else:
            raise NotImplemented(f"Unrecogonized nonadiabatic {method=}. Not implemented in pymddrive.")
        
# def post_step_callback(t, y, hamiltonian, stype, dtype, basis_rep) -> None:
#     s = State.from_unstructured(y, dtype=dtype, stype=stype)
#     R, _, _ = s.get_variables()
#     hami_return = eval_nonadiabatic_hamiltonian(t, R, hamiltonian, basis_rep)
#     hamiltonian.update_last_evecs(hami_return.evecs)
#     hamiltonian.update_last_deriv_couplings(hami_return.d)