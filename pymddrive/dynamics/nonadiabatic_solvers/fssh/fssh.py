import attr
import numpy as np
from attrs import define, field

from pymddrive.my_types import RealVector, ComplexVector, ComplexOperator, GenericOperator, GenericVectorOperator, ActiveSurface
from pymddrive.dynamics.cache import Cache
from pymddrive.dynamics.options import BasisRepresentation, QuantumRepresentation
from pymddrive.dynamics.nonadiabatic_solvers.nonadiabatic_solver_base import NonadiabaticSolverBase, NonadiabaticProperties
from pymddrive.dynamics.nonadiabatic_solvers.math_utils import adiabatic_equations_of_motion, compute_v_dot_d
from pymddrive.dynamics.nonadiabatic_solvers.fssh.fssh_math_utils import initialize_active_surface
from pymddrive.dynamics.nonadiabatic_solvers.fssh.populations import compute_floquet_populations, compute_populations
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, QuasiFloquetHamiltonianBase, evaluate_hamiltonian, evaluate_nonadiabatic_couplings, diagonalization
from pymddrive.low_level.states import State
from pymddrive.low_level.surface_hopping import fssh_surface_hopping


from typing import Tuple, Union


@define
class FSSH(NonadiabaticSolverBase):
    dim_hamiltonian: int = field(on_setattr=attr.setters.frozen)
    dim_electronic: int = field(on_setattr=attr.setters.frozen)
    dim_nuclear: int = field(on_setattr=attr.setters.frozen)
    dt: float = field(on_setattr=attr.setters.frozen)
    quantum_representation: QuantumRepresentation = field(on_setattr=attr.setters.frozen)
    basis_representation: BasisRepresentation = field(on_setattr=attr.setters.frozen)
    hamiltonian: HamiltonianBase = field(on_setattr=attr.setters.frozen)
    cache: Cache
    evecs_0: Union[None, GenericOperator] = field(default=None)
    
    def callback(self, t: float, state: State) -> Tuple[State, bool]:
        # compute the Hamiltonian at the current time
        R, P, rho_or_psi = state.get_variables()
        v = state.get_v()
        H, dHdR = evaluate_hamiltonian(t, R, self.hamiltonian)
        evals, evecs = diagonalization(H, self.hamiltonian._last_evecs)
        d, _ = evaluate_nonadiabatic_couplings(dHdR=dHdR, evals=evals, evecs=evecs)
        self.hamiltonian.update_last_evecs(evecs)
        
        # surface hopping
        current_active_surface: int = self.cache.active_surface
        dt: float = self.dt
        P_current: RealVector = P
        v_dot_d = compute_v_dot_d(v, d)
        mass = state.get_mass()
        # print(f"{type(dt)=}")
        # print(f"{type(current_active_surface)=}, {current_active_surface.dtype=}")
        # print(f"{type(P_current)=}, {P_current.dtype=} {P_current.shape=}")
        # print(f"{type(rho_or_psi)=}, {rho_or_psi.dtype=} {rho_or_psi.shape=}")
        # print(f"{type(evals)=}, {evals.dtype=} {evals.shape=}")
        # print(f"{type(v_dot_d)=}, {v_dot_d.dtype=} {v_dot_d.shape=}")
        # print(f"{type(d)=}, {d.dtype=}, {d.shape=}")
        
        hop_flag, new_active_surface, P_new = fssh_surface_hopping(
            dt, current_active_surface[0], P_current, rho_or_psi, evals, v_dot_d, d, mass
        )
        
        # rho = rho_or_psi if rho_or_psi.ndim == 2 else np.outer(rho_or_psi, rho_or_psi.conjugate())
        # hop_flag, new_active_surface, P_new = fssh_surface_hopping(
        #     dt, current_active_surface[0], P_current, rho, evals, v_dot_d, d, mass
        # )
        # print(f"With density matrix: {hop_flag=}, {new_active_surface=}")
                
        # update the cache 
        self.cache.update_cache(H=H, evals=evals, evecs=evecs, dHdR=dHdR, nac=d, active_surface=new_active_surface)
        
        # return the state after callback, as well as the update flag for the numerical integrator
        if hop_flag:
            new_state = state.from_unstructured(np.concatenate([R, P_new, rho_or_psi.flatten()], dtype=np.complex128))
            return new_state, True
        else:
            return state, False
        
    def derivative(self, t: float, state: State) -> State:
        R, P, rho_or_psi = state.get_variables()
        H, dHdR = evaluate_hamiltonian(t, R, self.hamiltonian)
        v = state.get_v()
        if self.basis_representation == BasisRepresentation.DIABATIC:
            dR, dP, drho = self.derivative_diabatic(v, rho_or_psi, H, dHdR, self.cache.F_langevin, self.cache.active_surface)
        elif self.basis_representation == BasisRepresentation.ADIABATIC:
            dR, dP, drho = self.derivative_adiabatic(v, rho_or_psi, H, dHdR, self.cache.F_langevin, self.hamiltonian._last_evecs, self.cache.active_surface)
        else:
            raise ValueError("Unsupported basis representation.")
        return state.from_unstructured(np.concatenate([dR, dP, drho.flatten()], dtype=np.complex128))
        
    @staticmethod
    def derivative_diabatic(
        v: RealVector,
        rho_or_psi: Union[ComplexVector, ComplexOperator],
        H: GenericOperator,
        dHdR: GenericVectorOperator,
        F_langevin: RealVector,
        active_surface: ActiveSurface
    ) -> Tuple[RealVector, RealVector, Union[ComplexVector, ComplexOperator]]:
        raise NotImplementedError(f"Surface hopping has poor performance in diabatic representation. Hence, this method is not implemented.")
    
    @staticmethod
    def derivative_adiabatic(
        v: RealVector,
        rho_or_psi: Union[ComplexVector, ComplexOperator],
        H: GenericOperator,
        dHdR: GenericVectorOperator,
        F_langevin: RealVector,
        last_evecs: GenericOperator,
        active_surface: ActiveSurface,
    ) -> Tuple[RealVector, RealVector, Union[ComplexVector, ComplexOperator]]:
        # diagonalize the Hamiltonian
        evals, evecs = diagonalization(H, last_evecs)
        # compute the nonadiabatic couplings
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        # evaluate the v_dot_d term
        v_dot_d = compute_v_dot_d(v, d)
        
        R_dot = v
        active_state = active_surface[0]
        P_dot = F[active_state, ...] + F_langevin
        rho_or_psi_dot = adiabatic_equations_of_motion(rho_or_psi, evals, v_dot_d)
        return R_dot, P_dot, rho_or_psi_dot
    
    @classmethod
    def initialize(
        cls,
        state: State, 
        hamiltonian: HamiltonianBase,
        basis_representation: BasisRepresentation,
        dt: float,
    ) -> 'FSSH':
        R, P, rho_or_psi = state.get_variables()
        active_surface = initialize_active_surface(rho_or_psi)
        
        dim_nuclear = R.shape[0]
        dim_hamiltonian = rho_or_psi.shape[0]
        dim_electronic = hamiltonian.dim
        quantum_representation = QuantumRepresentation.WAVEFUNCTION if rho_or_psi.ndim > 1 else QuantumRepresentation.DENSITY_MATRIX
        
        H, dHdR = evaluate_hamiltonian(0.0, R, hamiltonian)
        evals, evecs = diagonalization(H, hamiltonian._last_evecs)
        d, F = evaluate_nonadiabatic_couplings(dHdR=dHdR, evals=evals, evecs=evecs)
        
        cache = Cache.from_dimensions(dim_elec=dim_hamiltonian, dim_nucl=dim_nuclear)
        cache.update_cache(H=H, evals=evals, evecs=evecs, dHdR=dHdR, nac=d, active_surface=active_surface)
        
        return cls(
            dim_nuclear=dim_nuclear,
            dim_hamiltonian=dim_hamiltonian,
            dim_electronic=dim_electronic,
            dt=dt,
            quantum_representation=quantum_representation,
            basis_representation=basis_representation,
            hamiltonian=hamiltonian,
            cache=cache
        )
        
    def calculate_properties(self, t: float, s: State) -> NonadiabaticProperties:
        R, P, rho_or_psi = s.get_variables()
        
        KE = NonadiabaticSolverBase.calculate_KE(P, s.get_mass())
        _active_surface = self.cache.active_surface[0]
        evals = self.cache.evals
        PE = evals[_active_surface]
        
        if isinstance(self.hamiltonian, QuasiFloquetHamiltonianBase):
            H0 = self.hamiltonian.H0(R=R) 
            _, evecs_0 = diagonalization(H0, prev_evecs=self.evecs_0)
            if self.evecs_0 is None:
                self.evecs_0 = np.copy(evecs_0)
            else:
                np.copyto(self.evecs_0, evecs_0)
            adiabatic_populations = compute_floquet_populations(
                state=rho_or_psi, 
                dynamics_basis=self.basis_representation,
                floquet_basis=BasisRepresentation.DIABATIC,
                target_state_basis=BasisRepresentation.ADIABATIC,
                Omega=self.hamiltonian.get_carrier_frequency(),
                t=t,
                dim=self.hamiltonian.dim,
                NF=self.hamiltonian.NF,
                evecs_F=self.cache.evecs,
                evecs_0=evecs_0,
                active_surface=self.cache.active_surface
            )
            diabatic_populations = compute_floquet_populations(
                state=rho_or_psi, 
                dynamics_basis=self.basis_representation,
                floquet_basis=BasisRepresentation.DIABATIC,
                target_state_basis=BasisRepresentation.DIABATIC,
                Omega=self.hamiltonian.get_carrier_frequency(),
                t=t,
                dim=self.hamiltonian.dim,
                NF=self.hamiltonian.NF, 
                evecs_F=self.cache.evecs,
                evecs_0=evecs_0,
                active_surface=self.cache.active_surface
            )
        else:
            adiabatic_populations = compute_populations(rho_or_psi, self.basis_representation, BasisRepresentation.ADIABATIC, self.cache.evecs, self.cache.active_surface)
            diabatic_populations = compute_populations(rho_or_psi, self.basis_representation, BasisRepresentation.DIABATIC, self.cache.evecs, self.cache.active_surface)
            
        return NonadiabaticProperties(R=R, P=P, adiabatic_populations=adiabatic_populations, diabatic_populations=diabatic_populations, KE=KE, PE=PE)  
        
       
    def get_dim_nuclear(self) -> int:
        return self.dim_nuclear 
     
    def get_dim_electronic(self) -> int:
        return self.dim_electronic
    
    def update_F_langevin(self, F_langevin: RealVector) -> None:
        np.copyto(self.cache.F_langevin, F_langevin)
        