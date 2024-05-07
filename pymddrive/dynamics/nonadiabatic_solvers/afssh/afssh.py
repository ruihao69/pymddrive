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
from pymddrive.dynamics.nonadiabatic_solvers.afssh.aux_variables import AuxVariables
from pymddrive.dynamics.nonadiabatic_solvers.afssh.afssh_math_utils import evaluate_delta_F, tildify_diagonal_operator, un_tildify_diagonal_operator
from pymddrive.dynamics.nonadiabatic_solvers.afssh.aux_variables_vv import delta_R_dot, delta_P_dot
from pymddrive.dynamics.nonadiabatic_solvers.afssh.afssh_decoherence import afssh_decoherence
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, QuasiFloquetHamiltonianBase, evaluate_hamiltonian, evaluate_nonadiabatic_couplings, diagonalization
from pymddrive.models.nonadiabatic_hamiltonian import adiabatic_to_diabatic
from pymddrive.low_level.states import State
from pymddrive.low_level.surface_hopping import fssh_surface_hopping


from typing import Tuple, Union

@define
class AFSSH(NonadiabaticSolverBase):
    dim_hamiltonian: int = field(on_setattr=attr.setters.frozen)
    dim_electronic: int = field(on_setattr=attr.setters.frozen) 
    dim_nuclear: int = field(on_setattr=attr.setters.frozen)
    dt: float = field(on_setattr=attr.setters.frozen)
    quantum_representation: QuantumRepresentation = field(on_setattr=attr.setters.frozen)
    basis_representation: BasisRepresentation = field(on_setattr=attr.setters.frozen)
    hamiltonian: HamiltonianBase = field(on_setattr=attr.setters.frozen) 
    cache: Cache
    evecs_0: Union[None, GenericOperator] = field(default=None)
    auxvars: AuxVariables = field(on_setattr=attr.setters.frozen, default=None)
    
    def callback(self, t: float, state: State) -> Tuple[State, bool]:
        # compute the Hamiltonian at the current time
        R, P, rho = state.get_variables()
        v = state.get_v()
        H, dHdR = evaluate_hamiltonian(t, R, self.hamiltonian)
        evals, evecs = diagonalization(H, self.hamiltonian._last_evecs)
        d, F = evaluate_nonadiabatic_couplings(dHdR=dHdR, evals=evals, evecs=evecs)
        self.hamiltonian.update_last_evecs(evecs)
        
        # surface hopping
        current_active_surface: int = self.cache.active_surface
        dt: float = self.dt
        P_current: RealVector = P
        v_dot_d = compute_v_dot_d(v, d)
        mass = state.get_mass()
        hop_flag, new_active_surface, P_new = fssh_surface_hopping(
            dt, current_active_surface[0], P_current, rho, evals, v_dot_d, d, mass
        )
        new_active_surface = np.array([new_active_surface])
        
        # A-FSSH decoherence: Diagonal approximation of the equations of motion 
        # for the moments delta_R and delta_P
        # --- evaluate the new delta_F_tilde, and update the auxiliary variables
        delta_F_prev_tilde = self.auxvars.delta_F_prev.copy() # F_tilde(t0) = F(t0)
        delta_F = evaluate_delta_F(F, new_active_surface)     # evaluate F(t0+dt)
        self.auxvars.delta_F_prev[:] = delta_F                # update F(t0+dt)
        delta_F_tilde = tildify_diagonal_operator(delta_F, evecs) # evaulate F_tilde(t0+dt)
        
        # --- evaluate the new delta_R_tilde, and update the auxiliary variables
        rho_diabatic = adiabatic_to_diabatic(rho, evecs)
        delta_R_tilde = delta_R_dot(self.auxvars.delta_R, self.auxvars.delta_P, delta_F_prev_tilde, mass, dt, rho_diabatic)
        delta_P_tilde = delta_P_dot(self.auxvars.delta_P, delta_F_prev_tilde, delta_F_tilde, dt, rho_diabatic)
        delta_R = un_tildify_diagonal_operator(delta_R_tilde, evecs)
        delta_P = un_tildify_diagonal_operator(delta_P_tilde, evecs)
        
        # --- apply the decoherence to the density matrix and the auxiliary variables
        # --- update the density matrix and the auxiliary variables
        rho, delta_R, delta_P, collapsed_flag = afssh_decoherence(rho, delta_F, delta_R, delta_P, new_active_surface, d, evals, dt)
        self.auxvars.delta_R[:] = delta_R
        self.auxvars.delta_P[:] = delta_P
       
        # update the cache 
        self.cache.update_cache(H=H, evals=evals, evecs=evecs, dHdR=dHdR, nac=d, active_surface=new_active_surface)
        
        # return the state after callback, as well as the update flag for the numerical integrator
        if (hop_flag or collapsed_flag):
            new_state = state.from_unstructured(np.concatenate([R, P_new, rho.flatten()], dtype=np.complex128))
            return new_state, True
        else:
            return state, False 
        
    def derivative(self, t: float, state: State) -> State:
        R, P, rho = state.get_variables()
        H, dHdR = evaluate_hamiltonian(t, R, self.hamiltonian)
        v = state.get_v()
        if self.basis_representation == BasisRepresentation.DIABATIC:
            dR, dP, drho = self.derivative_diabatic(v, rho, H, dHdR, self.cache.F_langevin, self.cache.active_surface)
        elif self.basis_representation == BasisRepresentation.ADIABATIC:
            dR, dP, drho = self.derivative_adiabatic(v, rho, H, dHdR, self.cache.F_langevin, self.hamiltonian._last_evecs, self.cache.active_surface)
        else:
            raise ValueError("Invalid basis representation")
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
    ) -> 'AFSSH':
        R, P, rho_or_psi = state.get_variables()
        active_surface = initialize_active_surface(rho_or_psi)
        
        dim_nuclear = R.shape[0] 
        dim_hamiltonian = rho_or_psi.shape[0]
        dim_electronic = hamiltonian.dim
        if rho_or_psi.ndim == 1:
            raise ValueError("A-FSSH in pymddrive only supports density matrix representation, for now.")
        elif rho_or_psi.ndim == 2:
            quantum_representation = QuantumRepresentation.DENSITY_MATRIX
        else:
            raise ValueError(f"Invalid quantum representation: {rho_or_psi.ndim=}")
        
        H, dHdR = evaluate_hamiltonian(0.0, R, hamiltonian)
        evals, evecs = diagonalization(H, hamiltonian._last_evecs)
        d, F = evaluate_nonadiabatic_couplings(dHdR=dHdR, evals=evals, evecs=evecs)
        
        # initialize the cache 
        cache = Cache.from_dimensions(dim_elec=dim_hamiltonian, dim_nucl=dim_nuclear)
        cache.update_cache(H=H, evals=evals, evecs=evecs, dHdR=dHdR, nac=d, active_surface=active_surface) 
        
        # initialize the auxiliary variables
        delta_F = evaluate_delta_F(F, active_surface)
        auxvars = AuxVariables(
            delta_R=np.zeros((dim_hamiltonian, dim_nuclear), dtype=np.float64),
            delta_P=np.zeros((dim_hamiltonian, dim_nuclear), dtype=np.float64),
            delta_F_prev=delta_F,
        )
        
        return cls(
            dim_hamiltonian=dim_hamiltonian,
            dim_electronic=dim_electronic,
            dim_nuclear=dim_nuclear,
            dt=dt,
            quantum_representation=quantum_representation,
            basis_representation=basis_representation,
            hamiltonian=hamiltonian,
            cache=cache,
            auxvars=auxvars
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
        