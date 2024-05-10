import attr
import numpy as np
from attrs import define, field

from pymddrive.my_types import RealVector, ComplexVector, ComplexOperator, GenericOperator, GenericVectorOperator, ActiveSurface, ComplexVectorOperator
from pymddrive.dynamics.cache import Cache
from pymddrive.dynamics.options import BasisRepresentation, QuantumRepresentation
from pymddrive.dynamics.nonadiabatic_solvers.nonadiabatic_solver_base import NonadiabaticSolverBase, NonadiabaticProperties
from pymddrive.dynamics.nonadiabatic_solvers.math_utils import adiabatic_equations_of_motion, compute_v_dot_d
from pymddrive.dynamics.nonadiabatic_solvers.fssh.fssh_math_utils import initialize_active_surface
from pymddrive.dynamics.nonadiabatic_solvers.fssh.populations import compute_floquet_populations, compute_populations
from pymddrive.dynamics.nonadiabatic_solvers.afssh.moments_math_utils import dot_delta_P, dot_delta_R, evaluate_delta_vec_O, evaluate_delta_vec_O, delta_P_rescale
from pymddrive.dynamics.nonadiabatic_solvers.afssh.decoherence_rates import apply_decoherence
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, QuasiFloquetHamiltonianBase, evaluate_hamiltonian, evaluate_nonadiabatic_couplings, diagonalization
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

    def callback(self, t: float, state: State) -> Tuple[State, bool]:
        # compute the Hamiltonian at the current time
        # R, P, rho = state.get_variables()
        R, P, rho, delta_R, delta_P = state.get_afssh_variables()
        v = state.get_v()
        H, dHdR = evaluate_hamiltonian(t, R, self.hamiltonian)
        evals, evecs = diagonalization(H, self.hamiltonian._last_evecs)
        d, F, F_hellmann_feynman = evaluate_nonadiabatic_couplings(dHdR=dHdR, evals=evals, evecs=evecs)
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

        # --- if hopped, we reset delta_R and rescale delta_P
        if hop_flag:
            delta_R[:] = 0.0 # We reset the position moment
            delta_P[:] = delta_P_rescale(P=P_new, mass=mass, delta_P=delta_P, evals=evals, rho=rho, active_surface=new_active_surface, dc=d) # We rescale the momentum moment  
        
        # --- decoherence
        collapsed_or_reset_flag, new_rho, new_delta_R, new_delta_P = apply_decoherence(
            rho=rho, active_state=new_active_surface[0], F_hellmann_feynman=F_hellmann_feynman, delta_F=evaluate_delta_vec_O(F_hellmann_feynman, F[new_active_surface[0]]), delta_R=delta_R, delta_P=delta_P, dt=dt
        )
            
        # update the cache
        self.cache.update_cache(H=H, evals=evals, evecs=evecs, dHdR=dHdR, nac=d, active_surface=new_active_surface)

        # return the state after callback, as well as the update flag for the numerical integrator
        if (hop_flag or collapsed_or_reset_flag):
            new_state = state.from_unstructured(np.concatenate([R, P_new, new_rho.flatten(), new_delta_R.flatten('F'), new_delta_P.flatten('F')], dtype=np.complex128))
            return new_state, True
        else:
            return state, False

    def derivative(self, t: float, state: State) -> State:
        R, P, rho, delta_R, delta_P = state.get_afssh_variables()
        H, dHdR = evaluate_hamiltonian(t, R, self.hamiltonian)
        v = state.get_v()
        mass = state.get_mass()
        if self.basis_representation == BasisRepresentation.DIABATIC:
            raise NotImplementedError("Surface hopping has poor performance in diabatic representation. Hence, this method is not implemented.")
        elif self.basis_representation == BasisRepresentation.ADIABATIC:
            dR, dP, drho, delta_R_dot, delta_P_dot = self.derivative_adiabatic(v, rho, H, dHdR, self.cache.F_langevin, self.cache.evecs, self.cache.active_surface, delta_R, delta_P, mass)
        else:
            raise ValueError("Invalid basis representation")
        return state.from_unstructured(np.concatenate([dR, dP, drho.flatten(), delta_R_dot.flatten('F'), delta_P_dot.flatten('F')], dtype=np.complex128))

    @staticmethod
    def derivative_adiabatic(
        v: RealVector,
        rho: ComplexOperator,
        H: GenericOperator,
        dHdR: GenericVectorOperator,
        F_langevin: RealVector,
        last_evecs: GenericOperator,
        active_surface: ActiveSurface,
        delta_R: ComplexVectorOperator, 
        delta_P: ComplexVectorOperator,
        mass: Union[float, RealVector],
    ) -> Tuple[RealVector, RealVector, Union[ComplexVector, ComplexOperator]]:
        # diagonalize the Hamiltonian
        evals, evecs = diagonalization(H, last_evecs)
        # compute the nonadiabatic couplings
        d, F, F_hellmann_feynman = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        # evaluate the v_dot_d term
        v_dot_d = compute_v_dot_d(v, d)
        
        # evaluate R_dot
        R_dot = v
        
        # evaluate P_dot 
        active_state = active_surface[0]
        P_dot = F[active_state, ...] + F_langevin
        
        # evaluate delta_R_dot 
        delta_R_dot = dot_delta_R(evals, delta_R, delta_P, mass, v_dot_d, active_surface)
        
        # evaluate delta_P_dot
        delta_F = evaluate_delta_vec_O(F_hellmann_feynman, F[active_state])
        delta_P_dot = dot_delta_P(evals, delta_P, delta_F, v_dot_d, rho, active_surface)
        
        # evaluate the derivative of the density matrix or the wavefunction
        rho_dot = adiabatic_equations_of_motion(rho, evals, v_dot_d)
        
        return R_dot, P_dot, rho_dot, delta_R_dot, delta_P_dot

    @classmethod
    def initialize(
        cls,
        state: State,
        hamiltonian: HamiltonianBase,
        basis_representation: BasisRepresentation,
        dt: float,
    ) -> 'AFSSH':
        R, P, rho = state.get_variables()
        if rho.ndim == 1:
            raise NotImplementedError("A-FSSH in pymddrive only supports density matrix representation, for now.")
        
        active_surface = initialize_active_surface(rho)

        dim_nuclear = R.shape[0]
        dim_hamiltonian = rho.shape[0]
        dim_electronic = hamiltonian.dim
        
        if rho.ndim == 2:
            quantum_representation = QuantumRepresentation.DENSITY_MATRIX
        else:
            raise ValueError(f"Invalid quantum representation: {rho.ndim=}")

        H, dHdR = evaluate_hamiltonian(0.0, R, hamiltonian)
        evals, evecs = diagonalization(H, hamiltonian._last_evecs)
        d, F, F_hellmann_feynman = evaluate_nonadiabatic_couplings(dHdR=dHdR, evals=evals, evecs=evecs)

        # initialize the cache
        cache = Cache.from_dimensions(dim_elec=dim_hamiltonian, dim_nucl=dim_nuclear)
        cache.update_cache(H=H, evals=evals, evecs=evecs, dHdR=dHdR, nac=d, active_surface=active_surface)

        return cls(
            dim_hamiltonian=dim_hamiltonian,
            dim_electronic=dim_electronic,
            dim_nuclear=dim_nuclear,
            dt=dt,
            quantum_representation=quantum_representation,
            basis_representation=basis_representation,
            hamiltonian=hamiltonian,
            cache=cache,
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

