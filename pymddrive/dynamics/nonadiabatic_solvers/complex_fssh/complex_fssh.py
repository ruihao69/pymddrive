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
from pymddrive.dynamics.nonadiabatic_solvers.complex_fssh.complex_surface_hopping_py import complex_fssh_surface_hopping_py
from pymddrive.dynamics.nonadiabatic_solvers.complex_fssh.complex_fssh_math_utils import evaluate_Fmag
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, QuasiFloquetHamiltonianBase, evaluate_hamiltonian, evaluate_nonadiabatic_couplings, diagonalization
from pymddrive.low_level.states import State
from pymddrive.low_level.surface_hopping import fssh_surface_hopping

from typing import Tuple, Union


@define
class ComplexFSSH(NonadiabaticSolverBase):
    """
        FSSH for complex-valued hamiltonians. Following the J. Chem. Phys. 150, 124101 (2019). Denoted as Miao2019JCP.
        Two major ansatz are used in this theory:
        - “Magnetic force” ansatz, Lorentz-force like force arised from the Berry-curvature.
        - Robust direction of momentum rescaling: $\Re(e^{\im \eta} \bm{d}$), where $\eta$ is an optimal phase that maximizes the norm of the real part of the derivative coupling $\bm{d}$.
    """
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
        d, _, _ = evaluate_nonadiabatic_couplings(dHdR=dHdR, evals=evals, evecs=evecs)
        self.hamiltonian.update_last_evecs(evecs) # update the last eigenvectors for adiabatic phase alignment
        
        # complex surface hopping (Miao2019JCP)
        current_active_surface: int = self.cache.active_surface[0]
        dt: float = self.dt
        P_current: RealVector = P   
        v_dot_d = compute_v_dot_d(v, d)
        mass = state.get_mass()
        
        hop_flag, new_active_surface, P_new = complex_fssh_surface_hopping_py(
            dt, current_active_surface, P_current, rho_or_psi, evals, v_dot_d, d, mass
        )
        
        # update the cache
        self.cache.update_cache(H=H, evals=evals, evecs=evecs, dHdR=dHdR, nac=d, active_surface=new_active_surface)
        
        # return state. If the hopping is successful, the state is updated with the new active surface and momentum. Otherwise, the state is unchanged.
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
            dR, dP, drho_or_psi = self.derivative_diabatic(v, rho_or_psi, H, dHdR, self.cache.F_langevin, self.cache.active_surface)
        elif self.basis_representation == BasisRepresentation.ADIABATIC:
            dR, dP, drho_or_psi = self.derivative_adiabatic(v, rho_or_psi, H, dHdR, self.cache.F_langevin, self.hamiltonian._last_evecs, self.cache.active_surface)
        else:
            raise ValueError("Unsupported basis representation.")
        return state.from_unstructured(np.concatenate([dR, dP, drho_or_psi.flatten()], dtype=np.complex128))
    
    @staticmethod
    def derivative_diabatic(
        v: RealVector, 
        rho_or_psi: ComplexVector, 
        H: ComplexOperator, 
        dHdR: ComplexOperator, 
        F_langevin: ComplexVector, 
        active_surface: ActiveSurface
    ) -> Tuple[RealVector, RealVector, ComplexVector]:
        raise NotImplementedError(f"Derivative in diabatic basis is not implemented yet.")
    
    @staticmethod
    def derivative_adiabatic(
        v: RealVector, 
        rho_or_psi: ComplexVector, 
        H: ComplexOperator, 
        dHdR: ComplexOperator, 
        F_langevin: ComplexVector, 
        evecs: GenericOperator, 
        active_surface: ActiveSurface
    ) -> Tuple[RealVector, RealVector, ComplexVector]:
        # diagonalize the Hamiltonian
        evals, evecs = diagonalization(H, evecs)
        # evaluate the nonadiabatic couplings
        d, F, _ = evaluate_nonadiabatic_couplings(dHdR=dHdR, evals=evals, evecs=evecs)
        # compute the dot product of velocity and derivative coupling
        v_dot_d = compute_v_dot_d(v, d)
        
        # compute the F_mag term
        active_state = active_surface[0]        
        F_mag = evaluate_Fmag(v_dot_d=v_dot_d, d=d, active_surface=active_state)
        
        
        # The derivatives (Equations of motion)
        R_dot = v
        P_dot = F[active_state, ...] + F_langevin + F_mag
        rho_or_psi_dot = adiabatic_equations_of_motion(quantum_state=rho_or_psi, evals=evals, v_dot_d=v_dot_d)
        
        return R_dot, P_dot, rho_or_psi_dot
    
    @classmethod
    def initialize(
        cls,
        state: State,
        hamiltonian: HamiltonianBase,
        basis_representation: BasisRepresentation,
        dt: float,
    ) -> 'ComplexFSSH':
        R, P, rho_or_psi = state.get_variables()
        active_surface = initialize_active_surface(rho_or_psi)
        dim_hamiltonian = rho_or_psi.shape[0]
        dim_electronic = hamiltonian.dim
        dim_nuclear = R.shape[0]
        quantum_representation = QuantumRepresentation.WAVEFUNCTION if rho_or_psi.ndim == 1 else QuantumRepresentation.DENSITY_MATRIX
        
        H, dHdR = evaluate_hamiltonian(0.0, R, hamiltonian)
        evals, evecs = diagonalization(H, hamiltonian._last_evecs)
        d, F, _ = evaluate_nonadiabatic_couplings(dHdR=dHdR, evals=evals, evecs=evecs)
        
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