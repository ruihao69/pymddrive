import attr
import numpy as np
from attrs import define, field

from pymddrive.my_types import RealVector, ComplexVector, ComplexOperator, GenericOperator, GenericDiagonalVectorOperator, GenericVectorOperator
from pymddrive.dynamics.cache import Cache
from pymddrive.dynamics.options import BasisRepresentation, QuantumRepresentation
from pymddrive.dynamics.nonadiabatic_solvers.nonadiabatic_solver_base import NonadiabaticSolverBase, NonadiabaticProperties
from pymddrive.dynamics.nonadiabatic_solvers.math_utils import expected_value, diabatic_equations_of_motion, adiabatic_equations_of_motion, compute_v_dot_d
from pymddrive.dynamics.nonadiabatic_solvers.ehrenfest.ehrenfest_math_utils import mean_force_adiabatic_representation
from pymddrive.dynamics.nonadiabatic_solvers.ehrenfest.populations import compute_floquet_populations, compute_populations
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, QuasiFloquetHamiltonianBase, evaluate_hamiltonian, evaluate_nonadiabatic_couplings, diagonalization, adiabatic_to_diabatic, diabatic_to_adiabatic
from pymddrive.low_level.states import State


from typing import Tuple, Union



@define
class Ehrenfest(NonadiabaticSolverBase):
    # dim_quantum: int = field(on_setattr=attr.setters.frozen)
    dim_hamiltonian: int = field(on_setattr=attr.setters.frozen)
    dim_electronic: int = field(on_setattr=attr.setters.frozen)
    dim_nuclear: int = field(on_setattr=attr.setters.frozen)
    quantum_representation: QuantumRepresentation = field(on_setattr=attr.setters.frozen)
    basis_representation: BasisRepresentation = field(on_setattr=attr.setters.frozen)
    hamiltonian: HamiltonianBase = field(on_setattr=attr.setters.frozen)
    cache: Cache
    evecs_0: Union[None, GenericOperator] = field(default=None)

    # Implement the abstract methods from NonadiabaticSolverBase

    def callback(self, t: float, state: State) -> Tuple[State, bool]:
        """ Callback function for the Ehrenfest solver. """
        R, P, rho_or_psi = state.get_variables()
        H, dHdR = evaluate_hamiltonian(t, R, self.hamiltonian)
        evals, evecs = diagonalization(H, self.hamiltonian._last_evecs)
        if self.basis_representation == BasisRepresentation.ADIABATIC:
            # rho_or_psi_in_diabatic_basis = adiabatic_to_diabatic(rho_or_psi, self.hamiltonian._last_evecs)
            rho_or_psi_in_diabatic_basis = adiabatic_to_diabatic(rho_or_psi, self.cache.evecs)
            rho_or_psi = diabatic_to_adiabatic(rho_or_psi_in_diabatic_basis, evecs)
        self.hamiltonian.update_last_evecs(evecs)
        self.cache.update_cache(H=H, evals=evals, evecs=evecs, dHdR=dHdR)
        return state.from_unstructured(np.concatenate([R, P, rho_or_psi.flatten()], dtype=np.complex128)), True

    def derivative(self, t: float, state: State) -> State:
        R, P, rho = state.get_variables()
        H, dHdR = evaluate_hamiltonian(t, R, self.hamiltonian)
        v = state.get_v()
        if self.basis_representation == BasisRepresentation.DIABATIC:
            dR, dP, drho = self.derivative_diabatic(v, rho, H, dHdR, self.cache.F_langevin)
        elif self.basis_representation == BasisRepresentation.ADIABATIC:
            # dR, dP, drho = self.derivative_adiabatic(v, rho, H, dHdR, self.cache.F_langevin, self.hamiltonian._last_evecs)
            dR, dP, drho = self.derivative_adiabatic(v, rho, H, dHdR, self.cache.F_langevin, self.cache.evecs)
        else:
            raise ValueError("Unsupported basis representation.")
        return state.from_unstructured(np.concatenate([dR, dP, drho.flatten()], dtype=np.complex128))

    @classmethod
    def initialize(
        cls,
        state: State,
        hamiltonian: HamiltonianBase,
        basis_representation: BasisRepresentation
    ) -> 'Ehrenfest':
        R, P, rho_or_psi = state.get_variables()

        dim_nuclear = R.shape[0]
        dim_hamiltonian = rho_or_psi.shape[0]
        dim_electronic = hamiltonian.dim

        quantum_representation = QuantumRepresentation.WAVEFUNCTION if rho_or_psi.ndim > 1 else QuantumRepresentation.DENSITY_MATRIX

        H, dHdR = evaluate_hamiltonian(0.0, R, hamiltonian)
        # evals, evecs = diagonalization(H, hamiltonian._last_evecs)
        evals, evecs = diagonalization(H, prev_evecs=None)
        hamiltonian.update_last_evecs(evecs)
        d, F, _ = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)

        cache = Cache.from_dimensions(dim_elec=dim_hamiltonian, dim_nucl=dim_nuclear)
        cache.update_cache(H=H, evals=evals, evecs=evecs, dHdR=dHdR, nac=d)

        return cls(
            dim_nuclear=dim_nuclear,
            dim_hamiltonian=dim_hamiltonian,
            dim_electronic=dim_electronic,
            quantum_representation=quantum_representation,
            basis_representation=basis_representation,
            hamiltonian=hamiltonian,
            cache=cache
        )

    @staticmethod
    def derivative_diabatic(
        v: RealVector,
        rho_or_psi: Union[ComplexVector, ComplexOperator],
        H: GenericOperator,
        dHdR: GenericVectorOperator,
        F_langevin: RealVector,
    ) -> Tuple[RealVector, RealVector, Union[ComplexVector, ComplexOperator]]:
        R_dot = v
        P_dot = -expected_value(dHdR, rho_or_psi) + F_langevin
        rho_or_psi_dot = diabatic_equations_of_motion(rho_or_psi, H)
        return R_dot, P_dot, rho_or_psi_dot

    @staticmethod
    def derivative_adiabatic(
        v: RealVector,
        rho_or_psi: Union[ComplexVector, ComplexOperator],
        H: GenericOperator,
        dHdR: GenericVectorOperator,
        F_langevin: RealVector,
        last_evecs: GenericOperator
    ) -> Tuple[RealVector, RealVector, Union[ComplexVector, ComplexOperator]]:
        # diagonalize the Hamiltonian
        evals, evecs = diagonalization(H, last_evecs)
        # compute the nonadiabatic couplings
        d, F, _ = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        # evaluate the v_dot_d term
        v_dot_d = compute_v_dot_d(v, d)

        R_dot = v
        P_dot = mean_force_adiabatic_representation(F, evals, d, rho_or_psi) + F_langevin
        rho_or_psi_dot = adiabatic_equations_of_motion(rho_or_psi, evals, v_dot_d)
        return R_dot, P_dot, rho_or_psi_dot

    # Implement the Ehrenfest specific methods
    def calculate_properties(self, t: float, s: State) -> Tuple[float, float]:
        R, P, rho_or_psi = s.get_variables()

        KE = NonadiabaticSolverBase.calculate_KE(P, s.get_mass())
        if self.basis_representation == BasisRepresentation.ADIABATIC:
            PE = expected_value(self.cache.evals, rho_or_psi)
        else:
            PE = expected_value(self.cache.H, rho_or_psi)

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
                # evecs_F=self.hamiltonian._last_evecs,
                evecs_F=self.cache.evecs,
                evecs_0=evecs_0
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
                # evecs_F=self.hamiltonian._last_evecs,
                evecs_F=self.cache.evecs,
                evecs_0=evecs_0
            )
        else:
            adiabatic_populations = compute_populations(rho_or_psi, self.basis_representation, BasisRepresentation.ADIABATIC, self.hamiltonian._last_evecs)
            diabatic_populations = compute_populations(rho_or_psi, self.basis_representation, BasisRepresentation.DIABATIC, self.hamiltonian._last_evecs)

        return NonadiabaticProperties(R=R, P=P, adiabatic_populations=adiabatic_populations, diabatic_populations=diabatic_populations, KE=KE, PE=PE)

    def get_dim_nuclear(self) -> int:
        return self.dim_nuclear

    def get_dim_electronic(self) -> int:
        return self.dim_electronic

    def update_F_langevin(self, F_langevin: RealVector) -> None:
        np.copyto(self.cache.F_langevin, F_langevin)

