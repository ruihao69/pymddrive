import attr
import numpy as np
from attrs import define, field

from pymddrive.my_types import RealVector, ComplexVector, ComplexOperator, GenericOperator, GenericDiagonalVectorOperator, GenericVectorOperator
from pymddrive.dynamics.cache import Cache
from pymddrive.dynamics.options import BasisRepresentation, QuantumRepresentation
from pymddrive.dynamics.nonadiabatic_solvers.nonadiabatic_solver_base import NonadiabaticSolverBase
from pymddrive.dynamics.nonadiabatic_solvers.math_utils import expected_value, diabatic_equations_of_motion, adiabatic_equations_of_motion, compute_v_dot_d
from pymddrive.dynamics.nonadiabatic_solvers.ehrenfest.ehrenfest_math_utils import mean_force_adiabatic_representation
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, evaluate_hamiltonian, evaluate_nonadiabatic_couplings, diagonalization
from pymddrive.low_level.states import State


from typing import Tuple, Union



@define
class Ehrenfest(NonadiabaticSolverBase):
    dim_quantum: int = field(on_setattr=attr.setters.frozen)
    dim_nuclear: int = field(on_setattr=attr.setters.frozen)
    quantum_representation: QuantumRepresentation = field(on_setattr=attr.setters.frozen)
    basis_representation: BasisRepresentation = field(on_setattr=attr.setters.frozen)
    hamiltonian: HamiltonianBase = field(on_setattr=attr.setters.frozen)
    cache: Cache
    
    # Implement the abstract methods from NonadiabaticSolverBase
    
    def callback(self, ) -> None:
        pass 
    
    def derivative(self, t: float, state: State) -> State:
        R, P, rho = state.variables()
        H, dHdR = evaluate_hamiltonian(t, R, self.hamiltonian)
        v = state.get_v()
        if self.basis_representation == BasisRepresentation.DIABATIC:
            self.derivative_diabatic(v, rho, H, dHdR)
        
    @classmethod
    def initialize(
        cls, 
        state: State,
        hamiltonian: HamiltonianBase,
        basis_representation: BasisRepresentation
        ) -> 'Ehrenfest':
        R, P, rho_or_psi = state.variables()
        
        dim_nuclear = R.shape[0]
        dim_quantum = rho_or_psi.shape[0]
        
        quantum_representation = QuantumRepresentation.WAVEFUNCTION if dim_quantum > 1 else QuantumRepresentation.DENSITY_MATRIX
        
        H, dHdR = evaluate_hamiltonian(0.0, R, hamiltonian)
        evals, evecs = diagonalization(H)
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        
        cache = Cache(
            H=H,
            evals=evals,
            evecs=evecs,
            dHdR=dHdR,
            nac=d
        )
        
        return cls(
            dim_nuclear=dim_nuclear,
            dim_quantum=dim_quantum,
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
    ) -> Tuple[RealVector, RealVector, Union[ComplexVector, ComplexOperator]]:
        R_dot = v
        P_dot = -expected_value(dHdR, rho_or_psi)
        rho_or_psi_dot = diabatic_equations_of_motion(rho_or_psi, H)
        return R_dot, P_dot, rho_or_psi_dot
    
    @staticmethod 
    def derivative_adiabatic(
        v: RealVector,
        rho_or_psi: Union[ComplexVector, ComplexOperator],
        H: GenericOperator,
        dHdR: GenericVectorOperator,
    ) -> Tuple[RealVector, RealVector, Union[ComplexVector, ComplexOperator]]:
        # diagonalize the Hamiltonian
        evals, evecs = diagonalization(H)
        # compute the nonadiabatic couplings
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        # evaluate the v_dot_d term
        v_dot_d = compute_v_dot_d(v, d)
        
        R_dot = v
        P_dot = mean_force_adiabatic_representation(F, evals, v_dot_d, rho_or_psi)
        rho_or_psi_dot = adiabatic_equations_of_motion(rho_or_psi, evals, v_dot_d)
        return R_dot, P_dot, rho_or_psi_dot
    
