import numpy as np
from attrs import define

from pymddrive.my_types import RealVector, ComplexVector, ComplexOperator, GenericOperator, GenericDiagonalVectorOperator, GenericVectorOperator
from pymddrive.dynamics.cache import Cache
from pymddrive.dynamics.options import BasisRepresentation, QuantumRepresentation
from pymddrive.dynamics.nonadiabatic_solvers.nonadiabatic_solver_base import NonadiabaticSolverBase
from pymddrive.dynamics.nonadiabatic_solvers.math_utils import expected_value, diabatic_equations_of_motion, adiabatic_equations_of_motion, compute_v_dot_d
from pymddrive.dynamics.nonadiabatic_solvers.ehrenfest.ehrenfest_math_utils import mean_force_adiabatic_representation


from typing import Tuple, Union

from typing import TypeVar
HamiltonianBase = TypeVar('HamiltonianBase')    
State = TypeVar('State')



@define(frozen=True, slots=True)
class Ehrenfest(NonadiabaticSolverBase):
    dim_quantum: int
    dim_nuclear: int
    quantum_representation: QuantumRepresentation
    basis_representation: BasisRepresentation
    hamiltonian: HamiltonianBase
    cache: Cache
    
    # Implement the abstract methods from NonadiabaticSolverBase
    
    def callback(self, ) -> None:
        pass 
    
    def derivative(self, t: float, state: State) -> State:
        R, P, rho = state.variables()
        H, dHdR = evaluate_hamiltonian()
        v = state.get_v()
        if self.basis_representation == BasisRepresentation.DIABATIC:
            self.derivative_diabatic(v, rho, H, dHdR)
        
    @classmethod
    def initialize(cls, R: RealVector, P: RealVector, rho_or_psi: ComplexVector, basis_representation: BasisRepresentation) -> 'Ehrenfest':
        dim_nuclear = R.shape[0]
        dim_quantum = rho_or_psi.shape[0]
        quantum_representation = QuantumRepresentation.WAVEFUNCTION if dim_quantum > 1 else QuantumRepresentation.DENSITY_MATRIX
        
        pass 
    
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
        F: GenericDiagonalVectorOperator,
        evals: RealVector,
        v_dot_d: GenericOperator
    ) -> Tuple[RealVector, RealVector, Union[ComplexVector, ComplexOperator]]:
        R_dot = v
        P_dot = mean_force_adiabatic_representation(F, evals, v_dot_d, rho_or_psi)
        rho_or_psi_dot = adiabatic_equations_of_motion(rho_or_psi, evals, v_dot_d)
        return R_dot, P_dot, rho_or_psi_dot

