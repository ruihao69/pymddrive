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
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, QuasiFloquetHamiltonianBase, evaluate_hamiltonian, evaluate_nonadiabatic_couplings, diagonalization
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
    
    def callback(self, t: float, state: State) -> Tuple[State, bool]:
        """ Callback function for the Ehrenfest solver. """
        R, P, rho = state.get_variables()
        H, dHdR = evaluate_hamiltonian(t, R, self.hamiltonian)
        evals, evecs = diagonalization(H)
        self.hamiltonian.update_last_evecs(evecs)
        self.cache.update_cache(H=H, evals=evals, evecs=evecs, dHdR=dHdR)
        return state, False
    
    def derivative(self, t: float, state: State) -> State:
        R, P, rho = state.get_variables()
        H, dHdR = evaluate_hamiltonian(t, R, self.hamiltonian)
        v = state.get_v()
        if self.basis_representation == BasisRepresentation.DIABATIC:
            dR, dP, drho = self.derivative_diabatic(v, rho, H, dHdR, self.cache.F_langevin)
        elif self.basis_representation == BasisRepresentation.ADIABATIC:
            dR, dP, drho = self.derivative_adiabatic(v, rho, H, dHdR, self.cache.F_langevin)
        else:
            raise ValueError("Unsupported basis representation.")
        return state.from_unstructured(np.concatenate([dR, dP, drho.flatten(order='F')], dtype=np.complex128))
        
    @classmethod
    def initialize(
        cls, 
        state: State,
        hamiltonian: HamiltonianBase,
        basis_representation: BasisRepresentation
        ) -> 'Ehrenfest':
        R, P, rho_or_psi = state.get_variables()
        
        dim_nuclear = R.shape[0]
        dim_quantum = rho_or_psi.shape[0]
        
        quantum_representation = QuantumRepresentation.WAVEFUNCTION if dim_quantum > 1 else QuantumRepresentation.DENSITY_MATRIX
        
        H, dHdR = evaluate_hamiltonian(0.0, R, hamiltonian)
        evals, evecs = diagonalization(H, hamiltonian._last_deriv_couplings)
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        
        cache = Cache.from_dimensions(dim_elec=dim_quantum, dim_nucl=dim_nuclear)
        cache.update_cache(H=H, evals=evals, evecs=evecs, dHdR=dHdR, nac=d)
        
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
    ) -> Tuple[RealVector, RealVector, Union[ComplexVector, ComplexOperator]]:
        # diagonalize the Hamiltonian
        evals, evecs = diagonalization(H)
        # compute the nonadiabatic couplings
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        # evaluate the v_dot_d term
        v_dot_d = compute_v_dot_d(v, d)
        
        R_dot = v
        P_dot = mean_force_adiabatic_representation(F, evals, v_dot_d, rho_or_psi) + F_langevin
        rho_or_psi_dot = adiabatic_equations_of_motion(rho_or_psi, evals, v_dot_d)
        return R_dot, P_dot, rho_or_psi_dot
    
    # Implement the Ehrenfest specific methods
    def calculate_properties(self, s: State) -> Tuple[float, float]:
        R, P, rho_or_psi = s.get_variables()
        
        KE = NonadiabaticSolverBase.calculate_KE(P, s.get_mass())
        if self.basis_representation == BasisRepresentation.ADIABATIC:
            PE = expected_value(self.cache.evals, rho_or_psi)
        else:
            PE = expected_value(self.cache.H, rho_or_psi)
        
        if isinstance(self.hamiltonian, QuasiFloquetHamiltonianBase):
            raise NotImplementedError
        else:
            adiabatic_populations = compute_populations(rho_or_psi, self.basis_representation, BasisRepresentation.ADIABATIC, self.cache.evecs)
            # diabatic_populations = compute_populations(rho_or_psi, self.basis_representation, BasisRepresentation.DIABATIC, self.cache.evecs)
            diabatic_populations = rho_or_psi.diagonal().real
            
        return NonadiabaticProperties(R=R, P=P, adiabatic_populations=adiabatic_populations, diabatic_populations=diabatic_populations, KE=KE, PE=PE) 
            
    
