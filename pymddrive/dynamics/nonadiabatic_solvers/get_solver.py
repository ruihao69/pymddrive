import numpy as np

from pymddrive.low_level.states import State
from pymddrive.integrators.state import get_state
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.dynamics.options import NonadiabaticDynamicsMethods, BasisRepresentation
from pymddrive.dynamics.nonadiabatic_solvers.nonadiabatic_solver_base import NonadiabaticSolverBase
from pymddrive.dynamics.nonadiabatic_solvers.ehrenfest import Ehrenfest
from pymddrive.dynamics.nonadiabatic_solvers.fssh import FSSH
from pymddrive.dynamics.nonadiabatic_solvers.afssh import AFSSH

from copy import deepcopy

def get_solver(
    s0: State,    
    hamiltonian: HamiltonianBase,
    method: NonadiabaticDynamicsMethods,
    dynamics_basis: BasisRepresentation,
    dt: float
) -> NonadiabaticSolverBase:
    # copy the s0
    R, P, rho_or_psi = s0.get_variables()
    s0_local = get_state(mass=s0.get_mass(), R=R, P=P, rho_or_psi=rho_or_psi)
    
    # copy the hamiltonian
    hamiltonian_local = deepcopy(hamiltonian)
    H = hamiltonian_local.H(0.0, R)
    _, evecs = np.linalg.eigh(H)
    hamiltonian_local.update_last_evecs(evecs)
    
    # return the solver based on method
    if method == NonadiabaticDynamicsMethods.EHRENFEST:
        return Ehrenfest.initialize(
            state=s0_local,
            hamiltonian=hamiltonian_local,
            basis_representation=dynamics_basis,
        )
    elif method == NonadiabaticDynamicsMethods.FSSH:
        return FSSH.initialize(
            state=s0_local,
            hamiltonian=hamiltonian_local,
            basis_representation=dynamics_basis,
            dt=dt
        )
    elif method == NonadiabaticDynamicsMethods.AFSSH:
        return AFSSH.initialize(
            state=s0_local,
            hamiltonian=hamiltonian_local,
            basis_representation=dynamics_basis,
            dt=dt
        )
    else:
        raise NotImplementedError(f"Nonadiabataic dynamics method '{method.name}' has not been implemented yet.")