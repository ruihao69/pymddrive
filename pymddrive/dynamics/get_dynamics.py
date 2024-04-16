from pymddrive.low_level.states import State
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.dynamics.dynamics import Dynamics
from pymddrive.dynamics.options import BasisRepresentation, NonadiabaticDynamicsMethods
from pymddrive.dynamics.nonadiabatic_solvers.get_solver import get_solver
from pymddrive.dynamics.langevin import Langevin

def get_dynamics(
    t0: float,
    s0: State,
    dt: float,
    hamiltonian: HamiltonianBase,
    dynamics_basis: BasisRepresentation,
    method: NonadiabaticDynamicsMethods,
) -> Dynamics:
    # get the non-adiabatic solver
    solver = get_solver(s0=s0, hamiltonian=hamiltonian, method=method, dynamics_basis=dynamics_basis, dt=dt)
    
    # get langevin if there is langevin
    if (hamiltonian.get_friction() is not None) and (hamiltonian.get_kT() is not None):
        langevin = Langevin(
            kT=hamiltonian.get_kT(),
            mass=s0.get_mass(),
            gamma=hamiltonian.get_friction()
        )
        return Dynamics(t0=t0, s0=s0, dt=dt, solver=solver, langevin=langevin)
    else:
        return Dynamics(t0=t0, s0=s0, dt=dt, solver=solver,)
            
        