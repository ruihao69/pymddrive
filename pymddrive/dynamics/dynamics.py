# %%
import attr
from attrs import define, field

from pymddrive.my_types import GenericVector
from pymddrive.low_level.states import State
from pymddrive.dynamics.options import BasisRepresentation, BasisRepresentation
from pymddrive.dynamics.nonadiabatic_solvers import NonadiabaticSolverBase
from pymddrive.dynamics.langevin import LangevinBase, NullLangevin

@define
class Dynamics:
    t0: float = field(on_setattr=attr.setters.frozen)
    s0: State = field(on_setattr=attr.setters.frozen)
    dt: float = field(on_setattr=attr.setters.frozen)
    solver: NonadiabaticSolverBase = field(on_setattr=attr.setters.frozen)
    langevin: LangevinBase = field(default=NullLangevin(), on_setattr=attr.setters.frozen)
    atol: float = 1e-8
    rtol: float = 1e-8
    
    def deriv_wrapper(self, t: float, y: GenericVector) -> GenericVector:
        s = self.s0.from_unstructured(y)
        return self.solver.derivative(t, s).flatten()
        
    
def main():
    import numpy as np
    from pymddrive.integrators.state import get_state
    from pymddrive.models.tullyone import get_tullyone
    from pymddrive.dynamics.nonadiabatic_solvers import Ehrenfest
    
    t0 = 0.0 
    
    n_particle = 1
    R = np.random.normal(0, 1, n_particle)
    P = np.random.normal(0, 1, n_particle)
    rho_dummy = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)
    mass = 1.0
    
    s0 = get_state(mass, R, P, rho_dummy)
    dt = 0.01
    hamiltonian = get_tullyone()
    solver = Ehrenfest.initialize(
        state=s0,
        hamiltonian=hamiltonian,
        basis_representation=BasisRepresentation.DIABATIC, 
    )
    
    dyn = Dynamics(t0=t0, s0=s0, solver=solver, dt=dt)
    print(dyn)

# %%
if __name__ == "__main__":
    main()
# %%
