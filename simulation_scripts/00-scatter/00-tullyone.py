# %%
import numpy as np

from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes
from pymddrive.low_level.states import State
from pymddrive.integrators.state import get_state
from pymddrive.dynamics.options import BasisRepresentation, NonadiabaticDynamicsMethods, NumericalIntegrators  
from pymddrive.dynamics.get_dynamics import get_dynamics
from pymddrive.dynamics.run import run_ensemble

from tullyone_utils import outside_boundary, get_tully_one_p0_list
from scatter_postprocess import post_process

import os

def stop_condition(t, s):
    r, _, _ = s.get_variables()
    return outside_boundary(r, (-10, 10))

def run_tullyone(
    r0: float, 
    p0: float, 
    n_ensemble: int=100,
    mass: float=2000, 
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.FSSH,
    data_dir: str='./',
    filename: str="fssh.nc",
    basis_rep: BasisRepresentation = BasisRepresentation.ADIABATIC,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
    dt: float = 0.1,
) -> None:
    # get the initial time and state object
    t0 = 0.0
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = 1
    s0 = get_state(mass=mass, R=r0, P=p0, rho_or_psi=rho0)
    
    # intialize the model
    hamiltonian = get_tullyone()
    if basis_rep == BasisRepresentation.ADIABATIC:
        H = hamiltonian.H(t0, np.array([r0]))
        evals, evecs = np.linalg.eigh(H)
        rho0_adiabatic = evecs.T.conj() @ rho0 @ evecs
        s0 = get_state(mass=mass, R=r0, P=p0, rho_or_psi=rho0_adiabatic)
    
    # get the dynamics object
    dynamic_list = []
    for _ in range(n_ensemble):
        dyn = get_dynamics(t0=t0, s0=s0, dt=dt, hamiltonian=hamiltonian, dynamics_basis=basis_rep, method=solver)
        dynamic_list.append(dyn)
    
    # use run ensemble to run the dynamics
    momentum_signature = f"P0-{p0:.6f}"
    filename = os.path.join(data_dir, momentum_signature, filename)
    data_files_dir = os.path.dirname(filename)
    if not os.path.isdir(data_files_dir):
        os.makedirs(data_files_dir)
        
    run_ensemble(
        dynamics_list=dynamic_list,
        break_condition=stop_condition,
        filename=filename,
        numerical_integrator=integrator,
        save_every=10
    )
    
    post_process(data_files_dir)
    

def main(
    n_initial_momentum_samples: int=40,
    fssh_ensemble_size: int=10,
    sim_signature: str= "data_tullyone"
):
    r0 = -10.0
    p0_list = get_tully_one_p0_list(nsamples=n_initial_momentum_samples, pulse_type=TullyOnePulseTypes.NO_PULSE)
    
    # fssh_sim_signature = f'{sim_signature}_fssh'
    # run_tullyone(r0=r0, p0=p0_list[5], n_ensemble=fssh_ensemble_size, data_dir=fssh_sim_signature, filename='fssh.nc')
    # ehrenfest_sim_signature_a = f'{sim_signature}_ehrenfest_adiabatic'
    # run_tullyone(r0=r0, p0=p0_list[5], n_ensemble=1, solver=NonadiabaticDynamicsMethods.EHRENFEST, basis_rep=BasisRepresentation.ADIABATIC, filename="ehrenfest.nc", data_dir=ehrenfest_sim_signature_a)
    
    for p0 in p0_list:
        # FSSH simulations
        fssh_sim_signature = f'{sim_signature}_fssh'
        run_tullyone(r0=r0, p0=p0, n_ensemble=fssh_ensemble_size, data_dir=fssh_sim_signature, filename='fssh.nc')
    
        # Ehrenfest dynamics diabatic
        ehrenfest_sim_signature_d = f'{sim_signature}_ehrenfest_diabatic'
        run_tullyone(r0=r0, p0=p0, n_ensemble=1, solver=NonadiabaticDynamicsMethods.EHRENFEST, basis_rep=BasisRepresentation.DIABATIC, filename="ehrenfest.nc", data_dir=ehrenfest_sim_signature_d)
    
        # Ehrenfest dynamics adiabatic
        ehrenfest_sim_signature_a = f'{sim_signature}_ehrenfest_adiabatic'
        run_tullyone(r0=r0, p0=p0, n_ensemble=1, solver=NonadiabaticDynamicsMethods.EHRENFEST, basis_rep=BasisRepresentation.ADIABATIC, filename="ehrenfest.nc", data_dir=ehrenfest_sim_signature_a)
    

# %%

if __name__ == "__main__": 
    n_initial_k_samples = 40
    n_fssh_ensemble = 2000
    sim_signature = "data_tullyone"
    
    main(n_initial_momentum_samples=n_initial_k_samples, fssh_ensemble_size=n_fssh_ensemble, sim_signature=sim_signature)
# %%
