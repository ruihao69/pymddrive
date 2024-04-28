# %%
import numpy as np

from pymddrive.my_types import RealVector
# from pymddrive.models.landry_spin_boson import LandrySpinBoson
from pymddrive.models.landry_spin_boson import get_landry_spin_boson
from pymddrive.integrators.state import get_state
from pymddrive.dynamics.options import BasisRepresentation, NonadiabaticDynamicsMethods, NumericalIntegrators    
from pymddrive.dynamics.get_dynamics import get_dynamics
from pymddrive.dynamics.run import run_ensemble

from pymddrive.dynamics.misc_utils import eval_nonadiabatic_hamiltonian
from spin_boson_postprocess import post_process

import os 
from typing import Tuple

def stop_condition(t, s):
    return t > 100000

def break_condition(t, s):
    return False

def sample_lsb_boltzmann(
    n_samples: int, 
    lsb_hamiltonian,
    initialize_from_donor: bool =True
):
    # constants
    kT: float = lsb_hamiltonian.kT
    mass: float = lsb_hamiltonian.M
    beta: float = 1.0 / kT
    Omega_nuclear: float = lsb_hamiltonian.Omega_nuclear
    
    # sample the momentum
    sigma_momentum = np.sqrt(mass / beta)
    momentum_samples = np.random.normal(0, sigma_momentum, n_samples)
    
    R0 = lsb_hamiltonian.get_donor_R() if initialize_from_donor else lsb_hamiltonian.get_acceptor_R()
    sigma_R = 1.0 / np.sqrt(beta * mass) / Omega_nuclear
    position_samples = np.random.normal(R0, sigma_R, n_samples)
    return position_samples, momentum_samples

def run_landry_spin_boson(
    R0: RealVector,
    P0: RealVector,
    n_ensemble: int=100,
    mass: float=1.0,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.FSSH,
    data_dir: str='./',
    filename: str="fssh.nc",
    basis_rep: BasisRepresentation = BasisRepresentation.ADIABATIC,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
    dt: float = 0.1,
) -> None:
    # get the initial time and state object
    def get_initial_rho():
        rho0 = np.zeros((2, 2), dtype=np.complex128)
        rho0[0, 0] = 1
        return rho0
    
    t0 = 0.0
    rho0 = [get_initial_rho() for _ in range(n_ensemble)]
    
    # intialize the model
    hamiltonian = get_landry_spin_boson(
        param_set='SymmetricDoubleWell'
    ) 
    if basis_rep == BasisRepresentation.ADIABATIC:
        for ii in range(n_ensemble):
            H = hamiltonian.H(t0, np.array([R0[ii]]))
            evals, evecs = np.linalg.eigh(H)
            rho0_adiabatic = evecs.T.conj() @ rho0[ii] @ evecs
            rho0[ii] = rho0_adiabatic
    
    s0_list = [get_state(mass=mass, R=R0[ii], P=P0[ii], rho_or_psi=rho0[ii]) for ii in range(n_ensemble)]
    
    # get the dynamics object
    dynamics_list = []
    for ii in range(n_ensemble):
        dyn = get_dynamics(t0=t0, s0=s0_list[ii], dt=dt, hamiltonian=hamiltonian, dynamics_basis=basis_rep, method=solver)
        dynamics_list.append(dyn)
    
    filename = os.path.join(data_dir, filename)    
    data_files_dir = os.path.dirname(filename)
    if not os.path.isdir(data_files_dir):
        os.makedirs(data_files_dir)
    
    run_ensemble(
        dynamics_list=dynamics_list,
        break_condition=stop_condition,
        filename=filename,
        numerical_integrator=integrator,
        save_every=10,
    )
    
    post_process(data_files_dir)


def main(
    project_prefix: str,
    ntrajs: int, 
    basis_rep: BasisRepresentation = BasisRepresentation.DIABATIC, 
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST, 
    numerical_integrator: NumericalIntegrators = NumericalIntegrators.ZVODE
):
    _hamiltonian = get_landry_spin_boson(
        param_set='SymmetricDoubleWell'
    )
    
    position_samples, momentum_samples = sample_lsb_boltzmann(
        n_samples=ntrajs, lsb_hamiltonian=_hamiltonian, initialize_from_donor=True
    )
    if solver == NonadiabaticDynamicsMethods.FSSH:
        filename = "fssh.nc"
    elif solver == NonadiabaticDynamicsMethods.EHRENFEST:
        filename = "ehrenfest.nc"
    else:
        raise ValueError(f"Unknown solver {solver}")
    
    run_landry_spin_boson(
        R0=position_samples, P0=momentum_samples, n_ensemble=ntrajs, basis_rep=basis_rep, solver=solver, integrator=numerical_integrator, data_dir=project_prefix, filename=filename,
        dt=1.25
    )
    


# %%
def _test_sampling(T: float = 300):
    import matplotlib.pyplot as plt
    hamiltonian = get_landry_spin_boson(param_set='SymmetricDoubleWell')
    position_samples, momentum_samples = sample_lsb_boltzmann(3000, lsb_hamiltonian=hamiltonian)
    R0 = 0.5 * (hamiltonian.get_donor_R() + hamiltonian.get_acceptor_R())
    L = 15
    R = np.linspace(R0-L, R0+L)
    V = np.zeros((hamiltonian.dim, len(R)))
    for ii, rr in enumerate(R):
        H = hamiltonian.H(0, rr)
        for jj in range(hamiltonian.dim):
            V[jj, ii] = H[jj, jj]
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for jj in range(hamiltonian.dim):
        ax.plot(R, V[jj, :], label=f'V{jj+1}')
    print(position_samples)   
    hist, bins = np.histogram(position_samples, bins=100, density=True)
    r = (bins[1:] + bins[:-1]) / 2 
    ax_twin = ax.twinx()
    ax_twin.plot(r, hist, label='position samples') 
    ax_twin.fill_between(r, hist, alpha=0.5)
    ax.set_xlabel('R')
    ax.set_ylabel('Diabatic PES')
    ax_twin.set_ylabel('Initial position distribution')
    plt.show()
    
    fig  = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.hist(momentum_samples, bins=100, density=True)
    ax.set_xlabel('P')
    ax.set_ylabel('Initial momentum distribution')
    plt.show()
    
# %%
if __name__ == "__main__":
    ntraj = 128
    # numerical_integrator = NumericalIntegrators.ZVODE
    numerical_integrator = NumericalIntegrators.RK4
    project_prefix = "data_ehrenfset_diabatic"
    main(project_prefix=project_prefix, ntrajs=ntraj, basis_rep=BasisRepresentation.DIABATIC, numerical_integrator=numerical_integrator)
    
    project_prefix = "data_ehrenfset_adiabatic" 
    main(project_prefix=project_prefix, ntrajs=ntraj, basis_rep=BasisRepresentation.ADIABATIC, numerical_integrator=numerical_integrator)
    
    project_prefix = "data_fssh_diabatic"
    main(project_prefix=project_prefix, ntrajs=ntraj, basis_rep=BasisRepresentation.ADIABATIC, solver=NonadiabaticDynamicsMethods.FSSH, numerical_integrator=numerical_integrator)
    
    

# %%
if __name__ == "__main__":
    data_ehrenfest_diabatic = np.loadtxt("data_ehrenfset_diabatic/traj.dat")
    data_ehrenfest_adiabatic = np.loadtxt("data_ehrenfset_adiabatic/traj.dat")
    data_fssh_diabatic = np.loadtxt("data_fssh_diabatic/traj.dat")
    
    from plotter import SpinBosonPlotter
    sbp = SpinBosonPlotter()
    
    sbp.plot_all(dim=2, traj_data=data_ehrenfest_diabatic, label_base="Ehrenfest Diabatic")
    sbp.plot_all(dim=2, traj_data=data_ehrenfest_adiabatic, label_base="Ehrenfest Adiabatic")
    sbp.plot_all(dim=2, traj_data=data_fssh_diabatic, label_base="FSSH Diabatic")
    
    sbp.finalize()
    
    
    

# %%
