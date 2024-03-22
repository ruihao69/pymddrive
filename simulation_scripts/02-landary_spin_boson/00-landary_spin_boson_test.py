# %%
import numpy as np
from numpy.typing import ArrayLike

from pymddrive.models.landry_spin_boson import LandrySpinBoson
from pymddrive.integrators.state import State
from pymddrive.dynamics.options import BasisRepresentation, QunatumRepresentation, NonadiabaticDynamicsMethods, NumericalIntegrators    
from pymddrive.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics, run_nonadiabatic_dynamics_ensembles
from pymddrive.dynamics.misc_utils import eval_nonadiabatic_hamiltonian

import os 
from typing import Tuple

def stop_condition(t, s, states):
    return t > 3000

def break_condition(t, s, states):
    return False

def sample_lsb_boltzmann(
    n_samples: int, 
    lsb_hamiltonian: LandrySpinBoson,
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
    
def run_one_lsb(
    R0: float,
    P0: float,
    qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
    basis_rep: BasisRepresentation = BasisRepresentation.Diabatic,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
):
    from pymddrive.dynamics.misc_utils import eval_nonadiabatic_hamiltonian
    hamiltonian = LandrySpinBoson()
    dim = hamiltonian.dim
    rho0_diabatic = np.zeros((dim, dim), dtype=np.complex128)
    rho0_diabatic[0, 0] = 1.0
    hami_return = eval_nonadiabatic_hamiltonian(0, np.array([R0]), hamiltonian, basis_rep=BasisRepresentation.Diabatic)
    evecs = hami_return.evecs
    rho0_adiabatic = evecs.T.conj() @ rho0_diabatic @ evecs
    if basis_rep == BasisRepresentation.Adiabatic:
        s0 = State.from_variables(R=R0, P=P0, rho=rho0_adiabatic)
    elif basis_rep == BasisRepresentation.Diabatic:
        s0 = State.from_variables(R=R0, P=P0, rho=rho0_diabatic)
        
    # dt_max = 0.003
    mass = hamiltonian.M
    
    dyn = NonadiabaticDynamics(
        hamiltonian=hamiltonian,
        t0=0.0,
        s0=s0,
        mass=mass,
        basis_rep=basis_rep,
        qm_rep=qm_rep,
        solver=solver,
        numerical_integrator=integrator,
        dt=0.03,
        save_every=30,
        # max_step=dt_max,
    )
    
    return run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)

def generate_ensembles(
    initial_diabatic_states: int,
    R0_samples: ArrayLike,
    P0_samples: ArrayLike,
    qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
    basis_rep: BasisRepresentation = BasisRepresentation.Diabatic,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
) -> Tuple[NonadiabaticDynamics]:
    # initialize the electronic states
    assert (initial_diabatic_states >= 0) and (initial_diabatic_states < 4), f"Valid states should be between 0 and 3. Got {initial_diabatic_states}."
    
    assert (n_samples := len(R0_samples)) == len(P0_samples), "The number of R0 and P0 samples should be the same."
    ensemble = ()
    for ii, (R0, P0) in enumerate(zip(R0_samples, P0_samples)):
        hamiltonian = LandrySpinBoson() 
        mass = hamiltonian.M
        dim: int = hamiltonian.dim
        rho0_diabatic = np.zeros((dim, dim), dtype=np.complex128)
        rho0_diabatic[initial_diabatic_states, initial_diabatic_states] = 1.0
        if basis_rep == BasisRepresentation.Diabatic:
            s0 = State.from_variables(R=R0, P=P0, rho=rho0_diabatic)
        else:
            hami_return = eval_nonadiabatic_hamiltonian(0, np.array([R0]), hamiltonian, basis_rep=BasisRepresentation.Diabatic)
            evecs = hami_return.evecs
            rho0_adiabatic = evecs.T.conj() @ rho0_diabatic @ evecs
            s0 = State.from_variables(R=R0, P=P0, rho=rho0_adiabatic)
        dyn = NonadiabaticDynamics(
            hamiltonian=hamiltonian,
            t0=0.0,
            s0=s0,
            mass=mass,
            basis_rep=basis_rep,
            qm_rep=qm_rep,
            solver=solver,
            numerical_integrator=integrator,
            dt=0.3,
            save_every=10,
        )
        ensemble += (dyn,)
    return ensemble
        
   

def main(ntrajs: int, basis_rep: BasisRepresentation = BasisRepresentation.Diabatic, solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST, numerical_integrator: NumericalIntegrators = NumericalIntegrators.ZVODE):
    _hamiltonian = LandrySpinBoson()
    position_samples, momentum_samples = sample_lsb_boltzmann(
        n_samples=ntrajs, lsb_hamiltonian=_hamiltonian, initialize_from_donor=True
    )
    
    dyn_ensemble = generate_ensembles(
        0, position_samples, momentum_samples,
        qm_rep=QunatumRepresentation.DensityMatrix,
        basis_rep=basis_rep,
        # solver=NonadiabaticDynamicsMethods.EHRENFEST,
        solver=solver,
        integrator=numerical_integrator,
    )
    
    output_ensemble_averaged = run_nonadiabatic_dynamics_ensembles(dyn_ensemble, stop_condition, break_condition)
    plot_ensembles_averaged(output_ensemble_averaged)
    return output_ensemble_averaged



# %%
def _plot_adiabatic_diabatic(output_adiabatic, output_diabatic, output_fssh):
    import matplotlib.pyplot as plt
    ta, td, tfssh = output_adiabatic['time'], output_diabatic['time'], output_fssh['time']
    Ra, Rd, Rfssh = output_adiabatic['states']['R'], output_diabatic['states']['R'], output_fssh['states']['R']
    Pa, Pd, Pfssh = output_adiabatic['states']['P'], output_diabatic['states']['P'], output_fssh['states']['P']
    popa_diab, popd_diab, popfssh_diab = output_adiabatic['diab_populations'], output_diabatic['diab_populations'], output_fssh['diab_populations']
    popa_adiab, popd_adiab, popfssh_adiab = output_adiabatic['adiab_populations'], output_diabatic['adiab_populations'], output_fssh['adiab_populations']
    KE_a, KE_d, KE_fssh = output_adiabatic['KE'], output_diabatic['KE'], output_fssh['KE']
    PE_a, PE_d, PE_fssh = output_adiabatic['PE'], output_diabatic['PE'], output_fssh['PE']
    # rhoa, rhod = output_adiabatic['states']['rho'], output_diabatic['states']['rho']
    # popa, popd = output_adiabatic['populations'], output_diabatic['populations']
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(ta, Ra, ls='-', label='R adiabatic')
    ax.plot(td, Rd, ls='-.', label='R diabatic')
    ax.plot(tfssh, Rfssh, ls='--', label='R fssh')
    ax.legend()
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('R')
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(ta, Pa, ls='-', label='P adiabatic')
    ax.plot(td, Pd, ls='-.', label='P diabatic')
    ax.plot(tfssh, Pfssh, ls='--', label='P fssh')
    ax.legend()
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('P')
    
    fig = plt.figure(dpi=300,)
    gs = fig.add_gridspec(2, 1)
    axs = gs.subplots(sharex=True).flatten()
    for ii in range(popa_diab.shape[1]):
        ax = axs[ii]
        ax.plot(ta, popa_diab[:, ii], ls='-', label=f'pop{ii+1} adiabatic')
        ax.plot(td, popd_diab[:, ii], ls='-.', label=f'pop{ii+1} diabatic')
        ax.plot(tfssh, popfssh_diab[:, ii], ls='--', label=f'pop{ii+1} fssh')
    ax = axs[0]
    dat = np.loadtxt("exact.dat", delimiter=',') 
    timext, pop1 = dat.T
    ax.plot(timext, pop1, label='pop1 exact', ls=':')
    ax.set_xlim(0, ta[-1])
    ax.legend()
    
    fig = plt.figure(dpi=300,)
    gs = fig.add_gridspec(2, 1)
    axs = gs.subplots(sharex=True).flatten()
    for ii in range(popa_diab.shape[1]):
        ax = axs[ii]
        ax.plot(ta, popa_adiab[:, ii], ls='-', label=f'pop{ii+1} adiabatic')
        ax.plot(td, popd_adiab[:, ii], ls='-.', label=f'pop{ii+1} diabatic')
        ax.plot(tfssh, popfssh_adiab[:, ii], ls='--', label=f'pop{ii+1} fssh') 
    ax.legend()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(ta, KE_a, ls='-', label='KE adiabatic')
    ax.plot(td, KE_d, ls='-.', label='KE diabatic')
    ax.plot(tfssh, KE_fssh, ls='--', label='KE fssh')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Kinetic Energy')
    ax.legend()
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(ta, PE_a, ls='-', label='PE adiabatic')
    ax.plot(td, PE_d, ls='-.', label='PE diabatic')
    ax.plot(tfssh, PE_fssh, ls='--', label='PE fssh')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Potential Energy')
    ax.legend()
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(ta, PE_a+KE_a, ls='-', label='TE adiabatic')
    ax.plot(td, PE_d+KE_d, ls='-.', label='TE diabatic')
    ax.plot(tfssh, PE_fssh+KE_fssh, ls='--', label='TE fssh')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Total Energy')
    ax.legend()
    plt.show()
    
    
def plot_ensembles_averaged(output_ensemble_averaged):
    import matplotlib.pyplot as plt
    print(output_ensemble_averaged.keys())
    
    time = output_ensemble_averaged['time'] 
    R, P = output_ensemble_averaged['states']['R'], output_ensemble_averaged['states']['P']
    pop_diab = output_ensemble_averaged['diab_populations']
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time, R, ls='-', label='R')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('R')
    ax.legend()
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time, P, ls='-', label='P')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('P')
    ax.legend()
    plt.show()
    
    fig =  plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for ii in range(pop_diab.shape[1]):
        ax.plot(time, pop_diab[:, ii], label=f'pop{ii+1}')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Diabatic Population')
    ax.legend()
    plt.show()
    

def _test_sampling(T: float = 300):
    import matplotlib.pyplot as plt
    hamiltonian = LandrySpinBoson()
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
    _test_sampling()
    # output_fssh = run_one_four_level_morse(
    #     0.3, -4.0, basis_rep=BasisRepresentation.Adiabatic, solver=NonadiabaticDynamicsMethods.FSSH
    # )
    # output_adiabatic, output_diabatic = _compare_adiabatic_diabatic(0.3, -4.0)
    
    # _plot_adiabatic_diabatic(output_adiabatic, output_diabatic, output_fssh)
# %%
if __name__ == "__main__":
    ntraj = 48
    numerical_integrator = NumericalIntegrators.ZVODE
    output_diabatic = main(ntrajs=ntraj, basis_rep=BasisRepresentation.Diabatic, numerical_integrator=numerical_integrator)
    output_adiabatic = main(ntrajs=ntraj, basis_rep=BasisRepresentation.Adiabatic, numerical_integrator=numerical_integrator)
    output_fssh = main(ntrajs=ntraj, basis_rep=BasisRepresentation.Adiabatic, solver=NonadiabaticDynamicsMethods.FSSH, numerical_integrator=numerical_integrator)
    # _hamiltonian = LandrySpinBoson()
    # position_samples, momentum_samples = sample_lsb_boltzmann(
    #     n_samples=1, lsb_hamiltonian=_hamiltonian, initialize_from_donor=True
    # )
    # output_fssh = run_one_lsb(position_samples[0], momentum_samples[0], basis_rep=BasisRepresentation.Adiabatic, solver=NonadiabaticDynamicsMethods.FSSH)
    # # _plot_adiabatic_diabatic(output_adiabatic, output_diabatic, output_fssh)
    # import matplotlib.pyplot as plt 
    
    # # P
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # ax.plot(output_fssh['time'], output_fssh['states']['P'], ls='-', label='R fssh')
    
    # # KE
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # ax.plot(output_fssh['time'], output_fssh['KE'], ls='-', label='R fssh')
    # # PE 
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # ax.plot(output_fssh['time'], output_fssh['PE'], ls='-', label='R fssh')
    # # TE 
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # ax.plot(output_fssh['time'], output_fssh['KE']+output_fssh['PE'], ls='-', label='R fssh')
    # # pop
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # for ii in range(output_fssh['adiab_populations'].shape[1]):
    #     ax.plot(output_fssh['time'], output_fssh['adiab_populations'][:, ii], label=f'pop{ii+1} fssh')
    
    

# %%
if __name__ == "__main__":
    _plot_adiabatic_diabatic(output_adiabatic, output_diabatic, output_fssh)
# %%
