# %%
import numpy as np
from numpy.typing import ArrayLike

from pymddrive.models.morse import FourLevelMorse, morse, d_morse_dR
from pymddrive.integrators.state import State
from pymddrive.dynamics.options import BasisRepresentation, QunatumRepresentation, NonadiabaticDynamicsMethods, NumericalIntegrators    
from pymddrive.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics, run_nonadiabatic_dynamics_ensembles
from pymddrive.dynamics.misc_utils import eval_nonadiabatic_hamiltonian

import os 
from typing import Tuple

def stop_condition(t, s, states):
    return t > 200

def break_condition(t, s, states):
    return False

def get_minimum_of_morse_V11(params: dict):
    r_samples = np.linspace(0, 0.6, 10000)
    dVdR = d_morse_dR(r_samples, **params)
    min_mask = np.argmin(np.abs(dVdR))
    return r_samples[min_mask]

def approximate_second_derivative_of_morse_V11(R0, params: dict, h: float=1e-6) -> float:
    return (morse(R0+h, **params) - 2 * morse(R0, **params) + morse(R0-h, **params)) / h**2

def get_V11_harmonic(params: dict, mass: float):
    R0 = get_minimum_of_morse_V11(params)
    d2VdR2 = approximate_second_derivative_of_morse_V11(R0, params)
    Omega = np.sqrt(d2VdR2 / mass)
    print(f"{R0=}, {Omega=}")
    return R0, Omega

def sample_morse_boltzmann(
    n_samples: int, 
    T: float, 
    params: dict,
    mass: float = 2000
):
    kB_in_au = 3.166815e-6
    beta = 1 / (kB_in_au * T)
    sigma_momentum = np.sqrt(mass / beta)
    momentum_samples = np.random.normal(0, sigma_momentum, n_samples)
    R0, Omega = get_V11_harmonic(params, mass)
    sigma_R = 1.0 / np.sqrt(beta * mass) / Omega
    print(f"{sigma_R=}")
    position_samples = np.random.normal(R0, sigma_R, n_samples)
    return position_samples, momentum_samples
    
def run_one_four_level_morse(
    R0: float,
    P0: float,
    mass: float=2000,
    qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
    basis_rep: BasisRepresentation = BasisRepresentation.Diabatic,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
):
    from pymddrive.dynamics.misc_utils import eval_nonadiabatic_hamiltonian
    hamiltonian = FourLevelMorse(VV=0.02)
    dim = hamiltonian.dim
    rho0 = np.zeros((dim, dim), dtype=np.complex128)
    rho0_diabatic = np.array([[1.0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.complex128)
    hami_return = eval_nonadiabatic_hamiltonian(0, np.array([R0]), hamiltonian, basis_rep=BasisRepresentation.Diabatic)
    evecs = hami_return.evecs
    rho0_adiabatic = evecs.T.conj() @ rho0_diabatic @ evecs
    if basis_rep == BasisRepresentation.Adiabatic:
        s0 = State.from_variables(R=R0, P=P0, rho=rho0_adiabatic)
    elif basis_rep == BasisRepresentation.Diabatic:
        s0 = State.from_variables(R=R0, P=P0, rho=rho0_diabatic)
        
    # dt_max = 0.003
    
    dyn = NonadiabaticDynamics(
        hamiltonian=hamiltonian,
        t0=0.0,
        s0=s0,
        mass=mass,
        basis_rep=basis_rep,
        qm_rep=qm_rep,
        solver=solver,
        numerical_integrator=integrator,
        dt=0.1,
        save_every=10,
        # max_step=dt_max,
    )
    
    return run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)

def generate_ensembles(
    initial_diabatic_states: int,
    R0_samples: ArrayLike,
    P0_samples: ArrayLike,
    mass: float=2000,
    qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
    basis_rep: BasisRepresentation = BasisRepresentation.Diabatic,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
) -> Tuple[NonadiabaticDynamics]:
    # initialize the electronic states
    assert (initial_diabatic_states >= 0) and (initial_diabatic_states < 4), f"Valid states should be between 0 and 3. Got {initial_diabatic_states}."
    rho0_diabatic = np.zeros((4, 4), dtype=np.complex128)
    rho0_diabatic[initial_diabatic_states, initial_diabatic_states] = 1.0
    
    assert (n_samples := len(R0_samples)) == len(P0_samples), "The number of R0 and P0 samples should be the same."
    ensemble = ()
    for ii, (R0, P0) in enumerate(zip(R0_samples, P0_samples)):
        hamiltonian = FourLevelMorse(VV=0.02)
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
            dt=0.1,
            save_every=10,
        )
        ensemble += (dyn,)
    return ensemble
        
   

def main(ntrajs: int, basis_rep: BasisRepresentation = BasisRepresentation.Diabatic, solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST):
    T = 300
    mass = 2000
    _hamiltonian = FourLevelMorse(VV=0.02)
    position_samples, momentum_samples = sample_morse_boltzmann(ntrajs, T, _hamiltonian.params_1)
    
    dyn_ensemble = generate_ensembles(
        0, position_samples, momentum_samples, mass,
        qm_rep=QunatumRepresentation.DensityMatrix,
        basis_rep=basis_rep,
        solver=NonadiabaticDynamicsMethods.EHRENFEST,
        integrator=NumericalIntegrators.ZVODE,
    )
    
    output_ensemble_averaged = run_nonadiabatic_dynamics_ensembles(dyn_ensemble, stop_condition, break_condition)
    plot_ensembles_averaged(output_ensemble_averaged)
    return output_ensemble_averaged



# %%
def _compare_adiabatic_diabatic(
    R0: float,
    P0: float,
):
    output_adiabatic = run_one_four_level_morse(
        R0, P0, basis_rep=BasisRepresentation.Adiabatic,
    )
    
    output_diabatic = run_one_four_level_morse(
        R0, P0, basis_rep=BasisRepresentation.Diabatic,
    )
    return output_adiabatic, output_diabatic

def _plot_adiabatic_diabatic(output_adiabatic, output_diabatic, output_fssh):
    import matplotlib.pyplot as plt
    ta, td, tfssh = output_adiabatic['time'], output_diabatic['time'], output_fssh['time']
    Ra, Rd, Rfssh = output_adiabatic['states']['R'], output_diabatic['states']['R'], output_fssh['states']['R']
    Pa, Pd, Pfssh = output_adiabatic['states']['P'], output_diabatic['states']['P'], output_fssh['states']['P']
    popa_diab, popd_diab, popfssh_diab = output_adiabatic['diab_populations'], output_diabatic['diab_populations'], output_fssh['diab_populations']
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
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for ii in range(4):
        ax.plot(ta, popa_diab[:, ii], ls='-', label=f'pop{ii+1} adiabatic')
        ax.plot(td, popd_diab[:, ii], ls='-.', label=f'pop{ii+1} diabatic')
        ax.plot(tfssh, popfssh_diab[:, ii], ls='--', label=f'pop{ii+1} fssh')
        
    ax.legend()
    
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
    for ii in range(4):
        ax.plot(time, pop_diab[:, ii], label=f'pop{ii+1}')
    ax.set_xlabel('Time (a.u.)')
    ax.set_ylabel('Diabatic Population')
    ax.legend()
    plt.show()
    

def _test_sampling(T: float = 300):
    import matplotlib.pyplot as plt
    hamiltonian = FourLevelMorse(VV=0.02)
    position_samples, momentum_samples = sample_morse_boltzmann(3000, T, hamiltonian.params_1)
    R = np.linspace(0.0, 1.0, 10000)
    V = np.zeros((4, len(R)))
    for ii, rr in enumerate(R):
        H = hamiltonian.H(0, rr)
        for jj in range(4):
            V[jj, ii] = H[jj, jj]
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for jj in range(4):
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
    output_diabatic = main(ntrajs=200, basis_rep=BasisRepresentation.Diabatic)
    output_adiabatic = main(ntrajs=200, basis_rep=BasisRepresentation.Adiabatic)
    output_fssh = main(ntrajs=200, basis_rep=BasisRepresentation.Adiabatic, solver=NonadiabaticDynamicsMethods.FSSH)
    _plot_adiabatic_diabatic(output_adiabatic, output_diabatic, output_fssh)
    

# %%
