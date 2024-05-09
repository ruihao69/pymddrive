# %%
import numpy as np
import scipy.sparse as sps

from pymddrive.my_types import RealVector
from pymddrive.dynamics.options import NonadiabaticDynamicsMethods, BasisRepresentation, NumericalIntegrators
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.integrators.state import get_state
from pymddrive.dynamics.get_dynamics import get_dynamics
from pymddrive.dynamics.run import run_ensemble
from pymddrive.models.spin_boson_discrete import get_spin_boson, boltzmann_sampling, wigner_sampling

import os
from typing import Optional, List

def stop_condition(t, s):
    return t > 40
    # return t > 0.01

def test_sampling():
    n_modes = 9
    n_trajs = 10000
    hamiltonian = get_spin_boson(n_classic_bath=n_modes)
    initial_state = 0
    w_alpha = hamiltonian.omega_alpha
    R_eq = hamiltonian.get_donor_R() if initial_state == 0 else hamiltonian.get_acceptor_R()
    kT = hamiltonian.kT

    R_ensemble_boltz, P_ensemble_boltz = boltzmann_sampling(n_trajs, kT, w_alpha)
    R_ensemble_wign, P_ensemble_wign = wigner_sampling(n_trajs, kT, w_alpha, R_eq)


    def plot_distributions(R_ensemble_boltz, R_ensemble_wign):
        import matplotlib.pyplot as plt
        fig = plt.figure(dpi=300, figsize=(4*3, 3*3))
        gs = fig.add_gridspec(3, 3)
        axs = gs.subplots().flatten()

        def get_hist(dat):
            hist, bins = np.histogram(dat, bins=300, density=True)
            r_bins = 0.5 * (bins[1:] + bins[:-1])
            return r_bins, hist

        for ii, (ax, r_eq) in enumerate(zip(axs, R_eq, )):
            r_boltz = np.array(R_ensemble_boltz)[:, ii]
            r_wign = np.array(R_ensemble_wign)[:, ii]

            ax.plot(*get_hist(r_boltz), label="Boltzmann")
            ax.plot(*get_hist(r_wign), label="Wigner")

            ax.axvline(r_eq, ls='-.', c='k', alpha=0.5)
            ax.set_title(f"Mode {ii}")
            ax.legend()

    plot_distributions(R_ensemble_boltz, R_ensemble_wign)
    plot_distributions(P_ensemble_boltz, P_ensemble_wign)

def run_spin_boson(
    R0: List[RealVector],
    P0: List[RealVector],
    init_state: int,
    hamiltonian: HamiltonianBase,
    mass: float=1.0,
    dt: float=0.01,
    # dt: float=0.03,
    NF: Optional[int]=None,
    data_dir: str='./',
    filename: str="fssh.nc",
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.FSSH,
    basis_rep: BasisRepresentation = BasisRepresentation.ADIABATIC,
    integrator: NumericalIntegrators = NumericalIntegrators.RK4,
) -> None:
    R0 = np.array(R0)
    P0 = np.array(P0)
    assert R0.shape == P0.shape

    n_ensemble = R0.shape[0]

    def get_initial_psi(NF: Optional[int]=None):
        psi0 = np.zeros(2, dtype=np.complex128)
        psi0[init_state] = 1.0
        if NF is None:
            return psi0
        else:
            raise NotImplementedError("Floquet dynamics with state vector psi0 is not implemented yet")


    # initial states
    t0 = 0.0
    # rho0 = [get_initial_rho(NF=NF) for _ in range(n_ensemble)]
    psi0 = [get_initial_psi(NF=NF) for _ in range(n_ensemble)]
    if basis_rep == BasisRepresentation.ADIABATIC:
        for ii in range(n_ensemble):
            H = hamiltonian.H(t0, R0[ii])
            evals, evecs = np.linalg.eigh(H)
            psi0_ii = evecs.T.conjugate() @ psi0[ii]
            psi0[ii] = psi0_ii

    s0_list = [get_state(mass=mass, R=R0[ii], P=P0[ii], rho_or_psi=psi0[ii]) for ii in range(n_ensemble)]
    dynamics_list = []
    for ii in range(n_ensemble):
        dynamics = get_dynamics(
            t0=t0,
            s0=s0_list[ii],
            dt=dt,
            hamiltonian=hamiltonian,
            method=solver,
            dynamics_basis=basis_rep
        )
        dynamics_list.append(dynamics)

    # prepare the data directory
    filename = os.path.join(data_dir, filename)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    run_ensemble(
        dynamics_list=dynamics_list,
        break_condition=stop_condition,
        filename=filename,
        numerical_integrator=integrator,
        save_every=10,
        parallel=True
    )

def get_filename(solver: NonadiabaticDynamicsMethods) -> str:
    return f"{solver.name.lower()}.nc"

def main(
    project_prefix: str,
    ntrajs: int,
    E0: Optional[float]=None,
    Omega: Optional[float]=None,
    phi: Optional[float]=None,
    N: Optional[int]=None,
    t0: Optional[int]=None,
    pulse_type: str='no_pulse',
    init_state: int = 0,
    basis_rep: BasisRepresentation = BasisRepresentation.DIABATIC,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.RK4,
    NF: Optional[int]=None,
    dt:float=0.01
):
    n_classic_bath = 100
    hamiltonian = get_spin_boson(n_classic_bath=n_classic_bath)

    dt = 1 / (10 * hamiltonian.omega_alpha[-1])

    # get the Landry Spin Boson model with a sine-squared pulse
    filename = get_filename(solver)

    # get the initial conditions
    R0, P0 = boltzmann_sampling(ntrajs, hamiltonian.kT, hamiltonian.omega_alpha)
    run_spin_boson(
        R0=R0,
        P0=P0,
        init_state=init_state,
        hamiltonian=hamiltonian,
        mass=1,
        dt=dt,
        NF=NF,
        data_dir=project_prefix,
        filename=filename,
        solver=solver,
        basis_rep=basis_rep,
        integrator=integrator
    )


# %%
if __name__ == "__main__":
    ntrajs = 128
    # project_prefix = "data_test"
    # project_prefix = "data_ehrenfest_dibatic"
    project_prefix = "data_fssh"
    init_state = 0
    dt = 0.003

    # test_sampling()
    # main(project_prefix=project_prefix, ntrajs=ntrajs, solver=NonadiabaticDynamicsMethods.FSSH, basis_rep=BasisRepresentation.ADIABATIC, init_state=init_state, integrator=NumericalIntegrators.RK4, dt=dt)
    R0, P0 = boltzmann_sampling(ntrajs, 1.0, np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2, 0.3]))
    s0 = get_state(mass=1.0, R=R0[0], P=P0[0], rho_or_psi=np.array([1.0, 0.0]))
    print(s0)
    v = s0.get_v()
    print(P0[0])
    print(v)


#%%
