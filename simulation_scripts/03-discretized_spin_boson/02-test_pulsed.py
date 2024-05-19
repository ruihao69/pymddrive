# %%
import numpy as np
import scipy.sparse as sps

from pymddrive.my_types import RealVector
from pymddrive.dynamics.options import NonadiabaticDynamicsMethods, BasisRepresentation, NumericalIntegrators
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, TD_HamiltonianBase, QuasiFloquetHamiltonianBase
from pymddrive.integrators.state import get_state
from pymddrive.dynamics.get_dynamics import get_dynamics
from pymddrive.dynamics.run import run_ensemble
from pymddrive.models.spin_boson_discrete import get_spin_boson, boltzmann_sampling, wigner_sampling

import os
from typing import Optional, List

def stop_condition(t, s):
    return t > 10

def save_pulse_obj(hamiltonian: TD_HamiltonianBase, project_prefix: str):
    # create the project_prefix directory if it does not exist
    if not os.path.isdir(project_prefix):
        os.makedirs(project_prefix)

    # save the pulses
    file = os.path.join(project_prefix, "pulse_obj.npz")
    if isinstance(hamiltonian, QuasiFloquetHamiltonianBase):
        ultrafast_pulse = hamiltonian.ultrafast_pulse
        envelope_pulse = hamiltonian.envelope_pulse
        np.savez(file, ultrafast_pulse=ultrafast_pulse, envelope_pulse=envelope_pulse, allow_pickle=True)
    elif isinstance(hamiltonian, TD_HamiltonianBase):
        pulse = hamiltonian.pulse
        np.savez(file, pulse=pulse, allow_pickle=True)
    else:
        import warnings
        warnings.warn("Hamiltonian is not a TD_HamiltonianBase or QuasiFloquetHamiltonianBase. No pulse object saved.")
        # raise ValueError("Hamiltonian must be either TD_HamiltonianBase or QuasiFloquetHamiltonianBase")

def test_sampling():
    n_modes = 9
    n_trajs = 10000
    hamiltonian = get_spin_boson(n_classic_bath=n_modes)
    initial_state = 0
    w_alpha = hamiltonian.omega_alpha
    R_eq = hamiltonian.get_donor_R() if initial_state == 0 else hamiltonian.get_acceptor_R()
    kT = hamiltonian.kT

    R_ensemble_boltz, P_ensemble_boltz = boltzmann_sampling(n_trajs, kT, w_alpha)
    R_ensemble_wign, P_ensemble_wign = wigner_sampling(n_trajs, kT, w_alpha)


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

    def get_initial_rho(NF: Optional[int]=None):
        rho0 = np.zeros((2, 2), dtype=np.complex128)
        rho0[init_state, init_state] = 1
        if NF is None:
            return rho0
        else:
            zeros_like = np.zeros_like(rho0)
            data = [zeros_like] * NF + [rho0] + [zeros_like] * NF
            return sps.block_diag(data).toarray()

    # initial states
    t0 = 0.0
    rho0 = [get_initial_rho(NF=NF) for _ in range(n_ensemble)]
    if basis_rep == BasisRepresentation.ADIABATIC:
        for ii in range(n_ensemble):
            H = hamiltonian.H(t0, R0[ii])
            evals, evecs = np.linalg.eigh(H)
            rho0_ii = evecs.T.conjugate() @ rho0[ii] @ evecs
            rho0[ii] = rho0_ii
    is_afssh = (solver == NonadiabaticDynamicsMethods.AFSSH)
    s0_list = [get_state(mass=mass, R=R0[ii], P=P0[ii], rho_or_psi=rho0[ii], is_afssh=is_afssh) for ii in range(n_ensemble)]
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
    param_set: str='TempelaarJCP2018',
    init_state: int = 0,
    basis_rep: BasisRepresentation = BasisRepresentation.DIABATIC,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.RK4,
    NF: Optional[int]=None,
    dt:float=0.01
):
    n_classic_bath = 100
    hamiltonian = get_spin_boson(n_classic_bath=n_classic_bath, param_set=param_set, pulse_type=pulse_type, E0_in_au=E0, Nc=N, phi=phi, t0=t0, NF=NF)
    save_pulse_obj(hamiltonian, project_prefix)
    # print(f"{hamiltonian.get_carrier_frequency()=}")
    # print(f"{hamiltonian.driving_Omega=}")
    # print(f"{hamiltonian.ultrafast_pulse=}")
    # print(f"{hamiltonian.envelope_pulse=}")
    # print(f"{abs(hamiltonian.envelope_pulse.A)=}")
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
    ntrajs = 32

    def estimate_NF(A: float, Omega: float, tol: float=1e-6) -> int:
        from scipy.special import jv
        # n = math.ceil(A / Omega)
        n = A / Omega
        NF = 1
        while True:
            val = abs(jv(NF, n))
            if val < tol:
                break
            NF += 1
        return NF

    # parameters for the laser pulse
    dimless_to_au = 0.00095
    dipole = 0.04                         # dipole moment in atomic units
    E0 = 0.0925                           # 300 TW/cm^2 laser E-field amplitude in atomic units
    A = dipole * E0 / dimless_to_au       # light-matter interaction strength in atomic units
    Omega_in_au = 0.05696                 # 800 nm laser wavelength to carrier frequency in atomic units
    Omega = Omega_in_au / dimless_to_au   # light frequency in atomic units
    phi = 0.0                             # phase of the laser pulse
    Nc = 16                                # number of cycles in the sine-squared pulse
    pulse_type = 'morlet_real'            # pulse type
    T = 2 * np.pi / Omega                 # period of the light field
    tau = T * Nc / 3
    t0 = 4 * tau                          # time delay for the Morlet pulse
    print(f"Omega: {Omega}, A: {A}, NF: {estimate_NF(A, Omega)}, tau: {tau}, t0: {t0}")
    # pulse_type = 'sine_squared_pulse'     # pulse type
    # t0 = None


    param_set = "BiasedTempelaarJCP2018Pulsed"
    # project_prefix = "data_CW_floquet_fssh-E0_0.0925-Omega_0.05696-N_8-phi_0.0"
    # project_prefix = "data_floquet_afssh-E0_0.0925-Omega_0.05696-N_8-phi_0.0"
    # project_prefix = "data_floquet_fssh-E0_0.0925-Omega_0.05696-N_8-phi_0.0"
    # project_prefix = "data_floquet_fssh-E0_0.0925-Omega_0.05696-N_16-phi_0.0"
    # project_prefix = "data_ehrenfest_diabatic-E0_0.0925-Omega_0.05696-N_8-phi_0.0"
    # project_prefix = "data_CW_ehrenfest_diabatic-E0_0.0925-Omega_0.05696-N_8-phi_0.0"
    project_prefix = "data_f_ehrenfest_adiabatic-E0_0.0925-Omega_0.05696-N_16-phi_0.0"
    init_state = 0
    dt = 1 / 20 / Omega
    NF = estimate_NF(A, Omega)
    # NF = 10
    # NF = None

    main(project_prefix=project_prefix, ntrajs=ntrajs, E0=E0, Omega=Omega, N=Nc, phi=phi, pulse_type=pulse_type, solver=NonadiabaticDynamicsMethods.EHRENFEST, basis_rep=BasisRepresentation.ADIABATIC, integrator=NumericalIntegrators.RK4, NF=NF, dt=dt, init_state=init_state, param_set=param_set, t0=t0)
    # main(project_prefix=project_prefix, ntrajs=ntrajs, E0=E0, Omega=Omega, N=Nc, phi=phi, pulse_type=pulse_type, solver=NonadiabaticDynamicsMethods.FSSH, basis_rep=BasisRepresentation.ADIABATIC, integrator=NumericalIntegrators.RK4, NF=NF, dt=dt, init_state=init_state, param_set=param_set, t0=t0)
    # main(project_prefix=project_prefix, ntrajs=ntrajs, E0=E0, Omega=Omega, N=Nc, phi=phi, pulse_type=pulse_type, solver=NonadiabaticDynamicsMethods.EHRENFEST, basis_rep=BasisRepresentation.DIABATIC, integrator=NumericalIntegrators.RK4, NF=NF, dt=dt, init_state=init_state, param_set=param_set, t0=t0)

    # test_sampling()
    # main(project_prefix=project_prefix, ntrajs=ntrajs, solver=NonadiabaticDynamicsMethods.FSSH, basis_rep=BasisRepresentation.ADIABATIC, init_state=init_state, integrator=NumericalIntegrators.ZVODE, dt=dt)
    # main(project_prefix=project_prefix, ntrajs=ntrajs, solver=NonadiabaticDynamicsMethods.EHRENFEST, basis_rep=BasisRepresentation.DIABATIC, init_state=init_state, integrator=NumericalIntegrators.ZVODE, dt=dt)
    # main(project_prefix=project_prefix, ntrajs=ntrajs, solver=NonadiabaticDynamicsMethods.AFSSH, basis_rep=BasisRepresentation.ADIABATIC, init_state=init_state, integrator=NumericalIntegrators.RK4, dt=dt)


# %%
