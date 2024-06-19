# %%
import numpy as np
import scipy.sparse as sps

from pymddrive.my_types import RealVector
from pymddrive.models.landry_spin_boson import get_landry_spin_boson
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.integrators.state import get_state
from pymddrive.dynamics.options import NonadiabaticDynamicsMethods, BasisRepresentation, NumericalIntegrators
from pymddrive.dynamics.get_dynamics import get_dynamics
from pymddrive.dynamics.run import run_ensemble
from landry_utils import sample_boltzmann
from spin_boson_postprocess import post_process

import os
from typing import Optional



def stop_condition(t, s):
    return t > 5000

def plot_hamiltonian(hamiltonian: HamiltonianBase):
    import matplotlib.pyplot as plt

    R = np.linspace(-10, 10, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dim = (2 * hamiltonian.NF + 1) * 2
    H = np.zeros((dim, dim, R.shape[0]))
    for ii, r in enumerate(R):
        H[:, :, ii] = hamiltonian.H(0.0, r)

    for ii in range(dim):
        ax.plot(R, H[ii, ii, :], label=r"$H^\text{F}"+rf"_{ii}$")
    ax.legend()
    plt.show()



def floquet_populations(
    R0: RealVector,
    P0: RealVector,
    init_state: int,
    hamiltonian: HamiltonianBase,
    mass: float=1.0,
    dt: float=0.1,
    NF: Optional[int]=None,
    data_dir: str='./',
    filename: str="fssh.nc",
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.FSSH,
    basis_rep: BasisRepresentation = BasisRepresentation.ADIABATIC,
) -> None:

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

    # plot_hamiltonian(hamiltonian)

    s0_list = [get_state(mass=mass, R=R0[ii], P=P0[ii], rho_or_psi=rho0[ii]) for ii in range(n_ensemble)]

    # get the dynamics ensemble
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

    active_surf = np.zeros(n_ensemble, dtype=np.int64)
    populations = np.zeros((2, ntrajs))
    for ii, dyn in enumerate(dynamics_list):
        props = dyn.solver.calculate_properties(t0, s0_list[ii])
        # populations[:, ii] = props.adiabatic_populations
        populations[:, ii] = props.diabatic_populations
        active_surf[ii] = dyn.solver.cache.active_surface[0]


    print(np.mean(populations[0]))
    print(np.mean(populations[1]))
    # print(np.sum(active_surf==10))
    # print(np.sum(active_surf==11))
    # print(np.sum(active_surf==10) + np.sum(active_surf==11))
    # print(active_surf.shape[0])





def run_landry_spin_boson(
    R0: RealVector,
    P0: RealVector,
    init_state: int,
    hamiltonian: HamiltonianBase,
    mass: float=1.0,
    dt: float=0.1,
    NF: Optional[int]=None,
    data_dir: str='./',
    filename: str="fssh.nc",
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.FSSH,
    basis_rep: BasisRepresentation = BasisRepresentation.ADIABATIC,
    integrator: NumericalIntegrators = NumericalIntegrators.RK4,
) -> None:
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
    evecs_list = []
    rho0 = [get_initial_rho(NF=NF) for _ in range(n_ensemble)]
    if basis_rep == BasisRepresentation.ADIABATIC:
        for ii in range(n_ensemble):
            H = hamiltonian.H(t0, R0[ii])
            evals, evecs = np.linalg.eigh(H)
            hamiltonian._last_evecs[:] = evecs
            rho0_ii = evecs.T.conjugate() @ rho0[ii] @ evecs
            rho0[ii] = rho0_ii
            evecs_list.append(evecs)

    s0_list = [get_state(mass=mass, R=R0[ii], P=P0[ii], rho_or_psi=rho0[ii]) for ii in range(n_ensemble)]

    # get the dynamics ensemble
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
        dynamics.hamiltonian.update_last_evecs(evecs_list[ii])
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
    )

    post_process(data_dir)

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
    param_set: str='LandryJCP2013',
    init_state: int = 0,
    basis_rep: BasisRepresentation = BasisRepresentation.DIABATIC,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.RK4,
    NF: Optional[int]=None
):
    # get the Landry Spin Boson model with a sine-squared pulse
    hamiltonian = get_landry_spin_boson(E0=E0, Omega=Omega, N=N, phi=phi, t0=t0, pulse_type=pulse_type, NF=NF, param_set=param_set)
    filename = get_filename(solver)

    # sample the initial conditions
    R_eq = hamiltonian.get_donor_R() if init_state == 0 else hamiltonian.get_acceptor_R()
    R0, P0 = sample_boltzmann(n_samples=ntrajs, kT=hamiltonian.get_kT(), Omega=hamiltonian.Omega_nuclear, mass=1.0, R_eq=R_eq)

    # run_landry_spin_boson(
    #     R0=R0,
    #     P0=P0,
    #     init_state=init_state,
    #     hamiltonian=hamiltonian,
    #     data_dir=project_prefix,
    #     filename=filename,
    #     solver=solver,
    #     basis_rep=basis_rep,
    #     integrator=integrator,
    #     NF=NF
    # )

    floquet_populations(
        R0=R0,
        P0=P0,
        init_state=init_state,
        hamiltonian=hamiltonian,
        data_dir=project_prefix,
        filename=filename,
        solver=solver,
        basis_rep=basis_rep,
        NF=NF
    )






# %%
if __name__ == "__main__":
    ntrajs = 2000

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
    dipole = 0.09   # dipole moment in atomic units
    E0 = 0.0925     # 300 TW/cm^2 laser intensity to E-field amplitude in atomic units
    Omega = 0.05696 # 800 nm laser wavelength to carrier frequency in atomic units
    A = dipole * E0
    phi = 0.0       # laser carrier phase
    N = 8           # Sine square envelope pulse with 8 cycles
    t0 = 0.0
    # pulse_type = 'sine_squared_pulse'
    pulse_type = 'morlet_real'
    param_set = "ResonantDoubleWell"

    project_prefix = f"data_floquet_fssh_E0_{E0}-Omega_{Omega}-N_{N}-phi_{phi}"
    NF = estimate_NF(A, Omega)
    # NF = None
    main(project_prefix=project_prefix, ntrajs=ntrajs, E0=E0, Omega=Omega, phi=phi, N=N, t0=t0, pulse_type=pulse_type, param_set=param_set, solver=NonadiabaticDynamicsMethods.FSSH, basis_rep=BasisRepresentation.ADIABATIC, NF=NF)


# %%
# rho0 = np.array([[ 0.57960976+0.j,         -0.49362071-0.00095022j],
#                  [-0.49362071+0.00095022j,  0.42039024+0.j        ]])
# zeros_like = np.zeros_like(rho0)
# rhoF = sps.block_diag([zeros_like]*NF + [rho0] + [zeros_like]*NF).toarray()
# s0 = get_state(mass=1.0, R=0.0, P=0.0, rho_or_psi=rhoF)
# hamiltonian = get_landry_spin_boson(E0=0.0925, Omega=0.05696, N=8, phi=0.0, pulse_type='sine_squared_pulse', NF=NF)
# dynamics = get_dynamics(t0=0.0, s0=s0, dt=0.1, hamiltonian=hamiltonian, method=NonadiabaticDynamicsMethods.FSSH, dynamics_basis=BasisRepresentation.ADIABATIC)
# dynamics.solver.calculate_properties(0.0, s0)

# %%
