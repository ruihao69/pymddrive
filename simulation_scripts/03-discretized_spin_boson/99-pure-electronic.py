# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as LA

from pymddrive.my_types import ComplexOperator, GenericOperator, RealVector
from pymddrive.models.nonadiabatic_hamiltonian import diagonalization
from pymddrive.models.nonadiabatic_hamiltonian.math_utils import get_corrected_rho_or_psi
from pymddrive.models.spin_boson_discrete import get_spin_boson, boltzmann_sampling
from pymddrive.dynamics.options import BasisRepresentation
from pymddrive.pulses import PulseBase as Pulse
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.dynamics.nonadiabatic_solvers.ehrenfest.populations import compute_floquet_populations
from pymddrive.low_level.floquet import get_HF_cos

from typing import Tuple, Union

def get_hamiltonian(
    is_floquet: bool = False,
) -> HamiltonianBase:

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

    NF = estimate_NF(A, Omega) if is_floquet else None

    param_set = "BiasedTempelaarJCP2018Pulsed"
    return get_spin_boson(
        n_classic_bath=100,
        param_set=param_set,
        pulse_type=pulse_type,
        E0_in_au=E0,
        Nc=Nc,
        phi=phi,
        t0=t0,
        NF=NF,
    )

def derivative(
    H: GenericOperator,
    rho: ComplexOperator,
) -> ComplexOperator:
    return -1j * np.dot(H, rho) + 1j * np.dot(rho, H)

def get_pulse_dot(
    pulse: Pulse,
    t: float,
    h: float = 1e-8
) -> Union[float, complex]:
    return (pulse(t+h) - pulse(t-h)) / (2.0 * h)

def get_nac(
    dHdE: GenericOperator,
    evals: RealVector,
    evecs: GenericOperator,
    pulse_dot: Union[float, complex],
    h: float = 1e-8
) -> GenericOperator:
    dim = dHdE.shape[0]
    nac = np.zeros(shape=dHdE.shape, dtype=np.complex128)
    nac = evecs.T.conj() @ dHdE @ evecs
    for ii in range(dim):
        nac[ii, ii] = 0.0
        for jj in range(ii+1, dim):
            nac[ii, jj] = nac[ii, jj] / (evals[jj] - evals[ii]) * pulse_dot
            nac[jj, ii] = - np.conjugate(nac[ii, jj]) 
    # print(f"nac: {nac}")
    # nac = nac * pulse_dot
    # print(f"nac: {nac}")
    # raise ValueError("Stop here")
    return nac # * pulse_dot
    

def derivative_adiabatic(
    t: float,
    H: GenericOperator,
    mu: GenericOperator,
    pulse: Pulse,
    rho: ComplexOperator,
    last_evecs: GenericOperator,
    last_phase_correction: GenericOperator,
) -> ComplexOperator:
    evals, evecs, phase_correction = diagonalization(H, last_evecs) 
    pulse_dot = get_pulse_dot(pulse, t)
    nac = get_nac(mu, evals, evecs, pulse_dot)
    # H_diag = np.diagflat(evals)
    H = np.diagflat(evals) - 1.j * nac
    rho_dot = -1.j * (np.dot(H, rho) - np.dot(rho, H))
    return get_corrected_rho_or_psi(rho_or_psi=rho_dot, phase_correction=phase_correction/last_phase_correction)
    # return rho_dot

def derivative_adiabatic_floquet(
    t: float,
    H: GenericOperator,
    mu: GenericOperator,
    pulse: Pulse,
    rho: ComplexOperator,
    last_evecs: GenericOperator,
    last_phase_correction: GenericOperator,
) -> ComplexOperator:
    evals, evecs, phase_correction = diagonalization(H, last_evecs) 
    pulse_dot = get_pulse_dot(pulse, t)
    nac1 = get_nac(mu, evals, evecs, pulse_dot)
    nac2 = get_nac(mu.conjugate().T, evals, evecs, np.conjugate(pulse_dot))
    # print(f"nac1: {nac1}, nac2: {nac2}")
    nac = nac1 + nac2
    # H_diag = np.diagflat(evals)
    H = np.diagflat(evals) - 1.j * nac
    rho_dot = -1.j * (np.dot(H, rho) - np.dot(rho, H))
    return get_corrected_rho_or_psi(rho_or_psi=rho_dot, phase_correction=phase_correction/last_phase_correction)
    # return rho_dot 


def rk4(
    H0: GenericOperator,
    mu: GenericOperator,
    t: float,
    pulse: Pulse,
    rho: ComplexOperator,
    dt: float,
    last_evecs: GenericOperator=None,
    last_phase_correction: GenericOperator=None,
) -> ComplexOperator:
    def deriv_wrapper(t, H, rho):
        if last_evecs is None:
            return derivative(H, rho)
        else:
            return derivative_adiabatic(t, H, mu, pulse, rho, last_evecs, last_phase_correction)
        
    H1 = mu * pulse(t)
    H = H0 + H1
    k1 = deriv_wrapper(t, H, rho)
    
    H1 = mu * pulse(t + dt / 2)
    H = H0 + H1
    k2 = deriv_wrapper(t + dt / 2, H, rho + dt / 2 * k1)
    k3 = deriv_wrapper(t + dt / 2, H, rho + dt / 2 * k2)
    
    H1 = mu * pulse(t + dt)
    H = H0 + H1
    k4 = deriv_wrapper(t + dt, H, rho + dt * k3)
    return rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

# def get_cos_dHF_dE(
#     H1: GenericOperator,
#     NF: int,
#     dim: int
# ) -> GenericOperator:
def get_cos_dHF_dE(
    H1,
    NF: int,
    dim: int
):
    data = (H1 for _ in range(2*NF))
    diag_tmp = LA.block_diag(*data)
    vpatch = np.zeros((dim, diag_tmp.shape[1]), dtype=np.complex128)
    tmp = np.vstack((vpatch, diag_tmp))
    hpatch = np.zeros((tmp.shape[0], dim), dtype=np.complex128)
    tmp = np.hstack((tmp, hpatch))
    return 0.5 * (tmp + tmp.T.conj())   

def get_cos_dHF_dE_lower(
    H1,
    NF: int,
    dim: int
):
    data = (H1 for _ in range(2*NF))
    diag_tmp = LA.block_diag(*data)
    vpatch = np.zeros((dim, diag_tmp.shape[1]), dtype=np.complex128)
    tmp = np.vstack((vpatch, diag_tmp))
    hpatch = np.zeros((tmp.shape[0], dim), dtype=np.complex128)
    tmp = np.hstack((tmp, hpatch))
    return 0.5 * tmp    
    
    
def rk4_floquet(
    H0: GenericOperator,
    mu: GenericOperator,
    t: float,
    envelope_pulse: Pulse,
    rho: ComplexOperator,
    dt: float,
    Omega: float,
    NF: int,
    last_evecs: GenericOperator=None,
    last_phase_correction: GenericOperator=None,
) -> ComplexOperator:
    def deriv_wrapper(t, HF, rho):
        if last_evecs is None:
            return derivative(HF, rho)
        else:
            return derivative_adiabatic_floquet(t, HF, get_cos_dHF_dE_lower(mu, NF, dim=2).T.conj(), envelope_pulse, rho, last_evecs, last_phase_correction)
    
    H1 = mu * envelope_pulse(t)
    HF = get_HF_cos(H0, H1, Omega, NF)
    k1 = deriv_wrapper(t, HF, rho)
    
    H1 = mu * envelope_pulse(t + dt / 2)
    HF = get_HF_cos(H0, H1, Omega, NF)
    k2 = deriv_wrapper(t+0.5*dt, HF, rho + dt / 2 * k1)
    k3 = deriv_wrapper(t+0.5*dt, HF, rho + dt / 2 * k2)
    
    H1 = mu * envelope_pulse(t + dt)
    HF = get_HF_cos(H0, H1, Omega, NF)
    k4 = deriv_wrapper(t+dt, HF, rho + dt * k3)
    return rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    

def main_mean_field(
    rho0: ComplexOperator,
    H0: GenericOperator,
    mu: GenericOperator,
    ultrafast_pulse: Pulse,
    dt: float,
    save_every: int = 10,
    tfinal: float = 10,
) -> None:
    nsteps = int(tfinal / dt)
    time_out = np.zeros(nsteps//save_every+1, dtype=np.float64)
    populations_out = np.zeros((nsteps//save_every+1, 2), dtype=np.float64)

    t = 0.0
    rho = np.copy(rho0)
    for ii in range(nsteps):
        if ii % save_every == 0:
            time_out[ii//save_every] = t
            populations_out[ii//save_every] = np.real(np.diag(rho))

        rho = rk4(H0=H0, mu=mu, t=t, pulse=ultrafast_pulse, rho=rho, dt=dt,)
        t += dt
    return time_out, populations_out

def main_mean_field_adiabatic(
    rho0: ComplexOperator,
    H0: GenericOperator,
    mu: GenericOperator,
    ultrafast_pulse: Pulse,
    dt: float,
    save_every: int = 10,
    tfinal: float = 10,
) -> None:
    nsteps = int(tfinal / dt)
    time_out = np.zeros(nsteps//save_every+1, dtype=np.float64)
    populations_out = np.zeros((nsteps//save_every+1, 2), dtype=np.float64)

    t = 0.0
    evals, evecs, phase_correction = diagonalization(H0, None)
    evecs_last = np.copy(evecs)
    phase_correction_last = np.copy(phase_correction)
    rho = np.copy(evecs.T.conjugate() @ rho0 @ evecs)
    for ii in range(nsteps):
        if ii % save_every == 0:
            time_out[ii//save_every] = t
            rho_diabatic = evecs_last @ rho @ evecs_last.T.conj()
            populations_out[ii//save_every] = np.real(np.diag(rho_diabatic))

        rho = rk4(H0=H0, mu=mu, t=t, pulse=ultrafast_pulse, rho=rho, dt=dt, last_evecs=evecs_last, last_phase_correction=phase_correction_last)
        # rho_diabatic = evecs_last @ rho @ evecs_last.T.conj()
        
        H = H0 + mu * ultrafast_pulse(t)
        _, evecs, phase_correction = diagonalization(H, evecs_last)
        # rho = get_corrected_rho_or_psi(rho_or_psi=rho, phase_correction=phase_correction)
        # rho = evecs.T.conj() @ rho_diabatic @ evecs
        evecs_last[:] = evecs
        phase_correction_last[:] = phase_correction
        
        t += dt
    return time_out, populations_out

def main_floquet_mean_field(
    rho0: ComplexOperator,
    H0: GenericOperator,
    mu: GenericOperator,
    envelope_pulse: Pulse,
    dt: float,
    NF: int,
    save_every: int = 10,
    tfinal: float = 10,
) -> None:
    nsteps = int(tfinal / dt)
    time_out = np.zeros(nsteps//save_every+1, dtype=np.float64)
    populations_out = np.zeros((nsteps//save_every+1, 2), dtype=np.float64)

    zeros_like = np.zeros_like(rho0)
    data = [zeros_like] * NF + [rho0] + [zeros_like] * NF
    # data = [rho0] * (2 * NF + 1)
    rhoF = sp.block_diag(data).toarray()

    driving_Omega = hamiltonian_td.get_carrier_frequency()

    t = 0.0
    for ii in range(nsteps):
        if ii % save_every == 0:
            time_out[ii//save_every] = t
            # populations_out[ii//save_every] = np.real(np.diag(rho))
            populations_out[ii//save_every] = compute_floquet_populations(
                state=rhoF,
                dynamics_basis=BasisRepresentation.DIABATIC,
                floquet_basis=BasisRepresentation.DIABATIC,
                target_state_basis=BasisRepresentation.DIABATIC,
                Omega=driving_Omega,
                t=t,
                NF=NF,
                dim=2,
                evecs_0=None,
                evecs_F=None
            )
        rhoF = rk4_floquet(H0=H0, mu=mu, t=t, envelope_pulse=envelope_pulse, rho=rhoF, dt=dt, Omega=driving_Omega, NF=NF, )
        t += dt
    return time_out, populations_out

def main_floquet_mean_field_adiabatic(
    rho0: ComplexOperator,
    H0: GenericOperator,
    mu: GenericOperator,
    envelope_pulse: Pulse,
    dt: float,
    NF: int,
    save_every: int = 10,
    tfinal: float = 10,
) -> None:
    nsteps = int(tfinal / dt)
    time_out = np.zeros(nsteps//save_every+1, dtype=np.float64)
    populations_out = np.zeros((nsteps//save_every+1, 2), dtype=np.float64)

    zeros_like = np.zeros_like(rho0)
    data = [zeros_like] * NF + [rho0] + [zeros_like] * NF
    # data = [rho0] * (2 * NF + 1)
    rhoF = sp.block_diag(data).toarray()

    # unitary transformation to the adiabatic basis
    driving_Omega = hamiltonian_td.get_carrier_frequency()
    H1_initial = mu * envelope_pulse(0.0)
    HF_initial = get_HF_cos(H0, H1_initial, driving_Omega, NF)
    _, evecs_F = np.linalg.eigh(HF_initial)
    evals, evecs_F, phase_correction = diagonalization(HF_initial, evecs_F)
    rhoF = evecs_F.T.conj() @ rhoF @ evecs_F
    adiabatic_populations_out = np.zeros((nsteps//save_every+1, rhoF.shape[0]), dtype=np.float64)

    phase_correction_last = np.copy(phase_correction)
    evecs_F_last = np.copy(evecs_F)
    t = 0.0
    for ii in range(nsteps):
        if ii % save_every == 0:
            time_out[ii//save_every] = t
            # populations_out[ii//save_every] = np.real(np.diag(rho))
            HF = get_HF_cos(H0, mu * envelope_pulse(t), driving_Omega, NF)

            populations_out[ii//save_every] = compute_floquet_populations(
                state=rhoF,
                dynamics_basis=BasisRepresentation.ADIABATIC,
                floquet_basis=BasisRepresentation.DIABATIC,
                target_state_basis=BasisRepresentation.DIABATIC,
                Omega=driving_Omega,
                t=t,
                NF=NF,
                dim=2,
                evecs_0=None,
                evecs_F=evecs_F_last
            )
            adiabatic_populations_out[ii//save_every] = np.real(rhoF.diagonal())
            # print(f"t: {t}, populations: {populations_out[ii//save_every]}, adiabatic: {adiabatic_populations_out[ii//save_every].max()}")
        

        rhoF = rk4_floquet(H0=H0, mu=mu, t=t, envelope_pulse=envelope_pulse, rho=rhoF, dt=dt, Omega=driving_Omega, NF=NF, last_evecs=evecs_F_last, last_phase_correction=phase_correction_last)
        HF = get_HF_cos(H0, mu * envelope_pulse(t), driving_Omega, NF)
        _, evecs_F, phase_correction = diagonalization(HF, evecs_F_last)
        # rhoF = get_corrected_rho_or_psi(rho_or_psi=rhoF, phase_correction=phase_correction)
        # rho_F_diabatic = evecs_F_last @ rhoF @ evecs_F_last.T.conj() 
        # rhoF = evecs_F.T.conj() @ rho_F_diabatic @ evecs_F
        evecs_F_last[:] = evecs_F
        phase_correction_last[:] = phase_correction
        # rhoF_diabatic = evecs_F_last @ rhoF @ evecs_F_last.T.conj()
        t += dt
        # _, evecs_F = diagonalization(HF, evecs_F_last)
        # rhoF = evecs_F.T.conj() @ rhoF_diabatic @ evecs_F
        # evecs_F_last = np.copy(evecs_F)
    return time_out, populations_out

# %%
if __name__ == "__main__":
    hamiltonian_td = get_hamiltonian(is_floquet=False)
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = 1.0
    np.random.seed(300888)   
    R,P = boltzmann_sampling(1, hamiltonian_td.get_kT(), hamiltonian_td.omega_alpha)
    R = R[0]

    t_td, pop_td = main_mean_field(
        rho0=rho0,
        H0=hamiltonian_td.H0(R=R),
        mu=np.array([[0, 1], [1, 0]]) * hamiltonian_td.mu_in_au / hamiltonian_td.dimless2au,
        ultrafast_pulse=hamiltonian_td.pulse,
        dt=1/(20*hamiltonian_td.omega_alpha[-1]),
    )
    
    t_adtd, pop_adtd = main_mean_field_adiabatic(
        rho0=rho0,
        H0=hamiltonian_td.H0(R=R),
        mu=np.array([[0, 1], [1, 0]]) * hamiltonian_td.mu_in_au / hamiltonian_td.dimless2au,
        ultrafast_pulse=hamiltonian_td.pulse,
        dt=1/(20*hamiltonian_td.omega_alpha[-1]),
    )

    hamiltonian_td = get_hamiltonian(is_floquet=True)

    t_fq, pop_fq = main_floquet_mean_field(
        rho0=rho0,
        H0=hamiltonian_td.H0(R=R),
        mu=np.array([[0, 1], [1, 0]]) * hamiltonian_td.mu_in_au / hamiltonian_td.dimless2au,
        envelope_pulse=hamiltonian_td.envelope_pulse,
        dt=1/(20*hamiltonian_td.omega_alpha[-1]),
        NF=hamiltonian_td.NF,
    )

    t_adfq, pop_adfq = main_floquet_mean_field_adiabatic(
        rho0=rho0,
        H0=hamiltonian_td.H0(R=R),
        mu=np.array([[0, 1], [1, 0]]) * hamiltonian_td.mu_in_au / hamiltonian_td.dimless2au,
        envelope_pulse=hamiltonian_td.envelope_pulse,
        dt=1/(20*hamiltonian_td.omega_alpha[-1]),
        NF=hamiltonian_td.NF,
    )

    # NF = hamiltonian_td.NF
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t_td, pop_td[:, 0], label="Time-dependent")
    ax.plot(t_adtd, pop_adtd[:, 0], ls='--', label="Time-dependent Adiabatic")
    # ax.plot(t_fq, pop_fq[:, 0]/(2*NF+1), label="Floquet")
    ax.plot(t_fq, pop_fq[:, 0], ls='-.', label="Floquet")
    ax.plot(t_adfq, pop_adfq[:, 0], ls='--', label="Floquet Adiabatic")
    ax.legend()
    plt.show()



# %%
from scipy.linalg import expm
from tests.test_utils import get_random_O
np.random.seed(0)
H = get_random_O(2, is_complex=True)
print(H)
dt = 0.0003
rho0 = np.zeros((2, 2), dtype=np.complex128)
rho0[0, 0] = 1.0
U = expm(-1j * H * dt)

rho_next = np.copy(rho0)
rho_next_linear = np.copy(rho0)

evals, evecs = np.linalg.eigh(H)
rho_adiabatic = np.copy(evecs.T.conj() @ rho0 @ evecs)


print(f"Before dynamics {rho_adiabatic}")

for ii in range(30000):
    rho_next = U @ rho_next @ U.T.conj()
    rho_next_linear = rho_next_linear - 1j * dt * (H @ rho_next_linear - rho_next_linear @ H)
    rho_adiabatic = rho_adiabatic - 1j * dt * (np.diagflat(evals) @ rho_adiabatic - rho_adiabatic @ np.diagflat(evals))
    if ii % 100 == 0:
        print("===")
        print(rho_next)
        print(rho_next_linear)
        print(evecs @ rho_adiabatic @ evecs.T.conj())
        print("===")
        
print(f"After dynamics {rho_adiabatic}")


# %%
