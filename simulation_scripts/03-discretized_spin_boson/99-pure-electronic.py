# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

from pymddrive.my_types import ComplexOperator, GenericOperator
from pymddrive.models.spin_boson_discrete import get_spin_boson, boltzmann_sampling
from pymddrive.dynamics.options import BasisRepresentation
from pymddrive.pulses import PulseBase as Pulse
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.dynamics.nonadiabatic_solvers.ehrenfest.populations import compute_floquet_populations
from pymddrive.low_level.floquet import get_HF_cos, get_HF_sin


def _dim_to_dimF(dim: int, NF: int) -> int:
    return dim * (2 * NF + 1)

def _get_Floquet_offset(dim_sys: int, NF: int, Omega: float):
    return [np.identity(dim_sys) * ii * Omega for ii in range(-NF, NF+1)]

def get_HF_cos_python(
    H0: GenericOperator, # The time-independent part of the Hamiltonian,
    V: GenericOperator, # The time-dependent part of the Hamiltonian (times cosine function),
    Omega: float, # The frequency of the driving field,
    NF: int, # The number of floquet levels to consider,
    is_gradient: bool = False,
    to_csr: bool = False
) -> GenericOperator:
    """ Suppose the Hamiltonian is given by H(t) = H0 + V(t) * cos(Omega * t). """
    dim = H0.shape[0]
    dimF = _dim_to_dimF(dim, NF)
    dtype = np.complex128 if np.iscomplexobj(H0) or np.iscomplexobj(V) else np.float64  
    
    if NF == 0:
        return sp.bsr_matrix(H0, dtype=dtype)
    
    offsets = _get_Floquet_offset(dim, NF, Omega) 
    offsets = np.zeros_like(offsets) if is_gradient else offsets
    V_upper = V / 2
    V_lower = V.transpose().conj() / 2
    # V_upper = V.transpose().conj()
    # V_lower = V 
    
    
    data_first_row = (H0 + offsets[0], V_upper)
    data_middle = ((V_lower, H0+offsets[ii+1], V_upper) for ii in range(2*NF-1))
    data_last_row = (V_lower, H0 + offsets[-1])
    
    data = np.concatenate((data_first_row, *data_middle, data_last_row))
    
    indptr = np.concatenate([(0, ), 2+3*np.arange(0, 2*NF, dtype=int), (6*NF+1, )])
    indices = np.concatenate([(0, 1), *(i+np.arange(0, 3) for i in range(2*NF-1)), (2*NF-1, 2*NF)])
    
    HF = sp.bsr_matrix((data, indices, indptr), shape=(dimF, dimF), dtype=dtype) 
    # print(f"{LA.ishermitian(HF.toarray())=}")
    return HF.tocsr() if to_csr else HF.toarray()

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

def rk4(
    H: GenericOperator,
    rho: ComplexOperator,
    dt: float,
) -> ComplexOperator:
    k1 = derivative(H, rho)
    k2 = derivative(H, rho + dt / 2 * k1)
    k3 = derivative(H, rho + dt / 2 * k2)
    k4 = derivative(H, rho + dt * k3)
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
        
        H1 = mu * ultrafast_pulse(t) 
        H = H0 + H1
        rho = rk4(H, rho, dt)
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
        
        H1 = mu * envelope_pulse(t) 
        HF = get_HF_cos_python(H0, H1, driving_Omega, NF)
        rhoF = rk4(HF, rhoF, dt)
        t += dt
    return time_out, populations_out

# %% 
if __name__ == "__main__":
    hamiltonian_td = get_hamiltonian(is_floquet=False)   
    rho0 = np.zeros((2, 2), dtype=np.complex128)   
    rho0[0, 0] = 1.0
    R,P = boltzmann_sampling(1, hamiltonian_td.get_kT(), hamiltonian_td.omega_alpha)
    R = R[0]
    
    t_td, pop_td = main_mean_field(
        rho0=rho0,
        H0=hamiltonian_td.H0(R=R),
        mu=np.array([[0, 1], [1, 0]]) * hamiltonian_td.mu_in_au / hamiltonian_td.dimless2au,
        ultrafast_pulse=hamiltonian_td.pulse,
        dt=1/(100*hamiltonian_td.omega_alpha[-1]),
    )
    
    hamiltonian_td = get_hamiltonian(is_floquet=True)
    
    t_fq, pop_fq = main_floquet_mean_field(
        rho0=rho0,
        H0=hamiltonian_td.H0(R=R),
        mu=np.array([[0, 1], [1, 0]]) * hamiltonian_td.mu_in_au / hamiltonian_td.dimless2au,
        envelope_pulse=hamiltonian_td.envelope_pulse,
        dt=1/(100*hamiltonian_td.omega_alpha[-1]),
        NF=hamiltonian_td.NF,
    )
    
    NF = hamiltonian_td.NF
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t_td, pop_td[:, 0], label="Time-dependent")
    # ax.plot(t_fq, pop_fq[:, 0]/(2*NF+1), label="Floquet")    
    ax.plot(t_fq, pop_fq[:, 0], label="Floquet")    
    ax.legend()
    
        
    
    
    
# %%
