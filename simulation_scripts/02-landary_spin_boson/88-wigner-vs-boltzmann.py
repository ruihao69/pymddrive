# %%
import numpy as np
import matplotlib.pyplot as plt

from pymddrive.models.landry_spin_boson import get_landry_spin_boson

def sample_boltzmann(R_eq, mass, Omega, kT, n_samples):
    beta = 1.0 / kT
    
    sigma_P = np.sqrt(mass / beta)  
    P_list = np.random.normal(0, sigma_P, n_samples)
    
    sigm_R = 1.0 / np.sqrt(beta * mass) / Omega
    R_list = np.random.normal(R_eq, sigm_R, n_samples)
    return R_list, P_list

def sample_wigner(R_eq, mass, Omega, kT, n_samples):
    beta = 1.0 / kT

    sigma_dimless = np.sqrt(0.5 / np.tanh(0.5 * beta * Omega))
    
    dimless_to_P = np.sqrt(mass * Omega)
    dimless_to_R = np.sqrt(1.0 / (mass * Omega))
    
    sigma_P = sigma_dimless * dimless_to_P
    P_list = np.random.normal(0, sigma_P, n_samples)
    
    sigma_R = sigma_dimless * dimless_to_R
    R_list = np.random.normal(R_eq, sigma_R, n_samples)
    return R_list, P_list    

def plot_distribution_over_PES(R, H, E, R_list_bolz, R_list_wign):
    dim: int = H.shape[0]
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    handles = []
    for ii in range(dim):
        l, = ax.plot(R, H[ii, ii, :], label=f"Surface {ii}")
        handles.append(l)
    
    ax_two = ax.twinx() 
    hist, bins = np.histogram(R_list_bolz, bins=100, density=True)
    r_bins = 0.5 * (bins[1:] + bins[:-1])
    l2, = ax_two.plot(r_bins, hist, c='g', label='Boltzmann distribution')
    handles.append(l2)
    
    hist, bins = np.histogram(R_list_wign, bins=100, density=True)
    r_bins = 0.5 * (bins[1:] + bins[:-1])
    l3, = ax_two.plot(r_bins, hist, c='purple', label='Wigner distribution')
    handles.append(l3)
    
    
    ax.set_xlabel("R")
    ax.set_ylabel("Energy")
    ax_two.set_ylabel("Probability")
    ax.legend(handles=handles)
    
    
    

def main():
    hamiltonian = get_landry_spin_boson(param_set="ResonantDoubleWell")
    mass = hamiltonian.M
    kT = hamiltonian.kT
    Omega = hamiltonian.Omega_nuclear
    R_eq = hamiltonian.get_acceptor_R()
    
    R_list_bolz, P_list_bolz = sample_boltzmann(R_eq, mass, Omega, kT, 100000)
    R_list_wign, P_list_wign = sample_wigner(R_eq, mass, Omega, kT, 100000)
    
    COM = R_list_bolz.mean()
    L = R_list_bolz.max() - R_list_bolz.min()
    N = 1000
    r = np.linspace(COM - 3 * L, COM + 3 * L, 1000)
    H = np.zeros((2, 2, N))
    E = np.zeros((2, N))
    
    for ii in range(N):
        _H = hamiltonian.H(0.0, r[ii])
        evals, _ = np.linalg.eigh(_H)
        H[:, :, ii] = _H
        E[:, ii] = evals
    
    plot_distribution_over_PES(r, H, E, R_list_bolz, R_list_wign)
    plot_distribution_over_PES(r, H, E, P_list_bolz, P_list_wign)
     
    
# %%
if __name__ == "__main__":
    main()
# %%
