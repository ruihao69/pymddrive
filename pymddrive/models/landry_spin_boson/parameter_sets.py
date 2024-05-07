# %%
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class LandryJCP2013:
    Omega_nuclear: float = 0.021375
    M: float = 1.0
    V: float = 0.00475
    Er: float = 0.00475
    epsilon0: float = 0.00475
    gamma: float = 0.04275
    kT: float = 0.00095
    lambd: float = np.sqrt(Er * M * Omega_nuclear**2 / 2)
    
        
def get_reduced_mass(M1, M2):
    return M1 * M2 / (M1 + M2)
        
@dataclass(frozen=True)
class SymmetricDoubleWell:
    Omega_nuclear: float = 0.0085 # streching mode of nitrogen monoxide
    M: float = 7.466666666666667 # reduced mass of nitrogen monoxide
    V: float = 0.0005
    Er: float = 0.00475
    epsilon0: float = 0 
    gamma: float = 0.04275
    kT: float = 0.00095
    lambd: float = np.sqrt(Er * M * Omega_nuclear**2 / 2)
    
@dataclass(frozen=True)
class ResonantDoubleWell:  
    Omega_nuclear: float = 0.0009113 # 200 cm-1
    M: float = 1836 # proton mass in atomic units
    V: float = 0.0005 # weak coupling
    Er: float = 0.00475
    epsilon0: float = 0.05695 # energy difference between the two wells
                              # intended to be resonant with the laser frequency
    gamma: float = 0.04275
    kT: float = 0.00095
    lambd: float = np.sqrt(Er * M * Omega_nuclear**2 / 2)
    
@dataclass(frozen=True)
class AmberJCTC2016:
    Omega_nuclear: float = 0.0009113    
    M: float = 1836
    V: float = 0.000228
    Er: float = 0.029042
    epsilon0: float = 0.05695
    kT: float = 0.0018224881 # 575.5 K
    gamma: float = 0.0009113 # moderate friction
    lambd: float = np.sqrt(Er * M * Omega_nuclear**2 / 2)
    
    
    
def main(landry_p, label: str):
    import numpy as np
    import matplotlib.pyplot as plt 
    
    from pymddrive.models.landry_spin_boson import LandrySpinBoson
    
    hamiltonian = LandrySpinBoson(
        Omega_nuclear=landry_p.Omega_nuclear,
        M=landry_p.M,
        V=landry_p.V,
        Er=landry_p.Er,
        epsilon0=landry_p.epsilon0,
        gamma=landry_p.gamma,
        kT=landry_p.kT,
    )
    
    R = np.linspace(-10, 10, 100)
    H = np.zeros((2, 2, len(R)))    
    E = np.zeros((2, len(R)))
    for ii, rr in enumerate(R):
        H[:, :, ii] = hamiltonian.H(0, np.array([rr]))
        E[:, ii] = np.linalg.eigvalsh(H[:, :, ii])
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(R, H[0, 0, :], ls='--', c='k')  
    ax.plot(R, H[1, 1, :], ls='--', c='k')  
    ax.plot(R, E[0, :], label="Ground state")
    ax.plot(R, E[1, :], label="Excited state")
    ax.set_xlabel("R")
    ax.set_ylabel("Energy")
    ax.set_title(label)
    ax.legend()
    plt.show()

# %% 
if __name__ == '__main__':
    main(landry_p=LandryJCP2013(), label="Landry JCP 2013")
    main(landry_p=SymmetricDoubleWell(), label="Symmetric Double Well")
    # nitrogen monoxide
    
    

# %%
