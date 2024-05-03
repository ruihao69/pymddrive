# %%
import numpy as np
import matplotlib.pyplot as plt

from pymddrive.models.landry_spin_boson import get_landry_spin_boson

def main():
    hamiltonian = get_landry_spin_boson()
    R = np.linspace(-10, 10, 1000)
    H = np.zeros((R.size, 2, 2), dtype=np.complex128)
    for ii, r in enumerate(R):
        H[ii] = hamiltonian.H(0, r)
    
    E, U = np.linalg.eigh(H)     
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for ii in range(2):
        ax.plot(R, H[:, ii, ii].real, label=f"H{ii}")
        ax.plot(R, E[:, ii].real, label=f"E{ii}")   
    ax.set_xlabel("R")
    ax.set_ylabel("Energy")
    ax.legend()
    ax.legend()
    
# %%
if __name__ == "__main__":
    main()
    
    
# %%
