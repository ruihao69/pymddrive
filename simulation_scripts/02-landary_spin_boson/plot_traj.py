# %%
import numpy as np
from plotter import SpinBosonPlotter

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])

def plot_P0(time, P0, label: str, fig=None):
    if fig is None:
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
    else:
        ax = fig.gca()
        
    ax.plot(time, P0, label=label)
    
    return fig
    
    
    

def main():
    data_ehrenfest_diabatic = np.loadtxt("data_ehrenfset_diabatic/traj.dat")
    data_ehrenfest_adiabatic = np.loadtxt("data_ehrenfset_adiabatic/traj.dat")
    data_fssh_diabatic = np.loadtxt("data_fssh_diabatic/traj.dat")

    sbp = SpinBosonPlotter()

    sbp.plot_all(dim=2, traj_data=data_ehrenfest_diabatic, label_base="Ehrenfest Diabatic")
    sbp.plot_all(dim=2, traj_data=data_ehrenfest_adiabatic, label_base="Ehrenfest Adiabatic")
    sbp.plot_all(dim=2, traj_data=data_fssh_diabatic, label_base="FSSH Diabatic")

    sbp.finalize()
    
    fig = None 
    for data, label in zip([data_ehrenfest_diabatic, data_ehrenfest_adiabatic, data_fssh_diabatic], ["Ehrenfest Diabatic", "Ehrenfest Adiabatic", "FSSH"]):
        time = data[:, 0]
        P0 = data[:, 5]
        fig = plot_P0(time, P0, label, fig)
        
    data_xct = np.loadtxt("exact.dat", delimiter=',')
    time, P0 = data_xct[:, 0], data_xct[:, 1]
    fig = plot_P0(time, P0, "QuaPI", fig)
    ax = fig.gca()
    ax.set_xlabel("Time (arb. units)")
    ax.set_ylabel("Population")
    ax.legend()
    
    
        

# %%
if __name__ == '__main__':
    main()

# %%
