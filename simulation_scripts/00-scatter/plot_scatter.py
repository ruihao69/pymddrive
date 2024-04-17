# %%
import numpy as np
import matplotlib.pyplot as plt
import scienceplots 
plt.style.use(['science'])

import os
import glob
from typing import List

def load_P0(project_dir: str) -> np.ndarray:
    pattern = "P0-*"
    dirs = np.array(glob.glob(os.path.join(project_dir, pattern)))
    P0 = np.array([float(os.path.basename(dir).split("-")[1]) for dir in dirs])
    argsort = np.argsort(P0)
    return P0[argsort], dirs[argsort]
    

def load_scatter_data(project_dir: str) -> np.ndarray:
    P0, dirs = load_P0(project_dir)
    scatter_out = np.zeros((len(P0), 4))
    for ii, dir in enumerate(dirs):
        scatter_out[ii] = np.loadtxt(os.path.join(dir, "scatter_result.dat"))
        
    return P0, scatter_out

def plot_scatter(data: List[tuple], labels: List[str] = None):
    fig = plt.figure(dpi=300, figsize=(3*2, 2.25*2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    axs = gs.subplots().flatten()
    
    for ii, (P0, scatter_out) in enumerate(data):
        for jj, ax in enumerate(axs):
            ax.plot(P0, scatter_out[:, jj], label=labels[ii])
            ax.set_xlabel("initial momentum")
            ax.set_ylabel("Probability")
            ax.set_title(["Reflect lower", "Transmission lower", "Reflect upper", "Transmission upper"][jj])
        ax.legend()
    plt.show()
    
def main(project_dir_list: List[str], labels: List[str]):
    data = []
    for project_dir in project_dir_list:
        P0, scatter_out = load_scatter_data(project_dir)
        data.append((P0, scatter_out))
    plot_scatter(data, labels)
    
if __name__ == "__main__":
    # main("data_tullyone_ehrenfest_adiabatic")
    labels = ['Ehrenfest', 'Floquet Ehrenfest', 'Floquet FSSH']
    project_dir_list = ["data_ehrenfest_diabatic-Omega-0.05-tau-100-pulse-3", "data_floquet_ehrenfest_diabatic-Omega-0.05-tau-100-pulse-3", "data_floquet_fssh-Omega-0.05-tau-100-pulse-3"]
    main(project_dir_list, labels)
# %%
