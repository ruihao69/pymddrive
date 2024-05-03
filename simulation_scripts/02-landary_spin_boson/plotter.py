import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science'])

from typing import Tuple
from dataclasses import dataclass

@dataclass
class SpinBosonPlotter:
    fig_nuc: plt.Figure
    axs_nuc: plt.Axes
    fig_energy: plt.Figure
    axs_energy: plt.Axes
    fig_pop: plt.Figure
    axs_pop: plt.Axes
    
    def __init__(self):
        self.fig_nuc, self.axs_nuc = self.init_nuc_plot()
        self.fig_energy, self.axs_energy = self.init_energy_plot()
        self.fig_pop, self.axs_pop = self.init_pop_plot()
    
    def init_nuc_plot(self) -> Tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=(3*2, 2), constrained_layout=True, dpi=300)
        gs = fig.add_gridspec(1, 2)
        axs = gs.subplots().flatten()
        ylabels = ['R', 'P']
        for ax, ylabel in zip(axs, ylabels):
            ax.set_xlabel('Time (arb. units)')
            ax.set_ylabel(ylabel)
        return fig, axs
    
    def init_energy_plot(self) -> Tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=(3, 2*3), constrained_layout=True, dpi=300)
        gs = fig.add_gridspec(3, 1)
        axs = gs.subplots().flatten()
        ylabels = ['KE', 'PE', 'TE']
        for ax, ylabel in zip(axs, ylabels):
            ax.set_xlabel('Time (arb. units)')
            ax.set_ylabel(ylabel)
        return fig, axs
    
    def init_pop_plot(self) -> Tuple[plt.Figure, plt.Axes]:
        fig = plt.figure(figsize=(3*2, 2), constrained_layout=True, dpi=300)
        gs = fig.add_gridspec(1, 2)
        axs = gs.subplots().flatten()
        ylabels = ['Diabatic Population', 'Adiabatic Population']
        for ax, ylabel in zip(axs, ylabels):
            ax.set_xlabel('Time (arb. units)')
            ax.set_ylabel(ylabel)
        return fig, axs
    
    def plot_all(self, dim: int, traj_data: np.ndarray, label_base: str) -> None:
        t, R, P = traj_data[:, 0], traj_data[:, 1], traj_data[:, 2]
        adiabatic_populations = traj_data[:, 3:3+dim]
        diabatic_populations = traj_data[:, 3+dim:3+2*dim]
        KE, PE = traj_data[:, -2], traj_data[:, -1]
        
        self.plot_nuc(t, R, P, label_base)
        self.plot_energy(t, KE, PE, label_base)
        self.plot_pop(t, adiabatic_populations, diabatic_populations, dim, label_base)
        
    def plot_nuc(self, t: np.ndarray, R: np.ndarray, P: np.ndarray, label: str) -> None:
        self.axs_nuc[0].plot(t, R, label=label)
        self.axs_nuc[1].plot(t, P, label=label)
        
    def plot_energy(self, t: np.ndarray, KE: np.ndarray, PE: np.ndarray, label: str) -> None:
        self.axs_energy[0].plot(t, KE, label=label)
        self.axs_energy[1].plot(t, PE, label=label)
        self.axs_energy[2].plot(t, KE+PE, label=label)
        
    def plot_pop(self, t: np.ndarray, adiabatic_populations: np.ndarray, diabatic_populations: np.ndarray, dim: int, label: str) -> None:
        for ii in range(dim):
            self.axs_pop[0].plot(t, diabatic_populations[:, ii], label=f'{label} {ii}')
            self.axs_pop[1].plot(t, adiabatic_populations[:, ii], label=f'{label} {ii}')
            
    def finalize(self) -> None:
        for ax in self.axs_nuc:
            ax.legend()
        for ax in self.axs_energy:
            ax.legend()
        for ax in self.axs_pop:
            ax.legend()
        plt.show()
        