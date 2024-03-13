# %% 
import os
import time
import argparse
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes, TD_Methods
from pymddrive.models.nonadiabatic_hamiltonian import diabatic_to_adiabatic
from pymddrive.integrators.state import State

from pymddrive.dynamics import BasisRepresentation, QunatumRepresentation, NonadiabaticDynamicsMethods, NumericalIntegrators    
from pymddrive.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics, run_nonadiabatic_dynamics_ensembles
from pymddrive.dynamics.misc_utils import eval_nonadiabatic_hamiltonian

from tullyone_utils import *

def get_floquet_rho0(rho0: np.ndarray, NF: int):
    data = [rho0]
    indptr = np.concatenate((np.zeros(NF+1), np.ones(NF+1))).astype(int)
    indicies = np.array([NF])
    dimF = (2*NF+1) * rho0.shape[0]
    rho0_floquet_bsr = sp.bsr_matrix((data, indicies, indptr), shape=(dimF, dimF), dtype=np.complex128)
    return rho0_floquet_bsr.toarray()

# def get_floquet_rho0_fully_sampled(rho0: np.ndarray, NF: int):
#     return sp.block_diag((rho0, ) * (2*NF+1)).toarray()

def get_block_representation(matrix: np.ndarray, m: int, n: int) -> np.ndarray:
    """Block representation of a square matrix.

    Args:
        matrix (np.ndarray): a square matrix of shape (m*n, m*n)
        m (int): the dimension of the block square matrix
        n (int): the dimension of the matrix elements in the block square matrix

    Returns:
        np.ndarray: a block square matrix of shape (m, m, n, n)
    """
    return matrix.reshape(m, n, m, n).swapaxes(1, 2)
    

def stop_condition(t, s, states):
    r, _, _ = s.get_variables()
    return outside_boundary(r, (-10, 10))

def break_condition(t, s, states):
    R = np.array(states['R'])
    return is_trapped(R, r_TST=0.0, recross_tol=10)

def estimate_floquet_levels(Intensity: float, Omega: float):
    ratio = Intensity / Omega
    if ratio < 1.0:
        return 2
    elif ratio < 2.0:
        return 3
    elif ratio < 3.0:
        return 5
    else:
        raise ValueError(f"Cannot estimate the number of Floquet levels for the given ratio: {ratio}.")
    
def run_tullyone_pulsed(
    r0: float, 
    p0: float, 
    Omega: float, 
    tau: float, 
    pulse_type: TullyOnePulseTypes,
    mass: float = 2000, 
    qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
    basis_rep: BasisRepresentation = BasisRepresentation.Diabatic,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
):
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0
    
    # To estimate the delay time
    _delay = estimate_delay_time(A, B, C, D, p0)
    
    # initialize the model and states
    hamiltonian = get_tullyone(
        t0=_delay, Omega=Omega, tau=tau,
        pulse_type=pulse_type, td_method=TD_Methods.BRUTE_FORCE
    )
    pulse = hamiltonian.pulse
    
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    dyn = NonadiabaticDynamics(
        hamiltonian=hamiltonian,
        t0=0.0,
        s0=s0,
        mass=mass,
        basis_rep=basis_rep,
        qm_rep=qm_rep,
        solver=solver,
        numerical_integrator=integrator,
        dt=0.03,
        save_every=30,
    )
    
    output = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    
    return output, pulse

def run_tullyone_pulsed_floquet(
    r0: float, 
    p0: float, 
    Omega: float, 
    tau: float, 
    pulse_type: TullyOnePulseTypes,
    NF: int = 2,
    mass: float = 2000, 
    qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
    basis_rep: BasisRepresentation = BasisRepresentation.Diabatic,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
):
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0
    
    # To estimate the delay time
    _delay = estimate_delay_time(A, B, C, D, p0)
        
    hamiltonian = get_tullyone(
        t0=_delay, Omega=Omega, tau=tau,
        pulse_type=pulse_type, td_method=TD_Methods.FLOQUET, NF=NF
    )
    
    pulse = hamiltonian.pulse
    if basis_rep == BasisRepresentation.Diabatic: 
        rho0_floquet = get_floquet_rho0(np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128), NF)
        # rho0_floquet = get_floquet_rho0_fully_sampled(np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128), NF)
        s0 = State.from_variables(R=r0, P=p0, rho=rho0_floquet)
    elif basis_rep == BasisRepresentation.Adiabatic:
        rho0_flouqet_diabatic = get_floquet_rho0(np.array([[1, 0], [0, 0.0]], dtype=np.complex128), NF)
        # rho0_floquet_diabatic = get_floquet_rho0_fully_sampled(np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128), NF)
        hami_return = eval_nonadiabatic_hamiltonian(0.0, np.array([r0]), hamiltonian, basis_rep)
        rho0_flouqet_adiabatic = diabatic_to_adiabatic(rho0_flouqet_diabatic, hami_return.evecs)
        # rho0_floquet_diabatic = get_floquet_rho0_fully_sampled(np.array([[1, 0], [0, 0.0]], dtype=np.complex128), NF)
        # hami_return = eval_nonadiabatic_hamiltonian(0.0, np.array([r0]), hamiltonian, BasisRepresentation.Diabatic)
        # rho0_flouqet_adiabatic = diabatic_to_adiabatic(rho0_floquet_diabatic, hami_return.evecs)
        # print(f"{np.allclose(rho0_flouqet_adiabatic, rho0_flouqet_diabatic, 1e-6)=}")
        s0 = State.from_variables(R=r0, P=p0, rho=rho0_flouqet_adiabatic)
       
    # initialize the integrator 
    dyn = NonadiabaticDynamics(
        hamiltonian=hamiltonian,
        t0=0.0,
        s0=s0,
        mass=mass,
        solver=solver,
        basis_rep=basis_rep,
        qm_rep=qm_rep,
        numerical_integrator=integrator,
        dt=0.03,
        save_every=30,
    )
    
    output = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    return output, pulse
    
def estimate_delay_time(A, B, C, D, p0, mass: float=2000.0):
    hamiltonian = get_tullyone(
        A=A, B=B, C=C, D=D,
        pulse_type=TullyOnePulseTypes.NO_PULSE
    )
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=-10.0, P=p0, rho=rho0)
    dyn = NonadiabaticDynamics(
        hamiltonian=hamiltonian,
        t0=0.0,
        s0=s0,
        mass=mass,
        basis_rep=BasisRepresentation.Diabatic,
        qm_rep=QunatumRepresentation.DensityMatrix,
        solver=NonadiabaticDynamicsMethods.EHRENFEST,
        numerical_integrator=NumericalIntegrators.ZVODE,
        dt=1,
        save_every=5
    )
    def stop_condition(t, s, states):
        r, p, _ = s.get_variables()
        return (r>0.0) or (p<0.0)
    break_condition = lambda t, s, states: False
    res = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    return res['time'][-1]    

def generate_ensembles(
    initial_diabatic_states: int,
    R0_samples: ArrayLike,
    P0_samples: ArrayLike,
    _delay: float,
    Omega: float,
    tau: float,
    NF: int=1,
    mass: float=2000,
    pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.PULSE_TYPE2,
    qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
    basis_rep: BasisRepresentation = BasisRepresentation.Adiabatic,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.FSSH,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
) -> Tuple[NonadiabaticDynamics]:
    # initialize the electronic states
    assert (initial_diabatic_states >= 0) and (initial_diabatic_states < 2), f"Valid states should be between 0 and 1. Got {initial_diabatic_states}."
    rho0_diabatic = np.zeros((2, 2), dtype=np.complex128)
    rho0_diabatic[initial_diabatic_states, initial_diabatic_states] = 1.0
    rho0_diabatic_floquet = get_floquet_rho0(rho0_diabatic, NF)
    # rho0_diabatic_floquet = get_floquet_rho0_fully_sampled(rho0_diabatic, NF)   
    
    assert (n_samples := len(R0_samples)) == len(P0_samples), "The number of R0 and P0 samples should be the same."
    ensemble = ()
    for ii, (R0, P0) in enumerate(zip(R0_samples, P0_samples)):
        hamiltonian = get_tullyone(
            t0=_delay, Omega=Omega, tau=tau,
            pulse_type=pulse_type, td_method=TD_Methods.FLOQUET, NF=NF 
        ) 
        if basis_rep == BasisRepresentation.Diabatic:
            s0 = State.from_variables(R=R0, P=P0, rho=rho0_diabatic)
        else:
            hami_return = eval_nonadiabatic_hamiltonian(0, np.array([R0]), hamiltonian, basis_rep=BasisRepresentation.Diabatic)
            evecs = hami_return.evecs
            ## print(f"{evecs.shape=}")
            # rho0_adiabatic = evecs.T.conj() @ rho0_diabatic @ evecs
            rho0_adiabatic = evecs.T.conj() @ rho0_diabatic_floquet @ evecs
            s0 = State.from_variables(R=R0, P=P0, rho=rho0_adiabatic)
        dyn = NonadiabaticDynamics(
            hamiltonian=hamiltonian,
            t0=0.0,
            s0=s0,
            mass=mass,
            basis_rep=basis_rep,
            qm_rep=qm_rep,
            solver=solver,
            numerical_integrator=integrator,
            dt=0.1,
            save_every=1,
        )
        ensemble += (dyn,)
    return ensemble

def run_tullyone_pulsed_ensemble( 
    r0: float, 
    p0: float, 
    Omega: float, 
    tau: float, 
    pulse_type: TullyOnePulseTypes,
    n_samples: int = 100,
    NF: int = 1,
    mass: float = 2000, 
    qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
    basis_rep: BasisRepresentation = BasisRepresentation.Adiabatic,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.FSSH,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
):
    A, B, C, D = 0.01, 1.6, 0.005, 1.0
    R0_samples = np.array([r0]*n_samples)
    P0_samples = np.array([p0]*n_samples)
    _delay = estimate_delay_time(A, B, C, D, p0)
    emsemble = generate_ensembles(
        initial_diabatic_states=0,
        R0_samples=R0_samples,
        P0_samples=P0_samples,
        _delay=_delay,
        Omega=Omega,
        tau=tau,
        NF=NF,
        mass=mass,
        pulse_type=pulse_type,
        qm_rep=qm_rep,
        basis_rep=basis_rep,
        solver=solver,
        integrator=integrator,
    )
    pulse = emsemble[0].hamiltonian.pulse
    
    return run_nonadiabatic_dynamics_ensembles(emsemble, stop_condition, break_condition, inhomogeneous=True), pulse

# plot utilities
class AppendableFigure:
    def __init__(self, fig=None, ax=None, gs=None, axs=None, name_prefix=None):
        self.fig = fig
        self.ax = ax
        self.gs = gs
        self.axs = axs
        self.lines = []
        self.labels = []
        self.name_prefix: str = name_prefix
        
    def add_plot(self, x, y, **kwargs):
        line, = self.ax.plot(x, y, **kwargs)
        self.lines.append(line)
        self.labels.append(kwargs.get('label', None))
        
    def show(self, legend_each=False):
        if self.ax is not None:
            self.ax.legend(self.lines, self.labels)
            
        if self.axs is not None:
            if legend_each:
                for ax in self.axs:
                    ax.legend()
            else:
                # choose the last axis to represent the legend
                ax = self.axs[-1]
                fig = self.fig
                # self.fig.legend(axs[2].get_legend_handles_labels()[0], axs[2].get_legend_handles_labels()[1], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3)
                fig.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1], loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
        self.fig.tight_layout()
        # plt.show()
    
    def set_xlim(self, xlim):
        if self.ax is not None:
            self.ax.set_xlim(xlim)
        if self.axs is not None:
            for ax in self.axs:
                ax.set_xlim(xlim)
                
    def savefig(self, **kwargs):
        if self.name_prefix is None:
            import warnings
            warnings.warn(f"cannot save a AppendableFigure without a name prefix.")
            return None
        fname_str = self.name_prefix
        for key, val in kwargs.items():
            fname_str += f"_{key}-{val}" 
        fname_str += ".pdf"
        self.fig.savefig(fname_str)

class FigurePulse(AppendableFigure):
    def __init__(self):
        name_prefix = 'pulse'
        fig = plt.figure(dpi=300, figsize=(3.5, 2.5))
        ax = fig.add_subplot(111)
        super().__init__(fig, ax, name_prefix=name_prefix)
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Pulse')
        
class FigureNuclearDynamics(AppendableFigure):
    def __init__(self):
        name_prefix = 'nuclear'
        
        fig = plt.figure(dpi=300, figsize=(3.5*2, 2.5))
        gs = fig.add_gridspec(1, 2)
        axs = gs.subplots().flatten()
        super().__init__(fig, gs=gs, axs=axs, name_prefix=name_prefix)
        ylabels = ['R', 'P']
        for ax, ylabel in zip(axs, ylabels):
            ax.set_xlabel('Time')
            ax.set_ylabel(ylabel)
            
    def add_plot(self, output: dict, **kwargs):
        time: ArrayLike = output['time']
        R: ArrayLike = output['states']['R']
        P: ArrayLike = output['states']['P']
        self.axs[0].plot(time, R, **kwargs)
        self.axs[1].plot(time, P, **kwargs)
            
class FigurePopulations(AppendableFigure):
    def __init__(self):
        name_prefix = 'populations'
        
        fig = plt.figure(dpi=300, figsize=(3.5*2, 2.5))
        gs = fig.add_gridspec(1, 2)
        axs = gs.subplots().flatten()
        super().__init__(fig, gs=gs, axs=axs, name_prefix=name_prefix)
        ylabels = [r'$P_0$', r'$P_1$']
        for ax, ylabel in zip(axs, ylabels):
            ax.set_xlabel('Time')
            ax.set_ylabel(ylabel)
            
    def add_plot(self, output: dict, **kwargs):
        time: ArrayLike = output['time']
        adiab_populations: ArrayLike = output['adiab_populations']
        self.axs[0].plot(time, adiab_populations[:, 0], **kwargs)
        self.axs[1].plot(time, adiab_populations[:, 1], **kwargs)  
        
class FigureEnergetics(AppendableFigure):
    def __init__(self,):
        name_prefix = 'energetics'
        
        fig = plt.figure(dpi=300, figsize=(3.5*3, 2.5))
        gs = fig.add_gridspec(1, 3)
        axs = gs.subplots().flatten()
        super().__init__(fig, gs=gs, axs=axs, name_prefix=name_prefix)
        axs[0].set_ylabel('KE')
        axs[1].set_ylabel('PE')
        axs[2].set_ylabel(r'$\Delta$ TE')
        
        ylabels = ['KE', 'PE', r'$\Delta$ TE']
        
        for ax, ylabel in zip(axs, ylabels):
            ax.set_xlabel('Time')
            ax.set_ylabel(ylabel) 
            
    def add_plot(self, output: dict, **kwargs):
        time: ArrayLike = output['time']
        KE: ArrayLike = output['KE']
        PE: ArrayLike = output['PE']
        TE: ArrayLike = KE + PE
        Delta_TE = TE - TE[0]
        self.axs[0].plot(time, KE, **kwargs)
        self.axs[1].plot(time, PE, **kwargs)
        self.axs[2].plot(time, Delta_TE, **kwargs)
    
def main(
    Omega: float, 
    tau: float, 
    NF: int=1, 
    pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.PULSE_TYPE2,
    n_fssh_traj: int = 16
) -> tuple:  
    """ Compare diabatic Ehrenfest, diabatic and adiabatic Floquet Ehrenfest. """
    # prepare the figures
    fig_pulse = FigurePulse()
    fig_nuclear_dynamics = FigureNuclearDynamics()
    fig_populations = FigurePopulations()
    fig_energy = FigureEnergetics()
    
    figs = (fig_pulse, fig_nuclear_dynamics, fig_populations, fig_energy)
    
    def update_figs(output, pulse, label, ls):
        fig_pulse.add_plot(output['time'], [pulse(t) for t in output['time']], label=label, linestyle=ls)
        fig_nuclear_dynamics.add_plot(output, label=label, linestyle=ls)
        fig_populations.add_plot(output, label=label, linestyle=ls)
        fig_energy.add_plot(output, label=label, linestyle=ls)
        
        for fig in figs:
            fig.set_xlim((400, 1000))
        
    def show_figs():
        fig_pulse.show()
        fig_nuclear_dynamics.show()
        fig_populations.show()
        fig_energy.show()
        # for fig in figs:
        #     fig.fig.tight_layout()
        # for fig in figs:
        plt.show()
        
    def save_figs():
        for fig in figs:
            fig.savefig(Omega=Omega, tau=tau, NF=NF, ntraj=n_fssh_traj)
    
    r0 = -10.0
    p0 = 30
    print("====================================================", flush=True)
    start = time.perf_counter() 
    output_d, pulse_d = run_tullyone_pulsed(r0, p0, Omega, tau, pulse_type)
    update_figs(output_d, pulse_d, 'Ehrenfest Diabatic', '-')
    print(f"Time elapsed for Ehrenfest Diabatic is {time.perf_counter()-start:.5f} seconds.", flush=True)
    print("====================================================", flush=True)
    print("====================================================", flush=True)
    
    start = time.perf_counter()
    output_floq_diabatic, pulse_f_diabatic = run_tullyone_pulsed_floquet(r0, p0, Omega, tau, 
                                                                         pulse_type=pulse_type, NF=NF, 
                                                                         basis_rep=BasisRepresentation.Diabatic)
    update_figs(output_floq_diabatic, pulse_f_diabatic, 'Floquet Ehrenfest Diabatic', '--')
    print(f"Time elapsed for Floquet Ehrenfest Diabatic is {time.perf_counter()-start:.5f} seconds.", flush=True)
    print("====================================================", flush=True)
    print("====================================================", flush=True)
    
    start = time.perf_counter()
    output_floq_adiabatic, pulse_f_adiabatic = run_tullyone_pulsed_floquet(r0, p0, Omega, tau, 
                                                                           pulse_type=pulse_type, NF=NF, 
                                                                           basis_rep=BasisRepresentation.Adiabatic)
    update_figs(output_floq_adiabatic, pulse_f_adiabatic, 'Floquet Ehrenfest Adiabatic', '-.')
    print(f"Time elapsed for Floquet Ehrenfest Adiabatic is {time.perf_counter()-start:.5f} seconds.", flush=True)
    print("====================================================", flush=True)
    
    start = time.perf_counter()
    output_floq_fssh, pulse_floq_fssh = run_tullyone_pulsed_ensemble(r0, p0, Omega, tau, pulse_type=pulse_type, 
                                                                     n_samples=n_fssh_traj, NF=NF, 
                                                                     basis_rep=BasisRepresentation.Adiabatic)
    update_figs(output_floq_fssh, pulse_floq_fssh, 'Floquet FSSH Adiabatic', ':')
    print(f"Time elapsed for Floquet FSSH Adiabatic is {time.perf_counter()-start:.5f} seconds.", flush=True)
    print("====================================================", flush=True)
    
    show_figs() 
    save_figs()
    
# %%Figure,  
if __name__ == "__main__":
    Omega = 0.3; tau = 100; NF=1
    pulse_type = TullyOnePulseTypes.PULSE_TYPE3  
    n_fssh_traj = 2
    main(Omega, tau, NF=NF, pulse_type=pulse_type, n_fssh_traj=n_fssh_traj)
    

# %%
