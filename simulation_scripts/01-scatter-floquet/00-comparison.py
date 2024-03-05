# %% 
import os
import time
import argparse

import numpy as np
import scipy.sparse as sp

from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes, TD_Methods
from pymddrive.models.nonadiabatic_hamiltonian import diabatic_to_adiabatic
from pymddrive.integrators.state import State

from pymddrive.dynamics.options import BasisRepresentation, QunatumRepresentation, NonadiabaticDynamicsMethods, NumericalIntegrators    
from pymddrive.dynamics.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics 
from pymddrive.dynamics._misc import eval_nonadiabatic_hamiltonian

from tullyone_utils import *

def get_floquet_rho0(rho0: np.ndarray, NF: int):
    data = [rho0]
    indptr = np.concatenate((np.zeros(NF+1), np.ones(NF+1))).astype(int)
    indicies = np.array([NF])
    dimF = (2*NF+1) * rho0.shape[0]
    rho0_floquet_bsr = sp.bsr_matrix((data, indicies, indptr), shape=(dimF, dimF), dtype=np.complex128)
    return rho0_floquet_bsr.toarray()

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
    start = time.perf_counter()
    _delay = estimate_delay_time(A, B, C, D, p0)
    print(f"Time elapsed for estimating the delay time is {time.perf_counter()-start:.5f} seconds.", flush=True)
    
    # initialize the model and states
    hamiltonian = get_tullyone(
        t0=_delay, Omega=Omega, tau=tau,
        pulse_type=pulse_type, td_method=TD_Methods.BRUTE_FORCE
    )
    pulse = hamiltonian.pulse
    
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    
    # initialize the integrator 
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
        save_every=30
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
    start = time.perf_counter()
    _delay = estimate_delay_time(A, B, C, D, p0)
    print(f"Time elapsed for estimating the delay time is {time.perf_counter()-start:.5f} seconds.", flush=True)
    # initialize the model and states
    estimated_NF = estimate_floquet_levels(Omega, tau)
    if NF < estimated_NF:
        print(f"Warning: The number of Floquet levels is estimated to be {estimated_NF} for the given Omega and tau. The given NF is {NF}.", flush=True)
        
    hamiltonian = get_tullyone(
        t0=_delay, Omega=Omega, tau=tau,
        pulse_type=pulse_type, td_method=TD_Methods.FLOQUET, NF=NF
    )
    
    pulse = hamiltonian.pulse
    if basis_rep == BasisRepresentation.Diabatic: 
        rho0_flouqet = get_floquet_rho0(np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128), NF)
        s0 = State.from_variables(R=r0, P=p0, rho=rho0_flouqet)
    elif basis_rep == BasisRepresentation.Adiabatic:
        rho0_flouqet_diabatic = get_floquet_rho0(np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128), NF)
        hami_return = eval_nonadiabatic_hamiltonian(0.0, np.array([r0]), hamiltonian, basis_rep)
        rho0_flouqet_adiabatic = diabatic_to_adiabatic(rho0_flouqet_diabatic, hami_return.evecs)
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
        save_every=30
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
        save_every=1
    )
    def stop_condition(t, s, states):
        r, p, _ = s.get_variables()
        return (r>0.0) or (p<0.0)
    break_condition = lambda t, s, states: False
    res = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    return res['time'][-1]    
    
def main(
    Omega: float, 
    tau: float, 
    NF: int=1, 
    pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.PULSE_TYPE3
) -> tuple:  
    """ Compare diabatic Ehrenfest, diabatic and adiabatic Floquet Ehrenfest. """
    r0 = -10.0
    p0 = 10.0
    
    output_d, pulse_d = run_tullyone_pulsed(r0, p0, Omega, tau, pulse_type)
    output_floq_diabatic, pulse_f_diabatic = run_tullyone_pulsed_floquet(r0, p0, Omega, tau, pulse_type=pulse_type, NF=NF, basis_rep=BasisRepresentation.Diabatic)
    output_floq_adiabatic, pulse_f_adiabatic = run_tullyone_pulsed_floquet(r0, p0, Omega, tau, pulse_type=pulse_type, NF=NF, basis_rep=BasisRepresentation.Adiabatic)
    
    return pulse_d, output_d, pulse_f_adiabatic, output_floq_adiabatic, pulse_f_diabatic, output_floq_diabatic
    
    

def compute_reduced_rho(rho_F: np.ndarray, dim_sys: int=2):
    assert rho_F.shape[0] % dim_sys == 0
    dim_F = rho_F.shape[0] // dim_sys
    assert dim_F % 2 == 1
    NF = (dim_F - 1) // 2
    
    rho_block = get_block_representation(rho_F, dim_F, dim_sys)
    rho_reduced = np.zeros((dim_sys, dim_sys), dtype=np.complex128)
    for i in range(dim_F):
        rho_reduced += rho_block[i, i]
    return rho_reduced

def plot_all(pulse_d, output_d, pulse_fa, output_fa, pulse_fd, output_fd):
    import matplotlib.pyplot as plt
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    time_d = output_d['time']
    time_fa = output_fa['time']
    time_fd = output_fd['time']
    
    ax.plot(time_d,  [pulse_d (t) for t in time_d], label='Ehrenfest Diabatic')
    ax.plot(time_fa, np.abs([pulse_fa(t) for t in time_fa]), label='Floquet Ehrenfest Adiabatic')
    ax.plot(time_fd, np.abs([pulse_fd(t) for t in time_fd]), label='Floquet Ehrenfest Diabatic')
    ax.set_xlabel('Time')
    ax.set_ylabel('Pulse')
    
    ax.legend()
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time_d, output_d['states']['R'], label='Ehrenfest Diabatic')
    ax.plot(time_fa, output_fa['states']['R'], label='Floquet Ehrenfest Adiabatic')
    ax.plot(time_fd, output_fd['states']['R'], label='Floquet Ehrenfest Diabatic')
    ax.set_xlabel('Time')
    ax.set_ylabel('R')
    ax.legend()
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time_d, output_d['states']['P'], label='Ehrenfest Diabatic')
    ax.plot(time_fa, output_fa['states']['P'], label='Floquet Ehrenfest Adiabatic')
    ax.plot(time_fd, output_fd['states']['P'], label='Floquet Ehrenfest Diabatic')
    ax.set_xlabel('Time')
    ax.set_ylabel('P')
    ax.legend()
    plt.show()
    
    dimF = output_fd['states']['rho'][0].shape[0] // output_d['states']['rho'][0].shape[0]
    NF = (dimF - 1) // 2
    
    floquet_offsets = np.arange(-NF, NF+1).astype(int)
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    pop_fa_dim0 = sum(output_fa['populations'].T[:2*NF+1])
    pop_fa_dim1 = sum(output_fa['populations'].T[2*NF+1:])
    pop_fa = np.array([pop_fa_dim0, pop_fa_dim1]).T
    pop_fd_dim0 = sum(output_fd['populations'].T[:2*NF+1])
    pop_fd_dim1 = sum(output_fd['populations'].T[2*NF+1:])
    pop_fd = np.array([pop_fd_dim0, pop_fd_dim1]).T
    for i in range(output_d['populations'].shape[1]):
        ax.plot(time_d, output_d['populations'][:, i], label='Ehrenfest Diabatic')
        ax.plot(time_fa, pop_fa[:, i], label='Floquet Ehrenfest Adiabatic')
        ax.plot(time_fd, pop_fd[:, i], label='Floquet Ehrenfest Diabatic')
    ax.legend()
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time_d, output_d['KE'], label='Ehrenfest Diabatic')
    ax.plot(time_fa, output_fa['KE'], label='Floquet Ehrenfest Adiabatic')
    ax.plot(time_fd, output_fd['KE'], label='Floquet Ehrenfest Diabatic')
    ax.set_xlabel('Time')
    ax.set_ylabel('KE')
    ax.legend()
    plt.show() 
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time_d, output_d['PE'], label='Ehrenfest Diabatic')
    ax.plot(time_fa, output_fa['PE'], label='Floquet Ehrenfest Adiabatic')
    ax.plot(time_fd, output_fd['PE'], label='Floquet Ehrenfest Diabatic')
    ax.set_xlabel('Time')
    ax.set_ylabel('PE')
    ax.legend()
    plt.show()
    
    def get_TE(PE, KE):
        return np.array(PE) + np.array(KE)
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time_d, get_TE(output_d['KE'], output_d['PE']), label='Ehrenfest Diabatic')
    ax.plot(time_fa, get_TE(output_fa['KE'], output_fa['PE']), label='Floquet Ehrenfest Adiabatic')
    ax.plot(time_fd, get_TE(output_fd['PE'], output_fd['KE']), label='Floquet Ehrenfest Diabatic')
    ax.set_xlabel('Time')
    ax.set_ylabel('TE')
    ax.legend()
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time_d, output_d['meanF'], label='Ehrenfest Diabatic')
    ax.plot(time_fa, output_fa['meanF'], label='Floquet Ehrenfest Adiabatic')
    ax.plot(time_fd, output_fd['meanF'], label='Floquet Ehrenfest Diabatic')
    ax.set_xlabel('Time')
    ax.set_ylabel('meanF')
    ax.legend()
    plt.show()
    
# %%Figure,  
if __name__ == "__main__":
    Omega = 0.3; tau = 100
    pulse_d, output_d, pulse_fd, output_fd, pulse_fa, output_fa = main(Omega, tau, NF=1, pulse_type=TullyOnePulseTypes.PULSE_TYPE3)

# %%
if __name__ == "__main__":
    # plot_all(pulse_d, output_d, pulse_f, output_floquet)
    # plot_all(pulse_d, output_d, pulse_a, output_a, pulse_f, output_floquet)
    plot_all(pulse_d, output_d, pulse_fa, output_fa, pulse_fd, output_fd)

# %%
