""" Conclusion: for a fast varying Hamiltonian, the adiabatic representation is not suitable."""
""" Please use diabatic representation for fast varying Hamiltonian. """
""" As a consequence, the surface hopping algorithm is not suitable for fast varying Hamiltonian. """
# %% 
import os
import time

import numpy as np
from numbers import Real

from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes, TD_Methods
from pymddrive.integrators.state import State
from pymddrive.dynamics.options import BasisRepresentation, QunatumRepresentation, NonadiabaticDynamicsMethods, NumericalIntegrators
from pymddrive.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics 

from tullyone_utils import *

def stop_condition(t, s, states):
    r, _, _ = s.get_variables()
    return outside_boundary(r, (-10, 10))

def break_condition(t, s, states):
    R = np.array(states['R'])
    return is_trapped(R, r_TST=0.0, recross_tol=10)

def run_tullyone_pulsed(
    r0: float, 
    p0: float, 
    Omega: float, 
    tau: float, 
    pulse_type: TullyOnePulseTypes,
    mass: float = 2000, 
    qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
    basis_rep: BasisRepresentation = BasisRepresentation.Adiabatic,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
    _delay: float = None
):
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0
    
    # To estimate the delay time
    if _delay is None:
        start = time.perf_counter()
        _delay = estimate_delay_time(A, B, C, D, p0)
        print(f"Time elapsed for estimating the delay time is {time.perf_counter()-start:.5f} seconds.", flush=True)
    elif not isinstance(_delay, Real):
        raise ValueError(f"Invalid value for _delay: {_delay}")
        
    
    # initialize the model and states
    hamiltonian = get_tullyone(
        t0=_delay, Omega=Omega, tau=tau,
        pulse_type=pulse_type, td_method=TD_Methods.BRUTE_FORCE
    )
    pulse = None if pulse_type==TullyOnePulseTypes.NO_PULSE else hamiltonian.pulse
    
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    
    T = 2.0 * np.pi / Omega
    max_dt = 1/8 * T
    
    
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
        save_every=30,
        max_step=max_dt
    )
    
    output = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    
    return output, pulse
    

def estimate_delay_time(A, B, C, D, p0, mass: float=2000.0):
    # model = TullyOne(A, B, C, D)
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
        save_every=10
    )
    def stop_condition(t, s, states):
        r, p, _ = s.get_variables()
        return (r>0.0) or (p<0.0)
    break_condition = lambda t, s, states: False
    res = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    return res['time'][-1] 

def run(Omega: float, tau: float, pt: TullyOnePulseTypes):
    r0 = -10.0
    p0 = 30.0
    
    _delay = None 
    res_diab, pulse_diab = run_tullyone_pulsed(r0, p0, Omega, tau, pt, basis_rep=BasisRepresentation.Diabatic, _delay=_delay) 
    if pt==TullyOnePulseTypes.NO_PULSE or pt==TullyOnePulseTypes.UNITPULSE or pt==TullyOnePulseTypes.ZEROPULSE:
        _delay = None
    else:
        _delay = pulse_diab.t0
    res_adiab, pulse_adiab = run_tullyone_pulsed(r0, p0, Omega, tau, pt, basis_rep=BasisRepresentation.Adiabatic, _delay=_delay)
    return res_diab, res_adiab, pulse_diab, pulse_adiab
    
  
def plot_all(res_diab, res_adiab, pulse_diab, pulse_adiab):
    import matplotlib.pyplot as plt
    # compare the pulses
    time_diab = res_diab['time']; time_adiab = res_adiab['time']
    if pulse_diab is not None and pulse_adiab is not None:
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        ax.plot(time_diab, [pulse_diab(t) for t in time_diab], label="Diabatic")
        ax.plot(time_adiab, [pulse_adiab(t) for t in time_adiab], ls='--', label="Adiabatic")
        # ax.set_xlim(1000, 2000)
        ax.set_xlabel("Time")
        ax.set_ylabel("Pulse")
        ax.legend()
        plt.show()
    
    # compare the trajectories
    R_diab = res_diab['states']['R']; R_adiab = res_adiab['states']['R']
    P_diab = res_diab['states']['P']; P_adiab = res_adiab['states']['P']
    fig = plt.figure(figsize=(10, 5), dpi=300)
    ax = fig.add_subplot(121)
    ax.plot(time_diab, R_diab, label="Diabatic")
    ax.plot(time_adiab, R_adiab, ls='--', label="Adiabatic")
    # ax.set_xlim(1000, 2000)
    ax.set_xlabel("Time")
    ax.set_ylabel("R")
    ax.legend()
    ax = fig.add_subplot(122)
    ax.plot(time_diab, P_diab, label="Diabatic")
    ax.plot(time_adiab, P_adiab, ls='--', label="Adiabatic")
    # ax.set_xlim(1000, 2000)
    ax.set_xlabel("Time")
    ax.set_ylabel("P")
    ax.legend()
    plt.show()
    
    # compare the populations
    pop_diab = res_diab['populations']; pop_adiab = res_adiab['populations']
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time_diab, pop_diab[:, 0], label="Diabatic 1")
    ax.plot(time_diab, pop_diab[:, 1], label="Diabatic 2")
    ax.plot(time_adiab, pop_adiab[:, 0], ls='--', label="Adiabatic 1")
    ax.plot(time_adiab, pop_adiab[:, 1], ls='--', label="Adiabatic 2")
    # ax.set_xlim(1000, 2000)
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.legend()
    plt.show()
    
    # compare the energies
    KE_diab = res_diab['KE']; KE_adiab = res_adiab['KE']    
    PE_diab = res_diab['PE']; PE_adiab = res_adiab['PE']
    TE_diab = KE_diab + PE_diab; TE_adiab = KE_adiab + PE_adiab 
    fig = plt.figure(figsize=(15, 5), dpi=300)
    ax = fig.add_subplot(131)
    ax.plot(time_diab, KE_diab, label="Diabatic")
    ax.plot(time_adiab, KE_adiab, ls='--', label="Adiabatic")
    # ax.set_xlim(1000, 2000)
    ax.set_xlabel("Time")
    ax.set_ylabel("Kinetic Energy")
    ax.legend()
    ax = fig.add_subplot(132)
    ax.plot(time_diab, PE_diab, label="Diabatic")
    ax.plot(time_adiab, PE_adiab, ls='--', label="Adiabatic")
    # ax.set_xlim(1000, 2000)
    ax.set_xlabel("Time")
    ax.set_ylabel("Potential Energy")
    ax.legend()
    ax = fig.add_subplot(133)
    ax.plot(time_diab, TE_diab, label="Diabatic")
    ax.plot(time_adiab, TE_adiab, ls='--', label="Adiabatic")
    # ax.set_xlim(1000, 2000)
    ax.set_xlabel("Time")
    ax.set_ylabel("Total Energy")
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    # compare the forces
    meanF_diab = res_diab['meanF']; meanF_adiab = res_adiab['meanF']
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(time_diab, meanF_diab, label="Diabatic")
    ax.plot(time_adiab, meanF_adiab, ls='--', label="Adiabatic")
    # ax.set_xlim(1000, 2000)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Force")
    ax.legend()
    plt.show()
    
    
    
def main(Omega: float, tau: float, ):
    # test the TullyOne model with no pulse
    print("========================================")
    print("Running the TullyOne model with no pulse")
    print("========================================")
    res_diab, res_adiab, pulse_diab, pulse_adiab = run(Omega, tau, pt=TullyOnePulseTypes.NO_PULSE)
    plot_all(res_diab, res_adiab, pulse_diab, pulse_adiab)
    print("========================================")
    
    # test the TullyOne model with unit pulse
    # print("========================================")
    # print("Running the TullyOne model with unit pulse")
    # print("========================================")
    # res_diab, res_adiab, pulse_diab, pulse_adiab = run(Omega, tau, pt=TullyOnePulseTypes.UNITPULSE)
    # plot_all(res_diab, res_adiab, pulse_diab, pulse_adiab)
    # print("========================================")
    
    # test the TullyOne model with zero pulse
    # print("========================================")
    # print("Running the TullyOne model with unit pulse")
    # print("========================================")
    # res_diab, res_adiab, pulse_diab, pulse_adiab = run(Omega, tau, pt=TullyOnePulseTypes.ZEROPULSE)
    # plot_all(res_diab, res_adiab, pulse_diab, pulse_adiab)
    # print("========================================")

       
    # test the TullyOne model with pulse type I
    print("========================================")
    print("Running the TullyOne model with pulse type I")
    print("========================================")
    res_diab, res_adiab, pulse_diab, pulse_adiab = run(Omega, tau, pt=TullyOnePulseTypes.PULSE_TYPE1)
    plot_all(res_diab, res_adiab, pulse_diab, pulse_adiab)
    print("========================================")
    
    # test the TullyOne model with pulse type II
    print("========================================")
    print("Running the TullyOne model with pulse type II")
    print("========================================")
    res_diab, res_adiab, pulse_diab, pulse_adiab = run(Omega, tau, pt=TullyOnePulseTypes.PULSE_TYPE2)
    plot_all(res_diab, res_adiab, pulse_diab, pulse_adiab)
    print("========================================")

    
# %% 
if __name__ == "__main__":
    
    Omega, tau= 0.1, 100
    # res_diab, res_adiab, pulse_diab, pulse_adiab = main(Omega, tau, pulse_type)
    
    main(Omega, tau,)


# %%
