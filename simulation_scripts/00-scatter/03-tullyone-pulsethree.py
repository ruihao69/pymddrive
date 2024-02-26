# %% 
import os
import time
import numpy as np
from pymddrive.models.tully_w_pulse import TullyOne
from pymddrive.models.tully_w_pulse import TullyOnePulseThree
from pymddrive.pulses.morlet import MorletReal
from pymddrive.integrators.state import State
from pymddrive.dynamics.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics 

from tullyone_utils import *

import sys

import argparse

def stop_condition(t, s, states):
    r, _, _ = s.get_variables()
    return outside_boundary(r, (-10, 10))

def break_condition(t, s, states):
    R = np.array(states['R'])
    return is_trapped(R, r_TST=0.0, recross_tol=10)

def run_tullyone_pulsethree(r0, p0, Omega, tau, mass=2000, solver='Ehrenfest', method='vv_rk4'):
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0
    
    # To estimate the delay time
    start = time.time()
    _delay = estimate_delay_time(A, B, C, D, p0)
    print(f"Time elapsed for estimating the delay time is {time.time()-start:.5f} seconds.", flush=True)
    
    # initialize the model and states
    # pulse = MorletReal(A=C, t0=_delay, tau=tau, Omega=Omega, phi=0)
    pulse = MorletReal(A=C/2.0, t0=_delay, tau=tau, Omega=Omega, phi=0)
    
    # model = TullyOnePulseOne(A, B, pulse=pulse)
    model = TullyOnePulseThree(A, B, C/2.0, D, pulse=pulse)
    
    
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State(r0, p0, rho0)
    
    # initialize the integrator 
    t_bounds = (pulse.t0 - pulse.tau, pulse.t0 + pulse.tau)
    
    dyn = NonadiabaticDynamics(
        model=model,
        t0=0.0,
        s0=s0,
        mass=mass,
        solver=solver,
        method=method,
        r_bounds=(-10, 10),
        t_bounds=t_bounds
    )
    output = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    
    return output, pulse
    

def estimate_delay_time(A, B, C, D, p0, mass: float=2000.0):
    model = TullyOne(A, B, C, D)
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State(-5.0, p0, rho0)
    dyn = NonadiabaticDynamics(
        model=model,
        t0=5.0/(p0/mass),
        s0=s0,
        mass=2000,
        solver='Ehrenfest',
        method='vv_rk4',
        r_bounds=(-10, 10)
    )
    dyn.dt = dyn.dt * 3.0
    def stop_condition(t, s, states):
        r, p, _ = s.get_variables()
        return (r>0.0) or (p<0.0)
    break_condition = lambda t, s, states: False
    res = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    return res['time'][-1] 

def main(sim_signature: str, n_samples: int, Omega, tau, p_bounds: tuple=(0.5, 35.0)):
    from pararun import ParaRunScatter
    if not os.path.exists(sim_signature):
        os.makedirs(sim_signature)
        
    r0 = -10.0
    _r0_list = np.array([r0]*n_samples)
    _p0_list = linspace_log10(*p_bounds, n_samples)
    _Omega_list = np.array([Omega]*n_samples)
    _tau_list = np.array([tau]*n_samples)
    
    runner = ParaRunScatter(r0=_r0_list, p0=_p0_list, Omega=_Omega_list, tau=_tau_list)
    
    res_gen = runner.run(run_tullyone_pulsethree, accumulate_output, sim_signature)
    traj_dict, pulses = accumulate_output(_p0_list, res_gen)
    
    post_process_output(sim_signature, traj_dict, pulse_list=pulses)
    
# %% 
if __name__ == "__main__":
    desc = "The parser for TullyOne with Pulse Three"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--Omega', type=float, help='The Omega value')
    parser.add_argument('--tau', type=float, help='The tau value')
    args = parser.parse_args() 
    
    Omega, tau= args.Omega, args.tau 
    
    sim_signature = f"data_tullyone_pulsethree-Omega-{Omega}-tau-{tau}"
    nsamples = 48
    p_bounds = (0.5, 35)
    main(sim_signature, nsamples, Omega, tau, p_bounds)
    
    
# %%
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     output, pulse = run_tullyone_pulsethree(r0=-10, p0=10, Omega=0.1, tau=100)
#     times = output['time']
#     R = output['states']['R']
#     P = output['states']['P']
#     rho = output['states']['rho']
#     KE = output['KE']
#     PE = output['PE']
    
#     plt.plot(times, R)
#     plt.show()

#     plt.plot(times, P)
#     plt.show()
    
#     plt.plot(times, rho[:, 0, 0].real, label='rho_00')
#     plt.plot(times, rho[:, 1, 1].real, label='rho_11')
#     plt.show()
    
#     plt.plot(times, KE)
#     plt.show()
    
    
#     plt.plot(times, PE)
#     plt.show()
#     
#     pp = [pulse(t) for t in times]
#     plt.plot(times, pp)