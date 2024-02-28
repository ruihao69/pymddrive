# %% 
import os
import time
import argparse

import numpy as np

from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes, TD_Methods
from pymddrive.integrators.state import State
from pymddrive.dynamics.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics 

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
    solver: str ='Ehrenfest', 
    method: str ='vv_rk4'
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
    model = get_tullyone(
        t0=_delay, Omega=Omega, tau=tau,
        pulse_type=pulse_type, td_method=TD_Methods.BRUTE_FORCE
    )
    pulse = model.pulse
    
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    
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
    # model = TullyOne(A, B, C, D)
    model = get_tullyone(
        A=A, B=B, C=C, D=D,
        pulse_type=TullyOnePulseTypes.NO_PULSE
    )
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=-5.0, P=p0, rho=rho0)
    dyn = NonadiabaticDynamics(
        model=model,
        t0=5.0/(p0/mass),
        s0=s0,
        mass=2000,
        solver='Ehrenfest',
        method='vv_rk4',
        r_bounds=(-10, 10),
        save_every=100
    )
    dyn.dt = dyn.dt * 3.0
    def stop_condition(t, s, states):
        r, p, _ = s.get_variables()
        return (r>0.0) or (p<0.0)
    break_condition = lambda t, s, states: False
    res = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    return res['time'][-1] 

def main(
    sim_signature: str, 
    n_samples: int, 
    Omega: float, 
    tau: float, 
    pt: TullyOnePulseTypes, 
    p_bounds: tuple=(0.5, 35.0)
):
    from pararun import ParaRunScatter
    if not os.path.exists(sim_signature):
        os.makedirs(sim_signature)
        
    r0 = -10.0
    _r0_list = np.array([r0]*n_samples)
    _p0_list = linspace_log10(*p_bounds, n_samples)
    _Omega_list = np.array([Omega]*n_samples)
    _tau_list = np.array([tau]*n_samples)
    _pulse_type_list = np.array([pt]*n_samples) 
    
    runner = ParaRunScatter(r0=_r0_list, p0=_p0_list, Omega=_Omega_list, tau=_tau_list, pulse_type=_pulse_type_list)
    
    res_gen = runner.run(run_tullyone_pulsed, accumulate_output, sim_signature)
    traj_dict, pulses = accumulate_output(_p0_list, res_gen)
    
    post_process_output(sim_signature, traj_dict, pulse_list=pulses)
    
# %% 
if __name__ == "__main__":
    
    desc = "The parser for TullyOne with Pulse One"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--Omega', type=float, help='The Omega value')
    parser.add_argument('--tau', type=float, help='The tau value')
    parser.add_argument('--pulse_type', type=int, help='The Pulse Type (1, 2, or 3)')
    args = parser.parse_args() 
    
    Omega, tau, pulse_type= args.Omega, args.tau, args.pulse_type
    
    if pulse_type == 1:
        pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.PULSE_TYPE1
        sim_signature = f"data_tullyone_pulseone-Omega-{Omega}-tau-{tau}"
    elif pulse_type == 2:
        pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.PULSE_TYPE2
        sim_signature = f"data_tullyone_pulsetwo-Omega-{Omega}-tau-{tau}"
    elif pulse_type == 3:
        pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.PULSE_TYPE3
        sim_signature = f"data_tullyone_pulsethree-Omega-{Omega}-tau-{tau}"
    else:
        raise ValueError(f"The pulse_type must be 1, 2, or 3. But it is {pulse_type}.")
    # nsamples = 48
    # p_bounds = (0.5, 35)
    nsamples = 8
    p_bounds = (30, 35)
    main(sim_signature, nsamples, Omega, tau, pulse_type, p_bounds)

# %%
