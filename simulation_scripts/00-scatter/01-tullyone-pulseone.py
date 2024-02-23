# %% 
import os
import time
import numpy as np
from pymddrive.models.tully_w_pulse import TullyOne
from pymddrive.models.tully_w_pulse import TullyOnePulseOne
from pymddrive.pulses.morlet import MorletReal
from pymddrive.integrators.state import State
from pymddrive.dynamics.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics 


from tullyone_utils import *

def stop_condition(t, s, states):
    r, _, _ = s.get_variables()
    return outside_boundary(r, (-10, 10))

def break_condition(t, s, states):
    R = np.array(states['R'])
    return is_trapped(R, r_TST=0.0, recross_tol=10)

def run_tullyone_pulseone(r0, p0, Omega, tau, mass=2000, solver='Ehrenfest', method='vv_rk4'):
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0
    
    # To estimate the delay time
    start = time.time()
    _delay = estimate_delay_time(A, B, C, D, p0)
    print(f"Time elapsed for estimating the delay time is {time.time()-start:.5f} seconds.", flush=True)
    
    # initialize the model and states
    pulse = MorletReal(A=C, t0=_delay, tau=tau, Omega=Omega, phi=0)
    model = TullyOnePulseOne(A, B, pulse=pulse)
    
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
    
    res_gen = runner.run(run_tullyone_pulseone, accumulate_output, sim_signature)
    traj_dict, pulses = accumulate_output(_p0_list, res_gen)
    
    post_process_output(sim_signature, traj_dict, pulse_list=pulses)
    
# %% 
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # out, pulse = run_tullyone_pulseone(-10.0, 30.0, Omega=0.01*5, tau=60.0)
    
    # plt.plot(out['time'], out['states']['R'])
    # plt.show()
    
    # plt.plot(out['time'], out['states']['P'])
    # plt.show()
    
    # plt.plot(out['time'], out['states']['rho'][:, 0, 0].real)
    # plt.plot(out['time'], out['states']['rho'][:, 1, 1].real)
    # plt.show()
    
    # plt.plot(out['time'], [pulse(t) for t in out['time']])
    
    Omega = 0.01 * 3
    tau = 60.0
    
    sim_signature = "data_tullyone_pulseone-Omega-0.01-tau-60"
    nsamples = 8
    # p_bounds = (0.5, 35)
    p_bounds = (20, 35)
    main(sim_signature, nsamples, Omega, tau, p_bounds)

# %%
