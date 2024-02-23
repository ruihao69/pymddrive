# %%
import os 
import numpy as np
from pymddrive.models.tully import TullyOne
from pymddrive.integrators.state import State
from pymddrive.dynamics.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics

from tullyone_utils import *

def stop_condition(t, s, states):
    r, _, _ = s.get_variables()
    return outside_boundary(r, (-10, 10))

def break_condition(t, s, states):
    return False

def run_tullyone(r0: float, p0: float, mass: float=2000, solver='Ehrenfest', method='vv_rk4'):
    # intialize the model
    model = TullyOne()
    
    # initialize the states
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State(r0, p0, rho0)
    
    dyn = NonadiabaticDynamics(
        model=model,
        t0=0.0,
        s0=s0,
        mass=mass,
        solver=solver,
        method=method,
        r_bounds=(-10, 10)
    )
    
    return run_nonadiabatic_dynamics(dyn, stop_condition, break_condition) 

def main(sim_signature: str, n_samples: int, p_bounds: tuple=(0.5, 35.0)):
    import os
    from pararun import ParaRunScatter, get_ncpus
    
    ncpus = get_ncpus()
    if not os.path.exists(sim_signature):
        os.makedirs(sim_signature)
        
    r0 = -10.0
    _r0_list = np.array([r0]*n_samples)
    _p0_list = linspace_log10(*p_bounds, n_samples)
    
    runner = ParaRunScatter(n_jobs=ncpus, r0=_r0_list, p0=_p0_list)
    
    res_gen = runner.run(run_tullyone, accumulate_output, sim_signature)
    traj_dict, pulses = accumulate_output(_p0_list, res_gen)
    
    post_process_output(sim_signature, traj_dict, pulse_list=pulses)
    

# %%
if __name__ == "__main__": 
    sim_signature = "data_tullyone"
    nsamples = 48
    p_bounds = (0.5, 35)
    
    main(sim_signature, nsamples, p_bounds)
    
    p0_list, sr_list = load_data_for_plotting(os.path.join(sim_signature, 'scatter.dat'))

    print(p0_list)
    print(sr_list)
   
# %%
