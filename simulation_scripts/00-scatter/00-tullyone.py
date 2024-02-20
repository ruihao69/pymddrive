# %%
import numpy as np
from pymddrive.models.tully import TullyOne
from pymddrive.integrators.state import State
from pymddrive.dynamics.dynamics import NonadiabaticDynamics, _output_ehrenfest

from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

from typing import List
import matplotlib.pyplot as plt

from tullyone_utils import *

def run_tullyone(r0: float, p0: float, mass: float=2000, solver='Ehrenfest'):
    model = TullyOne()
    t0 = 0.0
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State(r0, p0, rho0)
    # dyn = NonadiabaticDynamics(model, t0, s0, mass, solver=solver, x_bounds=(-10, 10))
    dyn = NonadiabaticDynamics(model, t0, s0, mass, solver=solver)
    output = {
        't': np.array([t0]),
        'states': np.copy(s0.data),
        'KE': np.array([0.5 * p0**2 / mass]),
        'PE': np.array([-model.A]),
        'TE': np.array([0.5 * p0**2 / mass - model.A])
    }

    t = t0
    s = s0

    while inside_boundary(s):
        t, s = dyn.step(t, s)
        dyn.nsteps += 1
        if dyn.nsteps % dyn.save_every == 0:
            _output = _output_ehrenfest(t, s0, dyn.model, dyn.mass)
            _output.update({'t': t, 'states': s0.data})
            for key, value in _output.items():
                output[key] = np.append(output[key], value)
            # print(s.data['R'])

    sr = ehrenfest_scatter_result(s)
    print("Finished!")
    return output, sr


# %%
if __name__ == "__main__":
    import os 
    
    # size of the simulation
    n_samples = 40
    ncpus = get_ncpus()
    
    sim_signature = "data_tullyone"
    if not os.path.exists(sim_signature):
        os.makedirs(sim_signature)

    r0 = -10.0
    _r0_list = np.array([r0]*n_samples)
    _p0_list = linspace_log10(0.5, 35.0, n_samples)
    with Pool(ncpus) as p:
        results = p.starmap(run_tullyone, [(r0, p0) for r0, p0 in zip(_r0_list, _p0_list)])
    output_list = [result[0] for result in results]
    sr_list = [result[1] for result in results]
    
    fn_scatter = os.path.join(sim_signature, f"scatter.dat")
    fn_trajectory = os.path.join(sim_signature, f"trajectory.npz")
    fn_fig = os.path.join(sim_signature, f"scatter.pdf")
    
    res = save_data(_p0_list, output_list, sr_list, fn_scatter)
    plot_scatter_result(_p0_list, sr_list, fname=fn_fig)
    output_dict = {
        'p0': _p0_list,
        'output': output_list,
    }
    save_trajectory(output_dict, fn_trajectory)

# %%
