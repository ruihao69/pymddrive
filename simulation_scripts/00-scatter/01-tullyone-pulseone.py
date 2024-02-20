# %% 
import numpy as np
from pymddrive.models.tully import TullyOne
from pymddrive.models.tully_w_pulse import TullyOnePulseOne
from pymddrive.integrators.state import State
from pymddrive.dynamics.dynamics import NonadiabaticDynamics, _output_ehrenfest
from pymddrive.pulses.morlet import MorletReal


from dataclasses import dataclass
from multiprocessing import Pool

from typing import Tuple

from tullyone_utils import *

def map_interaction_region(A, B, C, D, r_boundary: Tuple=(-10, 10), n_samples: int=100, rtol:float=3):
    model = TullyOne(A, B, C, D)
    x_samples = np.linspace(r_boundary[0], r_boundary[1], n_samples)
    x_mask = np.zeros(n_samples, dtype=bool)
    dE = model.A 
    for i, x in enumerate(x_samples):
        _, _, _, d, _ = model(0.0, x)
        if np.abs(d[0, 1]) > dE * rtol:
            x_mask[i] = True
    x_int = x_samples[x_mask]
    return (x_int.min(), x_int.max())

def estimate_delay_tullyone(r0: float, p0: float, A, B, C, D, mass: float=2000.0, r_interaction: float=0.0) -> Tuple[float, float]:
    t_init = 0.0
    int_boundaries = map_interaction_region(A, B, C, D)
    t_before_interaction_region = (int_boundaries[0] - r0) / (p0 / mass)
    dt_reach_interaction_region = (r_interaction - int_boundaries[0]) / (p0 / mass)
    
    t0 = t_init + t_before_interaction_region + dt_reach_interaction_region
    fwhm = dt_reach_interaction_region * 2.0 / np.sqrt(2.0)
    sigma = FWHM_to_sigma(fwhm)
    return t0, sigma

def get_pulseone_for_tullyone(
    A, B, C, D,
    r0: float, p0: float,
    Omega: float, sigma: float,
    mass: float=2000.0,
    r_interaction: float=0.0,
    # Omega: float=0.03
):
    t0, _ = estimate_delay_tullyone(r0, p0, A, B, C, D, mass, r_interaction)
    return MorletReal(A=C, t0=t0, tau=sigma, Omega=Omega, phi=0)

def run_tully_one_pulse_one(r0: float, p0: float, Omega: float, sigma: float, mass: float=2000, solver='Ehrenfest'):
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0
    r_boundary = (-10, 10)
    
    pulse = get_pulseone_for_tullyone(A, B, C, D, r0, p0, Omega=Omega, sigma=sigma)
    t_range = (pulse.t0 - pulse.tau, pulse.t0 + pulse.tau)
    model = TullyOnePulseOne(A, B, pulse=pulse)
    
    t0 = 0.0
    s0 = State(r0, p0, np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128))
    dyn = NonadiabaticDynamics(model, t0, s0, mass, solver=solver, r_bounds=r_boundary, t_bounds=t_range)
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
                
    sr = ehrenfest_scatter_result(s)
    print("Finished!")
    return output, sr, pulse

def get_middle_sigma(p, sigma):
    from scipy.integrate import simps
    argsort = np.argsort(sigma)
    sigma = np.array(sigma)[argsort]
    p = np.array(p)[argsort]
    
    I = simps(p, sigma)
    for i in range(2, len(sigma)):
        if simps(p[:i], sigma[:i]) / I > 0.5:
            sig_mid = sigma[i]
            break
    return sig_mid

def sample_sigma(r0: float=-10.0):
    p0_list = linspace_log(10, 100, 1000)
    sigma_list = []
    for p0 in p0_list:
        _, sigma = estimate_delay_tullyone(r0, p0, A, B, C, D)
        sigma_list.append(sigma)    
        
    return get_middle_sigma(p0_list, sigma_list)

    

# %%
if __name__ == "__main__":
    import os
    import itertools 
    
    ncpus = get_ncpus()
    
    A = 0.01
    B = 1.6
    C = 0.005
    D = 1.0
    r_boundary = (-10, 10)
    
    # nsamples = 40
    nsamples = 4
    
    sigma_list = [60.0]
    Omega_list = [A, A*3, A*10, A*30]
    
    for sigma, Omega in itertools.product(sigma_list, Omega_list):
        sim_signature = f"data_tullyone-pulseone-Omega_{Omega}-sigma_{sigma}"
        if not os.path.exists(sim_signature):
            os.mkdir(sim_signature)
            
        _r0_list = [-10.0] * nsamples
        # _p0_list = linspace_log10(0.5, 35, nsamples)
        _p0_list = linspace_log10(10, 35, nsamples)
        _sigma_list = [sigma] * nsamples
        _Omega_list = [Omega] * nsamples  
        
        with Pool(ncpus) as pool:
            results = pool.starmap(run_tully_one_pulse_one, zip(_r0_list, _p0_list, _Omega_list, _sigma_list))
        
        output_list = [result[0] for result in results]
        sr_list = [result[1] for result in results]
        pulse_list = [result[2] for result in results]
        
        fn_scatter = os.path.join(sim_signature, f"scatter.dat")
        fn_trajectory = os.path.join(sim_signature, f"trajectory.npz")
        fn_fig = os.path.join(sim_signature, f"scatter.pdf")
        fn_pulse = os.path.join(sim_signature, f"pulses.npz")
        
        data_scatter = save_data(_p0_list, output_list, sr_list, fn_scatter)
        output_dict = {
            'p0': _p0_list,
            'output': output_list,
        }
        save_trajectory(output_dict, fn_trajectory)
        fig = plot_scatter_result(_p0_list, sr_list, fname=fn_fig)
        save_pulses(_p0_list, pulse_list, fn_pulse)
          
    
# %%
if __name__ == "__main__":
    pass
    # fig = plt.figure(dpi=200, figsize=(3*2, 2*3))
    # gs = fig.add_gridspec(3, 2)
    # axs = gs.subplots()
    # ax1, ax2, ax3 = axs[:, 0]
    # ax1.plot(output['t'], output['states']['R'])
    # ax1.set_ylabel('R')
    # ax2.plot(output['t'], output['states']['P'])
    # ax2.set_ylabel('P')
    # ax3.plot(output['t'], output['states']['rho'][:, 0, 0].real, label='rho11')
    # ax3.plot(output['t'], output['states']['rho'][:, 1, 1].real, label='rho22')
    # ax3.legend()
    # ax3.set_ylabel('rho')
    # ax3.set_xlabel('t')
    
    # ax4, ax5, ax6 = axs[:, 1]
    # ax4.plot(output['t'], output['KE'], label='KE')
    # ax5.plot(output['t'], output['PE'], label='PE')
    # ax6.plot(output['t'], output['TE'], label='TE')
    # for ax in axs[:, 1]:
    #     ax.set_xlabel('t')
    #     ax.legend()
    # ax4.set_ylabel('KE')    
    # ax5.set_ylabel('PE')
    # ax6.set_ylabel('TE')
    # fig.tight_layout()
    
    # fig = plt.figure(dpi=200)
    # ax = fig.add_subplot(111)
    # sig = [p(t) for t in output['t']]
    # ax.plot(output['t'], sig)
    # ax.set_ylabel('pulse')
    # ax.set_xlabel('t')
    # fig.tight_layout()
