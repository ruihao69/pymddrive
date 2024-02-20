import os
import numpy as np
import matplotlib.pyplot as plt
from pymddrive.integrators.state import State
from dataclasses import dataclass
from typing import List
from multiprocessing import cpu_count

__all__ = [
    'ScatterResult',
    'linspace_log',
    'linspace_log10',
    'inside_boundary',
    'ehrenfest_scatter_result',
    'FWHM_to_sigma',
    'sigma_to_FWHM',
    'save_data',
    'save_trajectory',
    'save_pulses',
    'get_ncpus',
    'plot_scatter_result',
    '_load_data_for_plotting',
]

@dataclass
class ScatterResult:
    is_transmission: bool
    prob_up: float
    prob_down: float
    
def linspace_log(start, stop, num=50):
    return np.exp(np.linspace(np.log(start), np.log(stop), num))

def linspace_log10(start, stop, num=50):
    return np.power(10, np.linspace(np.log10(start), np.log10(stop), num))

def inside_boundary(s: State):
    r, p, _ = s.get_variables()
    if (r > 10.0) and (p > 0.0):
        return False
    elif (r < -10.0) and (p < 0.0):
        return False
    else:
        return True
    
def ehrenfest_scatter_result(s: State):
    r, p, rho = s.get_variables()
    is_transmission = (r > 0) and (p > 0)
    prob_down = rho[0, 0].real
    prob_up = rho[1, 1].real
    return ScatterResult(is_transmission, prob_up, prob_down)

def FWHM_to_sigma(FWHM: float) -> float:
    return FWHM / (2.0 * np.sqrt(2.0 * np.log(2)))

def sigma_to_FWHM(sigma: float) -> float:
    return sigma * (2.0 * np.sqrt(2.0 * np.log(2)))

def save_data(p0_list: List, output_list: List, sr_list: List, filename: str):
    assert len(p0_list) == len(output_list) == len(sr_list)
    res = np.zeros((len(output_list), 5))
    for ii, (p0, output, sr) in enumerate(zip(p0_list, output_list, sr_list)):
        res[ii, 0] = p0
        res[ii, 1] = output['states']['R'][-1]
        res[ii, 2] = output['states']['P'][-1]
        res[ii, 3] = sr.prob_down
        res[ii, 4] = sr.prob_up

    fmt = ["%10.5f", "%10.5f", "%10.5f", "%10.5f", "%10.5f"]
    header = f"{'p0':>8s}{'r':>11s}{'p':>11s}{'prob_down':>11s}{'prob_up':>11s}"
    np.savetxt(filename, res, fmt=fmt, header=header)
    return res

def save_trajectory(output: dict, filename: str):
    np.savez(filename, **output)

def get_ncpus(): 
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        print("We are on a SLURM system!, using SLURM_CPUS_PER_TASK to determine ncpus.")
        ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        print("We are not on a SLURM system, using cpu_count to determine ncpus.")
        ncpus = cpu_count()
        
    return ncpus
        
def plot_scatter_result(p0_list: List, sr_list: List[ScatterResult], fig=None, fname: str = None):
    from cycler import cycler
    sr = np.zeros((4, len(sr_list)))
    tu, tl, ru, rl = sr
    for ii, sr in enumerate(sr_list):
        if sr.is_transmission:
            tu[ii] = sr.prob_up
            tl[ii] = sr.prob_down
        else:
            ru[ii] = sr.prob_up
            rl[ii] = sr.prob_down

    c_cycler = cycler(color=['r', 'g', 'b', 'y'])
    ls_cycler = cycler(linestyle=['-', '--', '-.', ':'])
    mk_cycler = cycler(marker=['o', 's', 'x', 'd'])
    prop_cycler = c_cycler + ls_cycler + mk_cycler
    if fig is None:
        fig = plt.figure(dpi=200, figsize=(3*2, 2*2))
        fig.supxlabel(r"$p_0$ (a.u.)")
        fig.supylabel("Probability")
        # fig.supxlabel(r"$p_0$ (a.u.)")
        # axs_label = ["Transmission L", "Reflection L", "Transmission U", "Reflection U"]
        axs_label = ['Transmission L', 'Reflection L', 'Reflection U', 'Transmission U']
        alph_label = ["(a)", "(b)", "(c)", "(d)"]
        y_list = [tl, rl, ru, tu]
        gs = fig.add_gridspec(2, 2, hspace=0.0, wspace=0.0)
        axs = gs.subplots(sharex=True, sharey=True, squeeze=False)
        for i, ax in enumerate(axs.flatten()):
            ax.plot(p0_list, y_list[i], label=axs_label[i], **tuple(p for p in prop_cycler)[i], markersize=2, lw=0.75)
            ax.text(0.90, 0.90, alph_label[i], ha='center', va='center', transform=ax.transAxes)
            ax.set_ylim(-0.05, 1.05)
            # ax.legend(loc='best')

    else:
        c_cycler = cycler(color=['g', 'b', 'y', 'r'])
        ls_cycler = cycler(linestyle=[':', '-', '--', '-.', ])
        mk_cycler = cycler(marker=['d', 'o', 's', 'x', ])
        prop_cycler = c_cycler + ls_cycler + mk_cycler
        for i, ax in enumerate(fig.get_axes()):
            ax.plot(p0_list, y_list[i], label=axs_label[i], **tuple(p for p in prop_cycler)[i], markersize=2, lw=0.75)

    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname)
    
def _load_data_for_plotting(filename: str):
    res = np.loadtxt(filename)
    pi_list = res[:, 0]
    rf_list = res[:, 1]
    prob_down_list = res[:, 3]
    prob_up_list = res[:, 4]
    sr_list = [ScatterResult(rf > 0, prob_up, prob_down) for rf, prob_up, prob_down in zip(rf_list, prob_up_list, prob_down_list)]
    return pi_list, sr_list

def save_pulses(p0_list, pulse_list, filename):
    pulse_dict = {
        'p0': p0_list,
        'pulse': pulse_list,
    }
    np.savez(filename, **pulse_dict)