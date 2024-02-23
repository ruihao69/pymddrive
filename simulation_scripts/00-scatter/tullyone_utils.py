import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Union, Generator

__all__ = [
    'ScatterResult',
    'linspace_log',
    'linspace_log10',
    'inside_boundary',
    'outside_boundary',
    'is_trapped',
    'get_scatter_result',
    'FWHM_to_sigma',
    'sigma_to_FWHM',
    # 'save_data',
    # 'save_trajectory',
    # 'save_pulses',
    # 'get_ncpus',
    'plot_scatter_result',
    'load_data_for_plotting',
    'accumulate_output',
    'post_process_output'
]

@dataclass(frozen=True)
class ScatterResult:
    is_transmission: Union[bool, None]
    is_trapped: bool
    prob_up: float
    prob_down: float
    
def linspace_log(start, stop, num=50):
    return np.exp(np.linspace(np.log(start), np.log(stop), num))

def linspace_log10(start, stop, num=50):
    return np.power(10, np.linspace(np.log10(start), np.log10(stop), num))

def outside_boundary(r: float, r_bounds) -> bool:
    return (r < r_bounds[0]) or (r > r_bounds[1])

def inside_boundary(r: float, r_bounds) -> bool:
    return (r >= r_bounds[0]) and (r <= r_bounds[1])
    
def _count_re_crossings(r, r_TST):
     r_sign = np.sign(r - r_TST)
     r_sign_diff = np.diff(r_sign)
     n = np.sum(r_sign_diff != 0) - 1
     n_re_crossings = 0 if n < 0 else n
     return n_re_crossings

def is_trapped(r_traj: np.array, r_TST: float=0.0, recross_tol:int=10) -> bool:
    return _count_re_crossings(r_traj, r_TST) > recross_tol
    

def get_scatter_result(traj: dict, r_TST: float = 0.0):
    r_f = traj['R'][-1]
    p_f = traj['P'][-1]
    rho_f = traj['rho'][-1]
    pop_f = rho_f.diagonal().real
    
    _is_trapped = is_trapped(traj['R'], r_TST)
    
    if _is_trapped:
        return ScatterResult(None, True, pop_f[1], pop_f[0])
    else:
        is_transmission = (r_f > r_TST) and (p_f > 0)
        return ScatterResult(is_transmission, False, pop_f[1], pop_f[0])

def FWHM_to_sigma(FWHM: float) -> float:
    return FWHM / (2.0 * np.sqrt(2.0 * np.log(2)))

def sigma_to_FWHM(sigma: float) -> float:
    return sigma * (2.0 * np.sqrt(2.0 * np.log(2)))

def save_data(p0_list: List, output_dict: dict, sr_list: List, filename: str):
    assert len(p0_list) == len(sr_list)
    res = np.zeros((len(p0_list), 6))
    for ii, (p0, sr) in enumerate(zip(p0_list, sr_list)):
        res[ii, 0] = p0
        res[ii, 1] = output_dict['R'][ii][-1]
        res[ii, 2] = output_dict['P'][ii][-1]
        res[ii, 3] = sr.prob_down
        res[ii, 4] = sr.prob_up
        res[ii, 5] = 1 if sr.is_trapped else 0

    fmt = ["%10.5f", "%10.5f", "%10.5f", "%10.5f", "%10.5f", "%10d"]
    header = f"{'p0':>8s}{'r':>11s}{'p':>11s}{'prob_down':>11s}{'prob_up':>11s}{'trapped':>11s}"
    np.savetxt(filename, res, fmt=fmt, header=header)
    return res

def save_trajectory(output: dict, filename: str):
    np.savez(filename, **output)

def plot_scatter_result(p0_list: List, sr_list: List[ScatterResult], fig=None, fname: str = None):
    from cycler import cycler
    sr = np.zeros((4, len(sr_list)))
    tu, tl, ru, rl = sr
    for ii, sr in enumerate(sr_list):
        if sr.is_trapped:
            tu[ii] = tl[ii] = ru[ii] = rl[ii] = np.nan
            continue
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

def load_data_for_plotting(filename: str):
    res = np.loadtxt(filename)
    p0_list = res[:, 0]
    rf_list = res[:, 1]
    prob_down_list = res[:, 3]
    prob_up_list = res[:, 4]
    is_trapped_list = res[:, 5].astype('bool')
    sr_list = []
    for rf, pop0, pop1, _is_trapped in zip(rf_list, prob_down_list, prob_up_list, is_trapped_list):
        if _is_trapped:
            sr_list.append(ScatterResult(None, True, pop0, pop1))
        else:
            sr_list.append(ScatterResult(rf>0, False, pop0, pop1))
            
    return p0_list, sr_list
        
def save_pulses(p0_list, pulse_list, filename):
    pulse_dict = {
        'p0_list': p0_list,
        'pulses': pulse_list,
    }
    np.savez(filename, **pulse_dict)
    
def accumulate_output(p0_list, out_gen: Generator):
    # assert len(p0_list) == len(out_gen)
    traj_dict = {
        'p0': np.array(p0_list),
        'traj': {
            'time': [],
            'R': [],
            'P': [],
            'rho': [],
            'KE': [],
            'PE': [],
        }
    }
    traj = traj_dict['traj']
    has_pulse = None
    pulse = None
    pulses = None
    for _output in out_gen:
        if has_pulse is None:
            has_pulse = True if (len(_output) == 2) and isinstance(_output, tuple) else False
            if has_pulse:
                pulses = []
        if has_pulse: 
            output, pulse = _output
        else:
            output = _output
            
        traj['time'].append(output['time'])
        traj['R'].append(output['states']['R'])
        traj['P'].append(output['states']['P'])
        traj['rho'].append(output['states']['rho'])
        traj['KE'].append(output['KE'])
        traj['PE'].append(output['PE'])
        if pulse is not None:
            pulses.append(pulse)
        
    return traj_dict, pulses

def loop_over_traj_dict(traj_dict: dict):
    get_single_traj = lambda traj_dict, i: {key: traj_dict[key][i] for key in traj_dict}
    n = len(traj_dict['time'])
    return (get_single_traj(traj_dict, i) for i in range(n))
    

def post_process_output(sim_signature: str, traj_dict: dict, pulse_list: list | None=None):
    fn_fig = os.path.join(sim_signature, f"scatter.pdf")
    fn_trajectories = os.path.join(sim_signature, f"trajectories.npz")
    fn_scatter = os.path.join(sim_signature, f"scatter.dat")
    fn_pulses = os.path.join(sim_signature, f"pulses.npz")
    
    # save txt file from plotting scattering results    
    scatter_list = [get_scatter_result(traj) for traj in loop_over_traj_dict(traj_dict['traj'])]
    save_data(traj_dict['p0'], traj_dict['traj'], scatter_list, fn_scatter)
    
    # save the trajectory into npz file
    save_trajectory(traj_dict, fn_trajectories)
    
    # plot the scatter result
    plot_scatter_result(traj_dict['p0'], scatter_list, fname=fn_fig)
    
    # save the pulse
    if pulse_list is not None:
        save_pulses(p0_list=traj_dict['p0'], pulse_list=pulse_list, filename=fn_pulses)