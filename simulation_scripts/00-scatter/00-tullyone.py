# %% 
import numpy as np
from pymddrive.models.tully import TullyOne
from pymddrive.integrators.state import State
from pymddrive.dynamics.dynamics import NonadiabaticDynamics, _output_ehrenfest

from dataclasses import dataclass
from multiprocessing import Pool

from typing import List
import matplotlib.pyplot as plt

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
    

def run_tullyone(r0: float, p0: float, mass: float=2000, solver='Ehrenfest'):
    model = TullyOne()
    t0 = 0.0
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State(r0, p0, rho0)
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


def plot_scatter_result(p0_list: List, sr_list: List[ScatterResult]):
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
    
    fig = plt.figure(dpi=200, figsize=(3*2, 2*2))
    fig.supylabel("Probability")
    # fig.supxlabel(r"$p_0$ (a.u.)")
    # axs_label = ["Transmission L", "Reflection L", "Transmission U", "Reflection U"]
    axs_label = ['Transmission L', 'Reflection L', 'Reflection U', 'Transmission U']
    alph_label = ["(a)", "(b)", "(c)", "(d)"]
    y_list = [tl, rl, ru, tu]
    gs = fig.add_gridspec(2, 2, hspace=0.0)
    axs = gs.subplots(sharex=True, sharey=True, squeeze=False)
    for i, ax in enumerate(axs.flatten()):
        ax.plot(p0_list, y_list[i], label=axs_label[i], **tuple(p for p in prop_cycler)[i])
        ax.text(0.90, 0.90, alph_label[i], ha='center', va='center', transform=ax.transAxes)
        ax.set_ylim(-0.05, 1.05)
        # ax.legend(loc='best')
        ax.set_xlabel(r"$p_0$ (a.u.)")
        
    fig.tight_layout()
    fig.savefig("00-scatter-tullyone.pdf", )
    

# %%
if __name__ == "__main__": 
    r0 = -10.0
    N = 8
    r0_list = np.array([r0]*N)
    p0_list = linspace_log10(0.5, 35.0, N)
    with Pool(32) as p:
        results = p.starmap(run_tullyone, [(r0, p0) for r0, p0 in zip(r0_list, p0_list)])
    sr_list = [result[1] for result in results] 
    plot_scatter_result(p0_list, sr_list)
    