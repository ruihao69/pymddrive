# %%
import numpy as np

from pymddrive.models.tullyone import get_tullyone
from pymddrive.integrators.state import get_state
from pymddrive.dynamics.options import BasisRepresentation, NonadiabaticDynamicsMethods, NumericalIntegrators  
from pymddrive.dynamics.get_dynamics import get_dynamics
from pymddrive.dynamics.run import run_ensemble

from scatter_postprocess import load_single_pymddrive_nc, load_trajectories

import os

def stop_condition(t, s):
    r, p, _ = s.get_variables()
    if (p < 0) or (r > 0):
        return True

def get_tullyone_ehrenfest_diab_dynamics(
    r0: float, 
    p0: float, 
    mass: float=2000, 
    dt: float = 0.03,
) -> None:
    # get the initial time and state object
    t0 = 0.0
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = 1
    s0 = get_state(mass=mass, R=r0, P=p0, rho_or_psi=rho0)
    
    # intialize the model
    hamiltonian = get_tullyone()
    
    # get the dynamics object
    basis_rep = BasisRepresentation.DIABATIC
    solver = NonadiabaticDynamicsMethods.EHRENFEST
    dyn = get_dynamics(t0=t0, s0=s0, dt=dt, hamiltonian=hamiltonian, dynamics_basis=basis_rep, method=solver)
    return dyn
    
def run():
    R0 = -10.0
    P0_list = np.linspace(0.5, 35, 100)
    dyn_list = []
    for P0 in P0_list:
        dyn = get_tullyone_ehrenfest_diab_dynamics(R0, P0)
        dyn_list.append(dyn)
    filename = os.path.join('data_delay_time', 'ehrenfest.nc')
    if not os.path.isdir('data_delay_time'):
        os.makedirs('data_delay_time')
    run_ensemble(
        dynamics_list=dyn_list,
        break_condition=stop_condition,
        filename=filename,
        save_every=10
    )
    
def tabulate_delay_time():
    project_dir = './data_delay_time'
    traj_ncfiles, _ = load_trajectories(project_dir)
    P0 = np.array([])
    delay = np.array([])
    for ii, traj_data in enumerate(traj_ncfiles):
        t, _, P, _, _, _, _ = load_single_pymddrive_nc(traj_data)
        delay = np.append(delay, t[-1])
        P0 = np.append(P0, P[0])
    sort_idx = np.argsort(P0)
    P0 = P0[sort_idx]
    delay = delay[sort_idx]
    file_out = os.path.join(project_dir, 'delay_time.txt')
    header = f'{"P0":>10} {"delay":>12}'
    np.savetxt(file_out, np.vstack((P0, delay)).T, header=header, fmt='%12.6f')
    
def get_delay_time_interpolant():
    project_dir = './data_delay_time'
    file_out = os.path.join(project_dir, 'delay_time.txt')
    P0, delay = np.loadtxt(file_out, unpack=True)
    from scipy.interpolate import interp1d
    return interp1d(P0, delay, kind='cubic')
        

# %%
if __name__ == '__main__':
    run()
    
# %%
tabulate_delay_time()
p_samples = np.linspace(0.5, 35, 1000)
interpolant = get_delay_time_interpolant()
delay_time_prediction = interpolant(p_samples)

import matplotlib.pyplot as plt
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111)
P0, delay = np.loadtxt('./data_delay_time/delay_time.txt', unpack=True)
ax.plot(p_samples, delay_time_prediction)
ax.plot(P0, delay, 'o')
ax.set_xlabel('Initial momentum')
ax.set_ylabel('Delay time')
ax.set_yscale('log')
plt.show()

# %%
