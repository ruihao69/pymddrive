# %% This file contains the main function to run the nonadiabatic dynamics simulation.
import numpy as np

from pymddrive.integrators.state import State
from pymddrive.dynamics.nonadiabatic_dynamics import NonadiabaticDynamics

import warnings
from typing import NamedTuple

def run_nonadiabatic_dynamics(
    dyn: NonadiabaticDynamics,
    stop_condition: callable,
    break_condition: callable,
    max_iters: int=int(1e8),
    save_traj: bool=True,
):
    check_stop_every = dyn.save_every * 30
    check_break_every = dyn.save_every * 30
    time_array = np.array([])
    traj_array = None
    properties_dict = {field: [] for field in dyn.properties_type._fields}
    
    t, s = dyn.t0, dyn.s0
    cache = None
    
    for istep in range(max_iters):
        if istep % dyn.save_every == 0:
            properties = dyn.calculate_properties(t, s)
            properties_dict = _append_properties(properties_dict, properties)
            time_array = np.append(time_array, t)
            traj_array = np.array([s.data]) if traj_array is None else np.append(traj_array, s.data)
            # print(f"{s.data['R']=}")
            if istep % check_stop_every == 0:
                if stop_condition(t, s, traj_array):
                    break
            if istep % check_break_every == 0:
                if break_condition(t, s, traj_array):
                    warnings.warn("The break condition is met.")
                    break
        t, s, cache = dyn.step(t, s, cache)
    properties_dict = {field: np.array(value) for field, value in properties_dict.items()}
    if save_traj:
        output = {'time': time_array, 'states': traj_array, **properties_dict}
    else:
        output = {'time': time_array, **properties_dict}
    return output

# helper functions
def _append_properties(properties_dict: dict, properties: NamedTuple) -> dict:
    for (field, value) in zip(properties._fields, properties):
        properties_dict[field].append(value)
    return properties_dict
    
# %% The temporary test code
def _debug_test():
    import time
    from pymddrive.models.tullyone import TullyOnePulseTypes, get_tullyone
    from pymddrive.dynamics.options import (
        BasisRepresentation, QunatumRepresentation, 
        NonadiabaticDynamicsMethods, NumericalIntegrators
    )
    import matplotlib.pyplot as plt
    
    # get a convenient model for testing
    hamiltonian = get_tullyone(
        pulse_type=TullyOnePulseTypes.NO_PULSE, 
    )
    mass = 2000.0
    
    # initial conditions
    t0 = 0.0; r0 = -10.0; p0 = 30.0
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    
    # prepare the dynamics object
    dyn = NonadiabaticDynamics( 
        hamiltonian=hamiltonian,
        t0=t0, s0=s0, mass=mass,
        qm_rep=QunatumRepresentation.DensityMatrix,
        basis_rep=BasisRepresentation.Diabatic,
        solver=NonadiabaticDynamicsMethods.EHRENFEST,
        numerical_integrator=NumericalIntegrators.RK4,
        r_bounds=(-10.0, 10.0),
    )
    
    def stop_condition(t, s, states):
        r, _, _ = s.get_variables()
        return (r>10.0) or (r<-10.0)
    
    def break_condition(t, s, states):
        r = np.array(states['R'])
        def count_re_crossings(r, r_TST=0.0):
            r_sign = np.sign(r - r_TST)
            r_sign_diff = np.diff(r_sign)
            n = np.sum(r_sign_diff != 0) - 1
            n_re_crossings = 0 if n < 0 else n
            return n_re_crossings
        return (count_re_crossings(r) > 10)
    
    start = time.perf_counter() 
    output = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    print(f"The time for the simulation is {time.perf_counter()-start} s.")
    
    
    # initial conditions
    t0 = 0.0; r0 = -10.0; p0 = 30.0
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    
    # prepare the dynamics object
    dyn = NonadiabaticDynamics( 
        hamiltonian=hamiltonian,
        t0=t0, s0=s0, mass=mass,
        qm_rep=QunatumRepresentation.DensityMatrix,
        basis_rep=BasisRepresentation.Diabatic,
        solver=NonadiabaticDynamicsMethods.EHRENFEST,
        numerical_integrator=NumericalIntegrators.ZVODE,
        r_bounds=(-10.0, 10.0)
    ) 
    start = time.perf_counter() 
    output_zvode = run_nonadiabatic_dynamics(dyn, stop_condition, break_condition)
    print(f"The time for the simulation using scipy zode is {time.perf_counter()-start} s.")
    
    t0, t1 = output['time'], output_zvode['time']
    r0, r1 = output['states']['R'], output_zvode['states']['R']
    p0, p1 = output['states']['P'], output_zvode['states']['P']
    # rho0, rho1= output['states']['rho'], output_zvode['states']['rho']
    pop0, pop1 = output['populations'], output_zvode['populations']
    
    plt.plot(t0, r0, ls='-', label='home-made RK4')
    plt.plot(t1, r1, ls='-.', label='scipy zvode')
    plt.xlabel('Time')
    plt.ylabel('R')
    plt.legend()
    plt.title('Position') 
    plt.show()
    plt.plot(t0, p0, ls='-', label='home-made RK4')
    plt.plot(t1, p1, ls='-.', label='scipy zvode')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('P')
    plt.title('Momentum')
    plt.show()
    plt.plot(t0, pop0[:, 0].real, ls='-', label='pop0: home-made RK4') 
    plt.plot(t0, pop0[:, 1].real, ls='-', label='pop1: home-made RK4') 
    plt.plot(t1, pop1[:, 0].real, ls='-.', label='pop0: scipy zvode')
    plt.plot(t1, pop1[:, 1].real, ls='-.', label='pop1: scipy zvode')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()
    
     
    
# %% the __main__ code
if __name__ == "__main__":
    _debug_test()
    
# %%


