# %%
from scipy.integrate import ode
from joblib import delayed, Parallel

from pymddrive.my_types import GenericVector
from pymddrive.dynamics.dynamics import Dynamics
from pymddrive.dynamics.misc_utils import assert_valid_real_positive_value, numerate_file_name
from pymddrive.dynamics.options import NumericalIntegrators, NonadiabaticDynamicsMethods
from pymddrive.low_level.states import State
from pymddrive.integrators.state_rk4 import state_rk4
from pymddrive.dynamics.output_writer import PropertiesWriter
from pymddrive.dynamics.restart_writer import RestartWriter
from pymddrive.utils import get_ncpus, is_empty_ncdf

import os
import re
import glob
from typing import Optional, Dict, Any, Tuple, Callable, List
import warnings

MAX_STEPS = int(1e8)

def get_ode_solver_options(
    dynamics: Dynamics,
    max_step: Optional[float]=None, 
    min_step: Optional[float]=None
) -> Dict[str, Any]:
    ode_integrator_options = {
        'name': 'zvode',
        'method': 'bdf',
        'atol': dynamics.atol, 
        'rtol': dynamics.rtol,
        'nsteps': MAX_STEPS
    }
    
    if max_step is not None:
        assert_valid_real_positive_value(max_step)
        ode_integrator_options['max_step'] = max_step
    if min_step is not None:
        assert_valid_real_positive_value(min_step)
        if max_step is not None and max_step < min_step:
            raise ValueError(f"The {max_step=} is smaller than the {min_step=}.")
        else:
            ode_integrator_options['min_step'] = min_step
    return ode_integrator_options

def run_dynamics_zvode(
    dynamics: Dynamics,
    save_every: int = 10,
    break_condition: Callable[[float, State], bool] = lambda t, s: False,
    filename: Optional[str] = None,
    restart: Optional[str] = None,
    restart_every: Optional[int] = None,
):
    # check whether the dynamics file already exists
    if filename is not None:
        # if the file exists and it's not empty, we skip the calculation
        if os.path.exists(filename):
            if not is_empty_ncdf(filename):
                return None
            
    # the initial dynamic variables
    t: float = dynamics.t0
    y: GenericVector = dynamics.s0.flatten()
    s: State = dynamics.s0.from_unstructured(y)
    
    # the ODE solver: scipy's ZVODE wrapper
    ode_solver = ode(dynamics.deriv_wrapper).set_integrator(**get_ode_solver_options(dynamics))
    ode_solver.set_initial_value(y, t)
    
    # integration step wrapper
    def step_zvode(t) -> Tuple[float, State]:
        ########
        # Integration using the zvode solver (call the fortran code)
        ########
        if not ode_solver.successful():
            raise RuntimeError("The ode solver is not successful.")
        
        ode_solver.integrate(ode_solver.t + dynamics.dt) 
        t += dynamics.dt
        
        ########
        # Callback step for the post-step
        ######## 
        current_state = dynamics.s0.from_unstructured(ode_solver.y)
        current_state_after_callback, update_integrator = dynamics.solver.callback(t, current_state)
        if update_integrator:
            # print(f"Hopping happed. Change the integrator.")
            ode_solver.set_initial_value(current_state_after_callback.flatten(), t)
        
        ########
        # Langevin dynamics step    
        ########
        R, P, rho = current_state_after_callback.get_variables()
        F_langevin = dynamics.langevin.evaluate_langevin(t, R, P, dynamics.dt)
        dynamics.solver.update_F_langevin(F_langevin)
        return t, current_state_after_callback
   
    _R, _, _rho = dynamics.s0.get_variables()
    traj_writer = PropertiesWriter(dim_elec=dynamics.solver.get_dim_electronic(), dim_nucl=dynamics.solver.get_dim_nuclear())
    restart_writer = RestartWriter(dim_elec=_rho.shape[0], dim_nucl=dynamics.solver.get_dim_nuclear())
    if (restart_every is not None) and not isinstance(restart_every, int):
        raise ValueError(f"The {restart_every=} is not an integer.")
    elif restart_every is None:
        restart_every = save_every * 1000 # save one frane of the restart file every 1000 steps

    # the main loop 
    for istep in range(MAX_STEPS):
        if (istep % save_every) == 0:
            properties = dynamics.solver.calculate_properties(t, s)
            traj_writer.write_frame(t=t, R=properties.R, P=properties.P, adiabatic_populations=properties.adiabatic_populations, diabatic_populations=properties.diabatic_populations, KE=properties.KE, PE=properties.PE)
            if break_condition(t, s):
                # write the last frame of the restart file
                restart_writer.write_frame(t=t, R=s.get_R(), P=s.get_P(), rho=s.get_rho())
                break
        if (istep % restart_every) == 0:
            restart_writer.write_frame(t=t, R=s.get_R(), P=s.get_P(), rho=s.get_rho())
        t, s = step_zvode(t)
    if filename is None:
        warnings.warn(f"You haven't provided an directory for the output file, you'll get nothing. Nonetheless, you can find the temperary data file at {writer.fn}.")
    else:
        traj_writer.save(filename)
        restart_writer.save(restart)

        
def run_dynamics_rk4(
    dynamics: Dynamics,
    save_every: int = 10,
    break_condition: Callable[[float, State], bool] = lambda t, x: False,
    filename: Optional[str] = None, 
    restart: Optional[str] = None,
    restart_every: Optional[int] = None,
):
    # check whether the dynamics file already exists
    if filename is not None:
        # if the file exists and it's not empty, we skip the calculation
        if os.path.exists(filename):
            if not is_empty_ncdf(filename):
                return None
            
    # the initial dynamic variables
    t: float = dynamics.t0
    y: GenericVector = dynamics.s0.flatten()
    s: State = dynamics.s0.from_unstructured(y)
    
    # integration step wrapper
    def step_rk4(t, s) -> Tuple[float, State]:
        ########
        # Integration using the Runge-Kutta 4th order method
        ########
        t, s = state_rk4(t, s, dynamics.solver.derivative, dt=dynamics.dt)
        
        ########
        # Callback step for the post-step
        ######## 
        s, update_integrator = dynamics.solver.callback(t, s)
        
        ########
        # Langevin dynamics step    
        ########
        R, P, rho = s.get_variables()
        F_langevin = dynamics.langevin.evaluate_langevin(t, R, P, dynamics.dt)
        dynamics.solver.update_F_langevin(F_langevin)
        return t, s
       
    _R, _, _rho = dynamics.s0.get_variables()
    traj_writer = PropertiesWriter(dim_elec=dynamics.solver.get_dim_electronic(), dim_nucl=dynamics.solver.get_dim_nuclear())
    restart_writer = RestartWriter(dim_elec=_rho.shape[0], dim_nucl=dynamics.solver.get_dim_nuclear())
    
    if (restart_every is not None) and not isinstance(restart_every, int):
        raise ValueError(f"The {restart_every=} is not an integer.")
    elif restart_every is None:
        restart_every = save_every * 1000
    
    import numpy as np
    from pymddrive.models.floquet import get_rhoF
    # the main loop 
    for istep in range(MAX_STEPS):
        if (istep % save_every) == 0:
            properties = dynamics.solver.calculate_properties(t, s)
            traj_writer.write_frame(t=t, R=properties.R, P=properties.P, adiabatic_populations=properties.adiabatic_populations, diabatic_populations=properties.diabatic_populations, KE=properties.KE, PE=properties.PE) 
            if break_condition(t, s):
                restart_writer.write_frame(t=t, R=s.get_R(), P=s.get_P(), rho=s.get_rho())
                break
        if (istep % restart_every) == 0:
            restart_writer.write_frame(t=t, R=s.get_R(), P=s.get_P(), rho=s.get_rho())
        t, s = step_rk4(t, s) 
    
    if filename is None:
        warnings.warn(f"You haven't provided an directory for the output file, you'll get nothing. Nonetheless, you can find the temperary data file at {writer.fn}.")
    else:
        traj_writer.save(filename)
        restart_writer.save(restart)
        
def run_ensemble(
    dynamics_list: List[Dynamics],
    save_every: int = 10,
    break_condition: Callable[[float, State], bool] = lambda t, x: False,
    filename: Optional[str] = None,
    numerical_integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
    mode: str = 'normal', # normal or append
    parallel: bool = True,
) -> None:
    if numerical_integrator == NumericalIntegrators.ZVODE:
        runner = run_dynamics_zvode
    elif numerical_integrator == NumericalIntegrators.RK4:
        runner = run_dynamics_rk4
    else:
        raise ValueError(f"Numerical integrator {numerical_integrator.name} has not been implemented yet.")
    
    ntrajectories = len(dynamics_list) 
    if mode == 'normal':
        filename_list = [numerate_file_name(filename, i) for i in range(ntrajectories)]
        restart_list = [numerate_file_name(filename, i, suffix='restart') for i in range(ntrajectories)]
    elif mode == 'append':
        file_dirname = os.path.dirname(filename)
        # the file pattern is like "*.*.nc"
        # where the second * is the index of the trajectory
        # represented by numers like 000, 001, 002, ...
        file_pattern = "*.?????.nc"
        current_files = glob.glob(os.path.join(file_dirname, file_pattern))
        N_current = len(current_files)
        filename_list = [numerate_file_name(filename, i+N_current) for i in range(ntrajectories)]
        restart_list = [numerate_file_name(filename, i+N_current, suffix='restart') for i in range(ntrajectories)]
    if parallel: 
        Parallel(n_jobs=get_ncpus(), verbose=5)(delayed(runner)(dynamics, save_every, break_condition, filename, restart) for dynamics, filename, restart in zip(dynamics_list, filename_list, restart_list))
    else:
        for dynamics, filename, restart in zip(dynamics_list, filename_list, restart_list):
            runner(dynamics, save_every, break_condition, filename, restart)
    
    

def test_main_tullyone():
    import numpy as np
    from pymddrive.integrators.state import get_state
    from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes
    from pymddrive.dynamics.nonadiabatic_solvers import Ehrenfest
    from pymddrive.dynamics.nonadiabatic_solvers import FSSH
    from pymddrive.dynamics.options import BasisRepresentation
    
    t0 = 0.0 
    
    R = -10.0
    P = 30
    rho_dummy = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    mass = 2000.0
    dt = 0.02
    
    def run_one_dynamics(runner: Callable, basis_representation: BasisRepresentation, filename: str):
        s0 = get_state(mass, R, P, rho_dummy)
        hamiltonian = get_tullyone()
        # solver = Ehrenfest.initialize(
        #     state=s0,
        #     hamiltonian=hamiltonian,
        #     basis_representation=basis_representation,
        # )
        solver = FSSH.initialize(
            state=s0,
            hamiltonian=hamiltonian,
            basis_representation=basis_representation,
            dt=dt
        )
    
        dyn = Dynamics(t0=t0, s0=s0, solver=solver, dt=dt)
    
        def break_condition(t:float , s: State) -> bool:
            R, P, rho = s.get_variables()
            return (R[0] > 10.0) or (R[0] < -10.0)
    
        import time
    
    
        start = time.perf_counter()
        runner(dyn, save_every=30, break_condition=break_condition, filename=filename)
        end = time.perf_counter()
        print(f"Elapsed time for runner {runner.__name__} is: {end - start}")
    
    def run_one_floquet_dynamics(runner: Callable, basis_representation: BasisRepresentation, filename: str, Omega: float, tau: float, delay: float):
        import scipy.sparse as sp
        rho_floquet = sp.block_diag([np.zeros_like(rho_dummy), rho_dummy, np.zeros_like(rho_dummy)]).toarray()
        s0 = get_state(mass, R, P, rho_floquet)
        hamiltonian = get_tullyone(
            Omega=Omega, 
            tau=tau, 
            t0=delay,
            pulse_type=TullyOnePulseTypes.PULSE_TYPE3, 
            NF=1,
        )
        
        solver = Ehrenfest.initialize(
            state=s0,
            hamiltonian=hamiltonian,
            basis_representation=basis_representation
        )
    
        dyn = Dynamics(t0=t0, s0=s0, solver=solver, dt=dt)
    
        def break_condition(t: float, s: State) -> bool:
            R, P, rho = s.get_variables()
            return (R[0] > 10.0) or (R[0] < -10.0)
    
        import time
    
    
        start = time.perf_counter()
        runner(dyn, save_every=30, break_condition=break_condition, filename=filename)
        end = time.perf_counter()
        print(f"Elapsed time for runner {runner.__name__} is: {end - start}") 
    
    
    
    dynamics_basis = BasisRepresentation.ADIABATIC
    # dynamics_basis = BasisRepresentation.DIABATIC
    filename = "test_ehrenfest.nc"
    Omega = 0.3
    tau = 100.0
    # run_one_floquet_dynamics(runner=run_dynamics_zvode, basis_representation=dynamics_basis, filename=filename, Omega=Omega, tau=tau, delay=600)
    run_one_dynamics(runner=run_dynamics_zvode, basis_representation=dynamics_basis, filename=filename)
    # run_one_dynamics(runner=run_dynamics_rk4, basis_representation=dynamics_basis, filename=filename)
     
    from netCDF4 import Dataset 
    import matplotlib.pyplot as plt
    
    nc = Dataset(filename, 'r')
    t = np.array(nc.variables['time'])
    R = np.array(nc.variables['R']) 
    P = np.array(nc.variables['P']) 
    adiabatic_populations = np.array(nc.variables['adiabatic_populations']) 
    diabatic_populations = np.array(nc.variables['diabatic_populations']) 
    KE = np.array(nc.variables['KE'])  
    PE = np.array(nc.variables['PE']) 
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(t, R)
    ax.set_xlabel("Time")
    ax.set_ylabel("R")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(t, P)
    ax.set_xlabel("Time")
    ax.set_ylabel("P")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for ii in range(adiabatic_populations.shape[1]):
        ax.plot(t, adiabatic_populations[:, ii], label=rf"$P_{ii}$")
    ax.legend() 
    ax.set_xlabel("Time")
    ax.set_title("Adiabatic populations")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for ii in range(diabatic_populations.shape[1]):
        ax.plot(t, diabatic_populations[:, ii], label=rf"$P_{ii}$")
    ax.legend() 
    ax.set_xlabel("Time")
    ax.set_title("Diabatic populations")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(t, KE)
    ax.legend() 
    ax.set_xlabel("Time")
    ax.set_ylabel("Kinetic energy")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(t, PE)
    ax.legend() 
    ax.set_xlabel("Time")
    ax.set_ylabel("Potential energy")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(t, KE+PE)
    ax.legend() 
    ax.set_xlabel("Time")
    ax.set_ylabel("Total energy")
    plt.show()
    
def test_main_landry_spin_boson():
    import numpy as np
    from pymddrive.dynamics.options import BasisRepresentation
    from pymddrive.models.landry_spin_boson import LandrySpinBoson
    from pymddrive.integrators.state import get_state
    from pymddrive.dynamics.nonadiabatic_solvers import Ehrenfest
    from pymddrive.dynamics.langevin import Langevin
    
    def stop_condition(t, s):
        return t > 3000
    
    def sample_lsb_boltzmann(
        n_samples: int, 
        lsb_hamiltonian: LandrySpinBoson,
        initialize_from_donor: bool =True
    ):
        # constants
        kT: float = lsb_hamiltonian.kT
        mass: float = lsb_hamiltonian.M
        beta: float = 1.0 / kT
        Omega_nuclear: float = lsb_hamiltonian.Omega_nuclear

        # sample the momentum
        sigma_momentum = np.sqrt(mass / beta)
        momentum_samples = np.random.normal(0, sigma_momentum, n_samples)

        R0 = lsb_hamiltonian.get_donor_R() if initialize_from_donor else lsb_hamiltonian.get_acceptor_R()
        sigma_R = 1.0 / np.sqrt(beta * mass) / Omega_nuclear
        position_samples = np.random.normal(R0, sigma_R, n_samples)
        return position_samples, momentum_samples
    
    hamiltonian = LandrySpinBoson() 
    dim = hamiltonian.dim
    rho_diabatic = np.zeros((dim, dim), dtype=np.complex128)
    rho_diabatic[0, 0] = 1.0
    dynamics_basis = BasisRepresentation.ADIABATIC
    mass = 1.0
    
    t0 = 0.0
    R, P = sample_lsb_boltzmann(1, hamiltonian)
    s0 = get_state(mass, R, P, rho_diabatic)
    
    ehrenfest = Ehrenfest.initialize(
        state=s0,
        hamiltonian=hamiltonian,
        basis_representation=dynamics_basis
    )
    langevin = Langevin(kT=hamiltonian.get_kT(), mass=s0.get_mass(), gamma=hamiltonian.get_friction())
    dynamics = Dynamics(t0=t0, s0=s0, solver=ehrenfest, dt=0.03, langevin=langevin)
    
    filename = "test_landry_spin_boson.nc" 
    import time
    start = time.perf_counter()
    # run_dynamics_zvode(dynamics, save_every=10, break_condition=stop_condition, filename=filename)
    run_dynamics_rk4(dynamics, save_every=10, break_condition=stop_condition, filename=filename)
    end = time.perf_counter()
    print(f"Elapsed time for runner run_dynamics_zvode is: {end - start}")
    
    from netCDF4 import Dataset 
    import matplotlib.pyplot as plt
    
    nc = Dataset(filename, 'r')
    t = np.array(nc.variables['time'])
    R = np.array(nc.variables['R']) 
    P = np.array(nc.variables['P']) 
    adiabatic_populations = np.array(nc.variables['adiabatic_populations']) 
    diabatic_populations = np.array(nc.variables['diabatic_populations']) 
    KE = np.array(nc.variables['KE'])  
    PE = np.array(nc.variables['PE']) 
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(t, R)
    ax.set_xlabel("Time")
    ax.set_ylabel("R")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(t, P)
    ax.set_xlabel("Time")
    ax.set_ylabel("P")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for ii in range(adiabatic_populations.shape[1]):
        ax.plot(t, adiabatic_populations[:, ii], label=rf"$P_{ii}$")
    ax.legend() 
    ax.set_xlabel("Time")
    ax.set_title("Adiabatic populations")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    for ii in range(diabatic_populations.shape[1]):
        ax.plot(t, diabatic_populations[:, ii], label=rf"$P_{ii}$")
    ax.legend() 
    ax.set_xlabel("Time")
    ax.set_title("Diabatic populations")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(t, KE)
    ax.legend() 
    ax.set_xlabel("Time")
    ax.set_ylabel("Kinetic energy")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(t, PE)
    ax.legend() 
    ax.set_xlabel("Time")
    ax.set_ylabel("Potential energy")
    plt.show()
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(t, KE+PE)
    ax.legend() 
    ax.set_xlabel("Time")
    ax.set_ylabel("Total energy")
    plt.show()
    
    
# %%
if __name__ == "__main__":
    test_main_tullyone()
    # test_main_landry_spin_boson()
    

# %%
