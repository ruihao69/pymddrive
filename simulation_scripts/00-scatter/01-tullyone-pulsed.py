# %% 
import numpy as np

from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes
from pymddrive.integrators.state import get_state
from pymddrive.dynamics.options import BasisRepresentation, NonadiabaticDynamicsMethods, NumericalIntegrators  
from pymddrive.dynamics.get_dynamics import get_dynamics
from pymddrive.dynamics.run import run_ensemble

from tullyone_utils import outside_boundary, get_tully_one_p0_list, get_tully_one_delay_time
from scatter_postprocess import post_process

import os
import time
import argparse
from typing import Optional

def stop_condition(t, s):
    r, _, _ = s.get_variables()
    return outside_boundary(r, (-10, 10))

def run_tullyone_pulsed(
    r0: float, 
    p0: float, 
    Omega: float,
    tau: float,
    pulse_type: TullyOnePulseTypes, 
    NF: Optional[int]=None,
    n_ensemble: int=100,
    mass: float=2000, 
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.FSSH,
    data_dir: str='./',
    basis_rep: BasisRepresentation = BasisRepresentation.ADIABATIC,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
    dt: float = 0.1,
    mode: str = 'normal',
) -> None:
    # get the delay time from cubic spline interpolation, and generate hamiltonian
    delay_time = get_tully_one_delay_time(R0=r0, P0=p0)
    hamiltonian = get_tullyone(
        t0=delay_time, Omega=Omega, tau=tau,
        pulse_type=pulse_type, NF=NF
    )
    
    # get the initial time and state object
    t0 = 0.0
    
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = 1
    if NF is not None:
        import scipy.sparse as sp
        zeros_like = np.zeros_like(rho0)
        rho0 = sp.block_diag([zeros_like]*NF + [rho0] + [zeros_like]*NF).toarray()
    s0 = get_state(mass=mass, R=r0, P=p0, rho_or_psi=rho0)
    if basis_rep == BasisRepresentation.ADIABATIC:
        _, _, rho0 = s0.get_variables()
        H = hamiltonian.H(t0, np.array([r0]))
        evals, evecs = np.linalg.eigh(H)
        rho0_adiabatic = evecs.T.conj() @ rho0 @ evecs
        s0 = get_state(mass=mass, R=r0, P=p0, rho_or_psi=rho0_adiabatic)
        
    # get the dynamics object
    dynamic_list = []
    for _ in range(n_ensemble):
        dyn = get_dynamics(t0=t0, s0=s0, dt=dt, hamiltonian=hamiltonian, dynamics_basis=basis_rep, method=solver)
        dynamic_list.append(dyn)
    
    # use run ensemble to run the dynamics
    momentum_signature = f"P0-{p0:.6f}"
    if solver == NonadiabaticDynamicsMethods.EHRENFEST:
        filename = "ehrenfest.nc"
    elif solver == NonadiabaticDynamicsMethods.FSSH:
        filename = "fssh.nc"
    else:
        raise ValueError(f"Unknown solver {solver}")
            
    filename = os.path.join(data_dir, momentum_signature, filename)
    data_files_dir = os.path.dirname(filename)
    if not os.path.isdir(data_files_dir):
        os.makedirs(data_files_dir)
        
    run_ensemble(
        dynamics_list=dynamic_list,
        break_condition=stop_condition,
        filename=filename,
        numerical_integrator=integrator,
        save_every=10,
        mode=mode
    )
    
    post_process(data_files_dir)

def main(
    Omega: float,
    tau: float,
    pulse_type: TullyOnePulseTypes,
    sim_signature: str,
    dynamics_method: NonadiabaticDynamicsMethods,
    basis_rep: BasisRepresentation,
    n_initial_momentum_samples: int=40,
    ensemble_size: int=10,
    NF: Optional[int]=None, 
    mode: str='normal'
):
    r0 = -10.0
    p0_list = get_tully_one_p0_list(n_initial_momentum_samples, pulse_type=pulse_type)
    for p0 in p0_list:
        run_tullyone_pulsed(
            r0=r0, p0=p0, Omega=Omega, tau=tau, pulse_type=pulse_type, 
            n_ensemble=ensemble_size, solver=dynamics_method, basis_rep=basis_rep, data_dir=sim_signature, NF=NF, mode=mode
        )
    
    
# %% 
if __name__ == "__main__":
    
    # desc = "The parser for TullyOne with Pulse One"
    # parser = argparse.ArgumentParser(description=desc)
    # parser.add_argument('--Omega', type=float, help='The Omega value')
    # parser.add_argument('--tau', type=float, help='The tau value')
    # parser.add_argument('--pulse_type', type=int, help='The Pulse Type (1, 2, or 3)')
    # args = parser.parse_args() 
    
    # Omega, tau, pulse_type= args.Omega, args.tau, args.pulse_type
    Omega, tau, pulse_num = 0.05, 100, 3
    
    if pulse_num == 1:
        pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.PULSE_TYPE1
        # sim_signature = f"data_tullyone_pulseone-Omega-{Omega}-tau-{tau}"
    elif pulse_num == 2:
        pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.PULSE_TYPE2
        # sim_signature = f"data_tullyone_pulsetwo-Omega-{Omega}-tau-{tau}"
    elif pulse_num == 3:
        pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.PULSE_TYPE3
        # sim_signature = f"data_tullyone_pulsethree-Omega-{Omega}-tau-{tau}"
    else:
        raise ValueError(f"The pulse_type must be 1, 2, or 3. But it is {pulse_num}.")
    sim_signature = f"data_floquet_fssh-Omega-{Omega}-tau-{tau}-pulse-{pulse_num}"
    nsamples = 40
    # n_ensemble = 1
    n_ensemble = 8
    main(
        Omega=Omega, 
        tau=tau, 
        pulse_type=pulse_type, 
        sim_signature=sim_signature,
        # dynamics_method=NonadiabaticDynamicsMethods.EHRENFEST,
        # basis_rep=BasisRepresentation.DIABATIC,
        dynamics_method=NonadiabaticDynamicsMethods.FSSH,
        basis_rep=BasisRepresentation.ADIABATIC,
        n_initial_momentum_samples=nsamples,
        ensemble_size=n_ensemble,
        NF=1,
    )

# %%
