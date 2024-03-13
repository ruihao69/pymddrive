# %%
import os 
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple
from pymddrive.models.tullyone import get_tullyone, TullyOnePulseTypes
from pymddrive.integrators.state import State
from pymddrive.dynamics.options import BasisRepresentation, QunatumRepresentation, NonadiabaticDynamicsMethods, NumericalIntegrators    
from pymddrive.dynamics import NonadiabaticDynamics, run_nonadiabatic_dynamics, run_nonadiabatic_dynamics_ensembles
from pymddrive.dynamics.misc_utils import eval_nonadiabatic_hamiltonian

from tullyone_utils import *

def stop_condition(t, s, states):
    r, _, _ = s.get_variables()
    return outside_boundary(r, (-10, 10))

def break_condition(t, s, states):
    return False

def run_tullyone(
    r0: float, 
    p0: float, 
    mass: float=2000, 
    qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
    basis_rep: BasisRepresentation = BasisRepresentation.Adiabatic,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
):
    # intialize the model
    hamiltonian = get_tullyone(pulse_type=TullyOnePulseTypes.NO_PULSE)
    
    # initialize the states
    rho0 = np.array([[1.0, 0], [0, 0.0]], dtype=np.complex128)
    s0 = State.from_variables(R=r0, P=p0, rho=rho0)
    
    dyn = NonadiabaticDynamics(
        hamiltonian=hamiltonian,
        t0=0.0,
        s0=s0,
        mass=mass,
        basis_rep=basis_rep,
        qm_rep=qm_rep,
        solver=solver,
        numerical_integrator=integrator,
        dt=0.03,
        save_every=30
    )
    
    return run_nonadiabatic_dynamics(dyn, stop_condition, break_condition) 

def main(sim_signature: str, n_samples: int, solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST):
    import os
    from pararun import ParaRunScatter, get_ncpus
    
    ncpus = get_ncpus()
    if not os.path.exists(sim_signature):
        os.makedirs(sim_signature)
        
    r0 = -10.0
    _r0_list = np.array([r0]*n_samples)
    _p0_list = get_tully_one_p0_list(n_samples, pulse_type=TullyOnePulseTypes.NO_PULSE)
    
    runner = ParaRunScatter(n_jobs=ncpus, r0=_r0_list, p0=_p0_list)
    
    res_gen = runner.run(run_tullyone, accumulate_output, sim_signature)
    traj_dict, pulses = accumulate_output(_p0_list, res_gen)
    
    post_process_output(sim_signature, traj_dict, pulse_list=pulses)
    

def generate_ensembles(
    R0_samples: ArrayLike,
    P0_samples: ArrayLike,
    mass: float=2000,
    qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
    basis_rep: BasisRepresentation = BasisRepresentation.Diabatic,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
) -> Tuple[NonadiabaticDynamics]:
    # initialize the electronic states
    rho0_diabatic = np.zeros((2, 2), dtype=np.complex128)
    rho0_diabatic[0, 0] = 1.0
    
    assert (n_samples := len(R0_samples)) == len(P0_samples), "The number of R0 and P0 samples should be the same."
    ensemble = ()
    for ii, (R0, P0) in enumerate(zip(R0_samples, P0_samples)):
        hamiltonian = get_tullyone(
            pulse_type=TullyOnePulseTypes.NO_PULSE
        )
        if basis_rep == BasisRepresentation.Diabatic:
            s0 = State.from_variables(R=R0, P=P0, rho=rho0_diabatic)
        else:
            hami_return = eval_nonadiabatic_hamiltonian(0, np.array([R0]), hamiltonian, basis_rep=BasisRepresentation.Diabatic)
            evecs = hami_return.evecs
            rho0_adiabatic = evecs.T.conj() @ rho0_diabatic @ evecs
            s0 = State.from_variables(R=R0, P=P0, rho=rho0_adiabatic)
        dyn = NonadiabaticDynamics(
            hamiltonian=hamiltonian,
            t0=0.0,
            s0=s0,
            mass=mass,
            basis_rep=basis_rep,
            qm_rep=qm_rep,
            solver=solver,
            numerical_integrator=integrator,
            dt=0.1,
            save_every=10,
        )
        ensemble += (dyn,)
    return ensemble

   

# %%
# if __name__ == "__main__": 
#     from pymddrive.models.tullyone import TullyOnePulseTypes
#     sim_signature = "data_tullyone"
#     nsamples = 10
#     
#     main(sim_signature, nsamples)
#     p0_list, sr_list = load_data_for_plotting(os.path.join(sim_signature, 'scatter.dat'))
# %%

if __name__ == "__main__": 
    sim_signature = "data_tullyone"
    
    r0 = -5.0
    p0 = 30.0
    ouput_ehrenfest = run_tullyone(r0, p0, solver=NonadiabaticDynamicsMethods.EHRENFEST)
    # ouput_fssh = run_tullyone(r0, p0, solver=NonadiabaticDynamicsMethods.FSSH) 
    
    nsamples = 1000
    R0_samples = np.array([r0]*nsamples)
    P0_samples = np.array([p0]*nsamples)
    dyn_ensemble = generate_ensembles(R0_samples, P0_samples, solver=NonadiabaticDynamicsMethods.FSSH, basis_rep=BasisRepresentation.Adiabatic)
    output_fssh = run_nonadiabatic_dynamics_ensembles(dyn_ensemble, stop_condition, break_condition, inhomogeneous=True)
# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    time_e = ouput_ehrenfest['time']
    time_fssh = output_fssh['time']
    adiab_population_e = ouput_ehrenfest['adiab_populations']
    adiab_population_fssh = output_fssh['adiab_populations']
    plt.plot(time_e, adiab_population_e[:, 0], label="state 1")
    plt.plot(time_e, adiab_population_e[:, 1], label="state 2")
    plt.plot(time_fssh, adiab_population_fssh[:, 0], label="state 1", ls='--')
    plt.plot(time_fssh, adiab_population_fssh[:, 1], label="state 2", ls='--')
    plt.axhline(0.28, ls='-.', color='k', label='state 1 final')
    plt.axhline(0.72, ls='-.', color='k', label='state 2 final')
    plt.legend()
    plt.show()
    
    KE_e = ouput_ehrenfest['KE']
    PE_e = ouput_ehrenfest['PE']
    KE_fssh = output_fssh['KE']
    PE_fssh = output_fssh['PE']
    plt.plot(time_e, KE_e, label="KE")
    plt.plot(time_fssh, KE_fssh, label="KE", ls='--')
    plt.show()
    plt.plot(time_e, PE_e, label="PE")
    plt.plot(time_fssh, PE_fssh, label="PE", ls='--')
    plt.show()
    
    TE_e = KE_e + PE_e
    TE_fssh = KE_fssh + PE_fssh
    
    plt.plot(time_e, TE_e-TE_e[0], label="TE")
    plt.plot(time_fssh, TE_fssh-TE_e[0], label="TE", ls='--')
    plt.show()
    
    


# %%
