# %%
import numpy as np

from pymddrive.models.tullyone import get_tullyone
from pymddrive.low_level.states import State
from pymddrive.integrators.state import get_state
from pymddrive.dynamics.options import BasisRepresentation, NonadiabaticDynamicsMethods, NumericalIntegrators  
from pymddrive.dynamics.get_dynamics import get_dynamics
from pymddrive.dynamics.run import run_ensemble

from tullyone_utils import outside_boundary

def stop_condition(t, s):
    r, _, _ = s.get_variables()
    return outside_boundary(r, (-5, 5))

def run_tullyone_fssh(
    r0: float, 
    p0: float, 
    n_ensemble: int=100,
    mass: float=2000, 
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.FSSH,
    filename: str="fssh.nc",
    basis_rep: BasisRepresentation = BasisRepresentation.ADIABATIC,
    integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
    dt: float = 0.1,
) -> None:
    # get the initial time and state object
    t0 = 0.0
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[0, 0] = 1
    s0 = get_state(mass=mass, R=r0, P=p0, rho_or_psi=rho0)
    
    # intialize the model
    hamiltonian = get_tullyone()
    if basis_rep == BasisRepresentation.ADIABATIC:
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
    run_ensemble(
        dynamics_list=dynamic_list,
        break_condition=stop_condition,
        filename=filename,
        numerical_integrator=integrator,
    )
       

def main(r0: float, p0: float, ntraj: int=100):
    run_tullyone_fssh(r0=r0, p0=p0, n_ensemble=ntraj)
    run_tullyone_fssh(r0=r0, p0=p0, n_ensemble=1, solver=NonadiabaticDynamicsMethods.EHRENFEST, basis_rep=BasisRepresentation.DIABATIC, filename="ehrenfest.nc")
    
# def generate_ensembles(
#     R0_samples: ArrayLike,
#     P0_samples: ArrayLike,
#     mass: float=2000,
#     qm_rep: QunatumRepresentation = QunatumRepresentation.DensityMatrix,
#     basis_rep: BasisRepresentation = BasisRepresentation.Diabatic,
#     solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
#     # integrator: NumericalIntegrators = NumericalIntegrators.ZVODE,
#     integrator: NumericalIntegrators = NumericalIntegrators.RK4,
# ) -> Tuple[NonadiabaticDynamics]:
#     # initialize the electronic states
#     rho0_diabatic = np.zeros((2, 2), dtype=np.complex128)
#     rho0_diabatic[0, 0] = 1.0
#     
#     assert (n_samples := len(R0_samples)) == len(P0_samples), "The number of R0 and P0 samples should be the same."
#     ensemble = ()
#     for ii, (R0, P0) in enumerate(zip(R0_samples, P0_samples)):
#         hamiltonian = get_tullyone(
#             pulse_type=TullyOnePulseTypes.NO_PULSE
#         )
#         if basis_rep == BasisRepresentation.Diabatic:
#             s0 = get_state(mass=mass, R=R0, P=P0, rho_or_psi=rho0_diabatic)
#         else:
#             hami_return = eval_nonadiabatic_hamiltonian(0, np.array([R0]), hamiltonian, basis_rep=BasisRepresentation.Diabatic)
#             evecs = hami_return.evecs
#             rho0_adiabatic = evecs.T.conj() @ rho0_diabatic @ evecs
#             s0 = get_state(mass=mass, R=R0, P=P0, rho_or_psi=rho0_adiabatic)
#         dyn = NonadiabaticDynamics(
#             hamiltonian=hamiltonian,
#             t0=0.0,
#             s0=s0,
#             basis_rep=basis_rep,
#             qm_rep=qm_rep,
#             solver=solver,
#             numerical_integrator=integrator,
#             dt=0.1,
#             save_every=10,
#         )
#         ensemble += (dyn,)
#     return ensemble

   

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
    main(r0, p0, 250)
    # ouput_ehrenfest = run_tullyone(r0, p0, solver=NonadiabaticDynamicsMethods.EHRENFEST)
    # ouput_fssh = run_tullyone(r0, p0, solver=NonadiabaticDynamicsMethods.FSSH) 
    
    # nsamples = 100
    # R0_samples = np.array([r0]*nsamples)
    # P0_samples = np.array([p0]*nsamples)
    # dyn_ensemble = generate_ensembles(R0_samples, P0_samples, solver=NonadiabaticDynamicsMethods.FSSH, basis_rep=BasisRepresentation.Adiabatic)
    # output_fssh = run_nonadiabatic_dynamics_ensembles(dyn_ensemble, stop_condition, break_condition, inhomogeneous=True)
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
