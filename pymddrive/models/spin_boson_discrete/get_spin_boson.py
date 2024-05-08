from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.models.spin_boson_discrete.discretize_debye import discretize_Debye_bath
from pymddrive.models.spin_boson_discrete.parameter_sets import TempelaarJCP2018
from pymddrive.models.spin_boson_discrete.spin_boson import SpinBoson

def get_spin_boson(
    n_classic_bath: int = 100,
    param_set: str="TempelaarJCP2018",
    pulse_type: str = 'no_pulse',
) -> HamiltonianBase:
    if param_set == "TempelaarJCP2018":
        params = TempelaarJCP2018()
        E, V, Omega, lambd, kT = params.E, params.V, params.Omega, params.lambd, params.kT
    else:
        raise ValueError(f"The parameters set: {param_set} is not recognized.")
    
    # initialize the bath parameters
    omega_alpha, g_alpha = discretize_Debye_bath(lambd, Omega, n_classic_bath)
    if pulse_type == 'no_pulse':
        return SpinBoson(omega_alpha=omega_alpha, g_alpha=g_alpha, E=E, V=V, Omega=Omega, lambd=lambd, kT=kT)
    else:
        raise NotImplementedError(f"The pulse type: {pulse_type} is not implemented yet.")
    