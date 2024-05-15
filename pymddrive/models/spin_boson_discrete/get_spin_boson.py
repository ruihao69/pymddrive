import numpy as np

from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.models.spin_boson_discrete.discretize_debye import discretize_Debye_bath
from pymddrive.models.spin_boson_discrete.parameter_sets import TempelaarJCP2018, BiasedTempelaarJCP2018, BiasedTempelaarJCP2018Pulsed
from pymddrive.models.spin_boson_discrete.spin_boson import SpinBoson
from pymddrive.models.spin_boson_discrete.spin_boson_pulsed import SpinBosonPulsed
from pymddrive.models.spin_boson_discrete.spin_boson_pulsed_floquet import SpinBosonPulsedFloquet
from pymddrive.models.spin_boson_discrete.spin_boson_pulse_types import SpinBosonPulseTypes
from pymddrive.pulses import SineSquarePulse, SineSquareEnvelope, MorletReal, Gaussian

from typing import Optional

def get_spin_boson(
    n_classic_bath: int = 100,
    param_set: str="TempelaarJCP2018",
    pulse_type: str = 'no_pulse',
    E0_in_au: Optional[float] = None,
    Nc: Optional[int] = None,
    phi: Optional[float] = None,
    t0: Optional[float] = None,
    NF: Optional[int] = None,
    mu_in_au: Optional[float] = None,
) -> HamiltonianBase:
    if param_set == "TempelaarJCP2018":
        params = TempelaarJCP2018()
        E, V, Omega, lambd, kT = params.E, params.V, params.Omega, params.lambd, params.kT
    elif param_set == "BiasedTempelaarJCP2018":
        params = BiasedTempelaarJCP2018()
        E, V, Omega, lambd, kT = params.E, params.V, params.Omega, params.lambd, params.kT
    elif param_set == "BiasedTempelaarJCP2018Pulsed":
        params = BiasedTempelaarJCP2018Pulsed()
        E, V, Omega, lambd, kT, mu_in_au = params.E, params.V, params.Omega, params.lambd, params.kT, params.mu_in_au
    else:
        raise ValueError(f"The parameters set: {param_set} is not recognized.")
    
    # get the pulse parameters
    try: 
        pulse_type_enum = SpinBosonPulseTypes[pulse_type.upper()]
    except KeyError:
        raise ValueError(f"Invalid pulse type: {pulse_type}. The valid pulse types are: {SpinBosonPulseTypes.__members__.keys()}")
    
    # initialize the bath parameters
    omega_alpha, g_alpha = discretize_Debye_bath(lambd, Omega, n_classic_bath)
    if pulse_type_enum == SpinBosonPulseTypes.NO_PULSE:
        return SpinBoson(omega_alpha=omega_alpha, g_alpha=g_alpha, E=E, V=V, Omega=Omega, lambd=lambd, kT=kT)
    elif pulse_type_enum == SpinBosonPulseTypes.SINE_SQUARED_PULSE:
        if not all([
            isinstance(E0_in_au, float),
            isinstance(Nc, int),
            isinstance(phi, float),
            isinstance(mu_in_au, float) 
        ]):
            raise ValueError("For a sine-squared pulse, the E0_in_au, Nc, and phi parameters must be provided.")
        ultrafast_pulse = SineSquarePulse(Omega=2*E, A=E0_in_au, N=Nc, phi=phi)
        envelope_pulse = SineSquareEnvelope.from_quasi_floquet_sine_square_pulse(ultrafast_pulse)
        if NF is None:
            return SpinBosonPulsed(omega_alpha=omega_alpha, g_alpha=g_alpha, E=E, V=V, Omega=Omega, lambd=lambd, kT=kT, mu_in_au=mu_in_au, pulse=ultrafast_pulse)
        elif isinstance(NF, int):
            return SpinBosonPulsedFloquet(omega_alpha=omega_alpha, g_alpha=g_alpha, E=E, V=V, Omega=Omega, lambd=lambd, kT=kT, mu_in_au=mu_in_au, ultrafast_pulse=ultrafast_pulse, envelope_pulse=envelope_pulse, NF=NF)
        else:
            raise ValueError(f"Invalid number of Floquet replicas: {NF}. Should be an integer or None. Got {type(NF)}")
    elif pulse_type_enum == SpinBosonPulseTypes.MORLET_REAL:
        if not all([
            isinstance(E0_in_au, float),
            isinstance(Nc, int),
            isinstance(phi, float),
            isinstance(t0, float),
            isinstance(mu_in_au, float)
        ]):
            raise ValueError("For a Morlet pulse, the E0_in_au, Nc, phi, and t0 parameters must be provided.")
        T_period = 2 * np.pi / (2*E)
        # this is adhoc, but we need to make sure that the pulse is centered at t0
        tau = T_period * Nc / 3
        ultrafast_pulse = MorletReal(Omega=2*E, A=E0_in_au, phi=phi, t0=t0, tau=tau)
        from pymddrive.pulses import get_carrier_frequency
        print(f"{get_carrier_frequency(ultrafast_pulse)=}")
        envelope_pulse = Gaussian.from_quasi_floquet_morlet_real(ultrafast_pulse)
        if NF is None:
            return SpinBosonPulsed(omega_alpha=omega_alpha, g_alpha=g_alpha, E=E, V=V, Omega=Omega, lambd=lambd, kT=kT, mu_in_au=mu_in_au, pulse=ultrafast_pulse)
        elif isinstance(NF, int):
            return SpinBosonPulsedFloquet(omega_alpha=omega_alpha, g_alpha=g_alpha, E=E, V=V, Omega=Omega, lambd=lambd, kT=kT, mu_in_au=mu_in_au, ultrafast_pulse=ultrafast_pulse, envelope_pulse=envelope_pulse, NF=NF)
        else:
            raise ValueError(f"Invalid number of Floquet replicas: {NF}. Should be an integer or None. Got {type(NF)}")
    else:
        raise NotImplementedError(f"The pulse type: {pulse_type} is not implemented yet.")
    