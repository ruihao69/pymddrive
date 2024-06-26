# %%
import numpy as np

from pymddrive.pulses import PulseBase as Pulse
# Continuous Wave pulses
from pymddrive.pulses import SineSquareEnvelope, SineSquarePulse
# Gaussian pulses
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.pulses import MorletReal, Gaussian
from pymddrive.models.landry_spin_boson.landry_spin_boson import LandrySpinBoson
from pymddrive.models.landry_spin_boson.landry_spin_boson_pulsed import LandrySpinBosonPulsed
from pymddrive.models.landry_spin_boson.landry_spin_boson_pulsed_floquet import LandrySpinBosonPulsedFloquet
from pymddrive.models.landry_spin_boson.landry_pulse_types import LandryPulseTypes
from pymddrive.models.landry_spin_boson.parameter_sets import LandryJCP2013, SymmetricDoubleWell, ResonantDoubleWell, AmberJCTC2016, MarcusNormalRegime

from typing import Optional

def get_landry_spin_boson(
    E0: Optional[float] = None,    # laser E-field amplitude
    Omega: Optional[float] = None, # laser carrier frequency
    N: Optional[int] = None,       # number of laser cycles in sine-squared pulse
    phi: Optional[float] = None,   # laser carrier phase
    t0: Optional[float] = None,    # time delay for Morlet pulse
    pulse_type: str = 'no_pulse',  # valid values are encompassed in LandryPulseTypes enum class
    NF: Optional[int] = None,      # number of Floquet replicas
    Omega_nuclear: Optional[float] = 0.021375,
    M: Optional[float] = 1.0,
    V: Optional[float] = 0.00475,
    Er: Optional[float] = 0.00475,
    epsilon0: Optional[float] = 0.00475,
    gamma: Optional[float] = 0.04275,
    kT: Optional[float] = 0.00095,
    mu: Optional[float] = 0.04,
    param_set: str = 'LandryJCP2013'
) -> HamiltonianBase:
    # get parameters from parameter set
    if param_set == 'LandryJCP2013':
        params = LandryJCP2013() 
        Omega_nuclear, M, V, Er, epsilon0, gamma, kT = params.Omega_nuclear, params.M, params.V, params.Er, params.epsilon0, params.gamma, params.kT
    elif param_set == 'SymmetricDoubleWell':
        params = SymmetricDoubleWell()
        Omega_nuclear, M, V, Er, epsilon0, gamma, kT = params.Omega_nuclear, params.M, params.V, params.Er, params.epsilon0, params.gamma, params.kT
    elif param_set == 'ResonantDoubleWell':
        params = ResonantDoubleWell()
        Omega_nuclear, M, V, Er, epsilon0, gamma, kT = params.Omega_nuclear, params.M, params.V, params.Er, params.epsilon0, params.gamma, params.kT
    elif param_set == 'AmberJCTC2016':
        params = AmberJCTC2016()
        Omega_nuclear, M, V, Er, epsilon0, gamma, kT = params.Omega_nuclear, params.M, params.V, params.Er, params.epsilon0, params.gamma, params.kT
    elif param_set == 'MarcusNormalRegime':
        params = MarcusNormalRegime()
        Omega_nuclear, M, V, Er, epsilon0, gamma, kT = params.Omega_nuclear, params.M, params.V, params.Er, params.epsilon0, params.gamma, params.kT
    else:
        raise ValueError(f"Invalid parameter set: {param_set}. The valid parameter sets for now are: ['LandryJCP2013', 'SymmetricDoubleWell']")
        
    # get pulse type from string
    try:
        pulse_type_enum = LandryPulseTypes[pulse_type.upper()]
    except KeyError:
        raise ValueError(f"Invalid pulse type: {pulse_type}. The valid pulse types are: {LandryPulseTypes.__members__.keys()}")
    
    # control flow for pulse types
    if pulse_type_enum == LandryPulseTypes.NO_PULSE:
        return LandrySpinBoson(Omega_nuclear=Omega_nuclear, M=M, V=V, Er=Er, epsilon0=epsilon0, gamma=gamma, kT=kT)
    elif pulse_type_enum == LandryPulseTypes.SINE_SQUARED_PULSE:
        # make sure all pulse parameters are provided as arguments
        if not all([
            isinstance(E0, float),
            isinstance(Omega, float),
            isinstance(N, int),
            isinstance(phi, float)
        ]):
            raise ValueError(f"Invalid pulse parameters: E0={E0}, Omega={Omega}, N={N}, phi={phi}.")
        ultrafast_pulse = SineSquarePulse(Omega=Omega, A=E0, N=N, phi=phi)
        envelope_pulse = SineSquareEnvelope.from_quasi_floquet_sine_square_pulse(ultrafast_pulse)
        if NF is None:
            # initialize the TD Hamiltonian
            return LandrySpinBosonPulsed(Omega_nuclear=Omega_nuclear, M=M, V=V, Er=Er, epsilon0=epsilon0, gamma=gamma, kT=kT, mu=mu, pulse=ultrafast_pulse)
        elif isinstance(NF, int):
            # initialize the Floquet Hamiltonian
            return LandrySpinBosonPulsedFloquet(Omega_nuclear=Omega_nuclear, M=M, V=V, Er=Er, epsilon0=epsilon0, gamma=gamma, kT=kT, mu=mu, ultrafast_pulse=ultrafast_pulse, envelope_pulse=envelope_pulse, NF=NF)
        else:
            raise ValueError(f"Invalid number of Floquet replicas: {NF}. Should be an integer or None. Got {type(NF)}")
    elif pulse_type_enum == LandryPulseTypes.MORLET_REAL:
        # raise NotImplementedError(f"Only CW(sine-squared) pulses are implemented for now.")
        if not all([
            isinstance(E0, float),
            isinstance(Omega, float),
            isinstance(N, int),
            isinstance(phi, float),
            isinstance(t0, float),
        ]): 
            raise ValueError(f"Invalid pulse parameters: E0={E0}, Omega={Omega}, N={N}, phi={phi}, t0={t0}.")
        T_period = 2 * np.pi / Omega
        # this is ad hoc
        tau = T_period * N / 3 # pulse duration defaults to N times the period of the carrier frequency
        ultrafast_pulse = MorletReal(Omega=Omega, A=E0, t0=t0, tau=tau, phi=phi)
        envelope_pulse = Gaussian.from_quasi_floquet_morlet_real(ultrafast_pulse)
        if NF is None:
            # initialize the TD Hamiltonian
            return LandrySpinBosonPulsed(Omega_nuclear=Omega_nuclear, M=M, V=V, Er=Er, epsilon0=epsilon0, gamma=gamma, kT=kT, mu=mu, pulse=ultrafast_pulse)
        elif isinstance(NF, int):
            # initialize the Floquet Hamiltonian
            return LandrySpinBosonPulsedFloquet(Omega_nuclear=Omega_nuclear, M=M, V=V, Er=Er, epsilon0=epsilon0, gamma=gamma, kT=kT, mu=mu, ultrafast_pulse=ultrafast_pulse, envelope_pulse=envelope_pulse, NF=NF)
    else:
        raise KeyError(f"Invalid pulse type: {pulse_type}. The valid pulse types are: {LandryPulseTypes.__members__.keys()}")
                
        
# %%
