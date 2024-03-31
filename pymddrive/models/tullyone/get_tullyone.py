# %% the package
from pymddrive.my_types import RealNumber
from pymddrive.pulses import PulseBase as Pulse
from pymddrive.pulses import MorletReal, Gaussian, UnitPulse
from pymddrive.models.tullyone.tullyone import TullyOne
from pymddrive.models.tullyone.tullyone_td_type1 import TullyOneTD_type1
from pymddrive.models.tullyone.tullyone_td_type2 import TullyOneTD_type2
from pymddrive.models.tullyone.tullyone_floquet_type1 import TullyOneFloquet_type1
from pymddrive.models.tullyone.tullyone_floquet_type2 import TullyOneFloquet_type2
from pymddrive.models.tullyone.tullyone_pulse_types import TullyOnePulseTypes

from typing import Optional
from enum import Enum

class TD_Methods(Enum):
    BRUTE_FORCE = 1
    FLOQUET = 2

def get_tullyone(
    A: RealNumber = 0.01, 
    B: RealNumber = 1.6, 
    C: RealNumber = 0.005, 
    D: RealNumber = 1.0, # Tully parameters
    t0: Optional[RealNumber] = None,
    Omega: Optional[RealNumber] = None,
    tau: Optional[RealNumber] = None, # pulse parameters
    pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.NO_PULSE,
    NF: Optional[int] = None,
): 
    
    td_method = TD_Methods.BRUTE_FORCE if NF is None else TD_Methods.FLOQUET
        
    if pulse_type == TullyOnePulseTypes.NO_PULSE:
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOne(A=A, B=B, C=C, D=D)
        else:
            raise ValueError(f"You are trying to get a floquet model for a time independent Hamiltonian.")
    else:
        if (t0 is None) or (Omega is None) or (tau is None):
            raise ValueError(f"You need to provide the pulse parameters t0, Omega, and tau for Time-dependent problems.")

    if td_method == TD_Methods.FLOQUET and NF is None:
        raise ValueError(f"You need to provide the number of Floquet replicas NF for Floquet models.")

    orig_pulse: Pulse
    floq_pulse: Pulse
    if pulse_type == TullyOnePulseTypes.PULSE_TYPE1:
        orig_pulse = MorletReal(A=C, t0=t0, tau=tau, Omega=Omega, phi=0)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=0, D=0, pulse=orig_pulse)
        elif (td_method == TD_Methods.FLOQUET) and (NF is not None):
            floq_pulse = Gaussian.from_quasi_floquet_morlet_real(orig_pulse)
            return TullyOneFloquet_type1(A=A, B=B, C=0, D=0, ultrafast_pulse=orig_pulse, envelope_pulse=floq_pulse, NF=NF)
        else:
            raise ValueError(f"Invalid TD method: {td_method}")

    elif pulse_type == TullyOnePulseTypes.PULSE_TYPE2:
        orig_pulse = MorletReal(A=1.0, t0=t0, tau=tau, Omega=Omega, phi=0)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type2(A=A, B=B, C=C, D=D, pulse=orig_pulse)
        elif (td_method == TD_Methods.FLOQUET) and (NF is not None):
            floq_pulse = Gaussian.from_quasi_floquet_morlet_real(orig_pulse)
            return TullyOneFloquet_type2(A=A, B=B, C=C, D=D, ultrafast_pulse=orig_pulse, envelope_pulse=floq_pulse, NF=NF)
        else:
            raise ValueError(f"Invalid TD method: {td_method}")

    elif pulse_type == TullyOnePulseTypes.PULSE_TYPE3:
        orig_pulse = MorletReal(A=C/2, t0=t0, tau=tau, Omega=Omega, phi=0)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=C/2, D=D, pulse=orig_pulse)
        elif (td_method == TD_Methods.FLOQUET) and (NF is not None):
            floq_pulse = Gaussian.from_quasi_floquet_morlet_real(orig_pulse)
            return TullyOneFloquet_type1(A=A, B=B, C=C/2, D=D, ultrafast_pulse=orig_pulse, envelope_pulse=floq_pulse, NF=NF)
        else:
            raise ValueError(f"Invalid TD method: {td_method}")
    elif pulse_type == TullyOnePulseTypes.ZEROPULSE:
        orig_pulse = UnitPulse(A=0)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=C, D=D, pulse=orig_pulse)
        else:
            raise ValueError(f"For the UnitPulse, you can only use the BruteForce method. But you are trying to use {td_method}.")
    elif pulse_type == TullyOnePulseTypes.UNITPULSE:
        orig_pulse = UnitPulse(A=C)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=0, D=0, pulse=orig_pulse)
        else:
            raise ValueError(f"For the UnitPulse, you can only use the BruteForce method. But you are trying to use {td_method}.")
    else:
        raise ValueError(f"Invalid pulse type: {pulse_type}")

