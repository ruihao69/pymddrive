from pymddrive.pulses import PulseBase, Morlet, MorletReal, CosinePulse, SinePulse

from .floquet_types import FloquetType
from .floquetable_pulses import FloquetablePulses
from .valid_envolpe_functions import ValidEnvolpeFunctions

def is_floquetable_pulse(pulse: PulseBase) -> bool:
    try:
        FloquetablePulses(pulse.__class__.__name__)
    except ValueError:
        return False
    return True

def get_floquet_type(pulse: PulseBase) -> FloquetType:
    if (not is_floquetable_pulse(pulse)):
        raise ValueError(f"The pulse {pulse} is not a floquetable pulse.")
    
    if isinstance(pulse, MorletReal):
        return FloquetType.COSINE
    elif isinstance(pulse, Morlet):
        return FloquetType.EXPONENTIAL
    elif isinstance(pulse, CosinePulse):
        return FloquetType.COSINE
    elif isinstance(pulse, SinePulse):
        return FloquetType.SINE
    else:
        raise NotImplementedError(f"The (quasi-)Floquet methods for pulse type {pulse} is not implemented yet.")
    
def get_envelope_function_type(pulse: PulseBase) -> ValidEnvolpeFunctions:
    if isinstance(pulse, MorletReal):
        return ValidEnvolpeFunctions.GAUSSIAN
    elif isinstance(pulse, Morlet):
        return ValidEnvolpeFunctions.GAUSSIAN
    elif isinstance(pulse, CosinePulse):
        return ValidEnvolpeFunctions.UNIT
    elif isinstance(pulse, SinePulse):
        return ValidEnvolpeFunctions.UNIT
    else:
        raise NotImplementedError(f"The quasi-floquet model for pulse type {pulse} is not implemented yet.")
    