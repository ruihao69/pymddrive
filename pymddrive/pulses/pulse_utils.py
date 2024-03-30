from pymddrive.pulses.pulse_base import PulseBase

import logging

logger = logging.getLogger(__name__)

def get_carrier_frequency(pulse: PulseBase) -> float:
    if pulse.Omega is None:
        # give a warning to the logger
        logger.warning(f"Tried to get the carrier frequency from the pulse type {pulse.__class__.__name__}, which doesn't have a carrier frequency. Return nan instead.")
        return float('nan')
    else:
        return pulse.Omega
    
        