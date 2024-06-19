from pymddrive.pulses.pulse_base import PulseBase

def get_carrier_frequency(pulse: PulseBase) -> float:
    return pulse.Omega