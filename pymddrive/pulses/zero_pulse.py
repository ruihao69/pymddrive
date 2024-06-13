from pymddrive.my_types import AnyNumber, RealNumber
from pymddrive.pulses.pulse_base import PulseBase

class ZeroPulse(PulseBase):
        
    def _pulse_func(self, t: RealNumber) -> AnyNumber:
        return 0
    
    def _gradient_func(self, t: RealNumber) -> AnyNumber:
        return 0
    
    def cannonical_amplitude(self, t: float) -> AnyNumber:
        raise NotImplementedError(f"Envelope pulse 'ZeroPulse' does not support the method <cannonical_amplitude>.")