from pymddrive.my_types import AnyNumber, RealNumber
from pymddrive.pulses.pulse_base import PulseBase

class ZeroPulse(PulseBase):
        
    def _pulse_func(self, t: RealNumber) -> AnyNumber:
        return 0