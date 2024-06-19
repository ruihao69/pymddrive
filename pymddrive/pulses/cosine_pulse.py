import attr
from attrs import define, field
import numpy as np

from pymddrive.my_types import AnyNumber, RealNumber
from pymddrive.pulses.pulse_base import PulseBase


@define
class CosinePulse(PulseBase):
    A: AnyNumber = field(on_setattr=attr.setters.frozen)
    Omega: RealNumber = field(on_setattr=attr.setters.frozen)
    
    def _pulse_func(self, time: RealNumber) -> AnyNumber:
        return self.A * np.cos(self.Omega * time)
    
    def _gradient_func(self, time: RealNumber) -> AnyNumber:
        return -self.A * self.Omega * np.sin(self.Omega * time)
    
    def cannonical_amplitude(self, t: float) -> float:
        return self.A 
    
if __name__ == "__main__":
    A = 1.00
    Omega = 1.00
    pulse = CosinePulse(A=A, Omega=Omega)
    time = 1.00
    print(pulse(time))