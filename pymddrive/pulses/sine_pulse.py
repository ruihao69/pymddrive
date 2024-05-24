import attr
from attrs import define, field
import numpy as np

from pymddrive.my_types import AnyNumber, RealNumber
from pymddrive.pulses.pulse_base import PulseBase

@define
class SinePulse(PulseBase):
    A: AnyNumber = field(on_setattr=attr.setters.frozen)
    Omega: RealNumber = field(on_setattr=attr.setters.frozen)
    
    def _pulse_func(self, time: RealNumber) -> AnyNumber:
        self.Omega: RealNumber
        return self.A * np.sin(self.Omega * time)
    
    def _gradient_func(self, time: RealNumber) -> AnyNumber:
        return self.A * self.Omega * np.cos(self.Omega * time)