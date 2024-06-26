import attr
from attrs import define, field
import numpy as np

from pymddrive.my_types import AnyNumber, RealNumber
from pymddrive.pulses.pulse_base import PulseBase
from pymddrive.pulses.sine_pulse import SinePulse
from pymddrive.pulses.cosine_pulse import CosinePulse

@define
class UnitPulse(PulseBase):
    A: AnyNumber = field(default=float('nan'), on_setattr=attr.setters.frozen)
    
        
    def _pulse_func(self, t: RealNumber) -> AnyNumber:
        return self.A
    
    def _gradient_func(self, t: RealNumber) -> AnyNumber:
        return 0.0
    
    def cannonical_amplitude(self, t: float) -> AnyNumber:
        raise NotImplementedError(f"Envelope pulse 'UnitPulse' does not support the method <cannonical_amplitude>.")
    
    @classmethod
    def from_cosine_pulse(cls, cosine_pulse: "CosinePulse") -> "UnitPulse":
        return cls(A=cosine_pulse.A)
    
    @classmethod
    def from_sine_pulse(cls, sine_pulse: "SinePulse") -> "UnitPulse":
        return cls(A=sine_pulse.A)
    