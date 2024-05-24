# %% The package
import attr
from attrs import define, field
import numpy as np

from pymddrive.my_types import AnyNumber, RealNumber
from pymddrive.pulses.pulse_base import PulseBase
from pymddrive.pulses.sine_square_pulse import SineSquarePulse


@define
class SineSquareEnvelope(PulseBase):
    A: AnyNumber = field(on_setattr=attr.setters.frozen)
    Omega: RealNumber = field(on_setattr=attr.setters.frozen)
    N: int = field(on_setattr=attr.setters.frozen)
    

    def _pulse_func(self, time: RealNumber) -> AnyNumber:
        return SineSquareEnvelope.sine_square(self.A, self.Omega, self.N, time)
    
    def _gradient_func(self, time: RealNumber) -> AnyNumber:
        return SineSquareEnvelope.sine_square_gradient(self.A, self.Omega, self.N, time)

    @staticmethod
    def sine_square(
        A: AnyNumber,
        Omega: RealNumber,
        N: int,
        time: RealNumber
    ) -> AnyNumber:
        return A * np.square(np.sin(0.5*Omega/N*time))
    
    @staticmethod
    def sine_square_gradient(
        A: AnyNumber,
        Omega: RealNumber,
        N: int,
        time: RealNumber
    ) -> AnyNumber:
        return A * Omega * np.sin(0.5*Omega/N*time) * np.cos(0.5*Omega/N*time) / N
    
    @classmethod
    def from_quasi_floquet_sine_square_pulse(cls, sine_suqare_pulse: SineSquarePulse) -> "SineSquareEnvelope":
        phase_factor = np.exp(1.j * (sine_suqare_pulse.phi - 0.5 * np.pi))
        return cls(A=sine_suqare_pulse.A * phase_factor, Omega=sine_suqare_pulse.Omega, N=sine_suqare_pulse.N)


# %% the temperary testting/debugging code
def _debug_test():
    import numpy as np
    import matplotlib.pyplot as plt
    
    p = SineSquarePulse(A=1, N=8, Omega=10, phi=np.pi) 
    e = SineSquareEnvelope.from_quasi_floquet_sine_square_pulse(p)
    
    time = np.linspace(-10, 10, 3000)
    pulse = [p(tt) for tt in time]
    envelope = [e(tt) for tt in time]
    
    fig = plt.figure(figsize=(3, 2), dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(time, np.abs(pulse), lw=.5, label="SineSquare ABS")
    ax.plot(time, np.abs(envelope), lw=.5, label="Envelope ABS")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    plt.show()
    
    fig = plt.figure(figsize=(3, 2), dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(time, np.real(pulse), lw=.5, label="SineSquare Real")
    ax.plot(time, np.imag(pulse), lw=.5, label="SineSquare Imag")
    ax.plot(time, np.real(envelope), lw=.5, label="Envelope Real")
    ax.plot(time, np.imag(envelope), lw=.5, label="Envelope Imag")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    ax.legend()
    plt.show()
    
    

# %% the __main__ test code
if __name__ == "__main__":
    _debug_test()

# %%
