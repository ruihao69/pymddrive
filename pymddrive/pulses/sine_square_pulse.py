# %% The package
import attr
from attrs import define, field
import numpy as np

from pymddrive.my_types import AnyNumber, RealNumber
from pymddrive.pulses.pulse_base import PulseBase


@define
class SineSquarePulse(PulseBase):
    A: AnyNumber = field(on_setattr=attr.setters.frozen)
    Omega: RealNumber = field(on_setattr=attr.setters.frozen)
    N: int = field(on_setattr=attr.setters.frozen)
    phi: RealNumber = field(on_setattr=attr.setters.frozen)
    

    def _pulse_func(self, time: RealNumber) -> AnyNumber:
        return SineSquarePulse.sine_square_pulse(self.A, self.Omega, self.N, self.phi, time)

    @staticmethod
    def sine_square_pulse(
        A: AnyNumber,
        Omega: RealNumber,
        N: int,
        phi: RealNumber,
        time: RealNumber
    ) -> AnyNumber:
        return A * np.square(np.sin(0.5*Omega/N*time)) * np.sin(Omega * time + phi)


# %% the temperary testting/debugging code
def _debug_test():
    import numpy as np
    import matplotlib.pyplot as plt
    
    p = SineSquarePulse(A=1, N=8, Omega=10, phi=np.pi)
     
    t = np.linspace(-30, 30, 3000)
    sig = np.array([p(tt) for tt in t])
    
    fig = plt.figure(figsize=(3, 2), dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(t, np.abs(sig), lw=.5, label="Morlet ABS")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    plt.show()
    
    fig = plt.figure(figsize=(3, 2), dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(t, np.real(sig), lw=.5, label="Morlet Real")
    ax.plot(t, np.imag(sig), lw=.5, label="Morlet Imag")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    ax.legend()
    plt.show()

# %% the __main__ test code
if __name__ == "__main__":
    _debug_test()

# %%
