# %% The package
import attr
from attrs import define, field
import numpy as np

from pymddrive.my_types import AnyNumber, RealNumber
from pymddrive.pulses.pulse_base import PulseBase


@define
class Morlet(PulseBase):
    A: AnyNumber = field(on_setattr=attr.setters.frozen)
    t0: RealNumber = field(on_setattr=attr.setters.frozen)
    tau: RealNumber = field(on_setattr=attr.setters.frozen)
    Omega: RealNumber = field(on_setattr=attr.setters.frozen)
    phi: RealNumber = field(on_setattr=attr.setters.frozen)
    

    def _pulse_func(self, time: RealNumber) -> AnyNumber:
        return Morlet.morlet_pulse(self.A, self.t0, self.tau, self.Omega, self.phi, time)

    @staticmethod
    def morlet_pulse(
        A: AnyNumber,
        t0: RealNumber,
        tau: RealNumber,
        Omega: RealNumber,
        phi: RealNumber,
        time: RealNumber
    ) -> complex:
        return A * np.exp(-1j * (Omega * (time - t0) + phi)) * np.exp(-0.5 * (time - t0)**2 / tau**2)


# %% the temperary testting/debugging code
def _debug_test():
    import numpy as np
    import matplotlib.pyplot as plt
    
    p = Morlet(A=1, t0=4, tau=1, Omega=10, phi=0)
    
    t = np.linspace(-0, 12, 3000)
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
