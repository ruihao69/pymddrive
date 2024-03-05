# %% The package
import numpy as np

from numbers import Real, Complex
from typing import TypeAlias

from pymddrive.pulses.pulse_base import PulseBase

AnyNumber : TypeAlias = int | float | complex   
RealNumber : TypeAlias = int | float

class Morlet(PulseBase):
    def __init__(
        self,
        A: AnyNumber = 1,
        t0: RealNumber = 0,
        tau: RealNumber = 1,
        Omega: RealNumber = 1,
        phi: RealNumber = 0,
        cache_length: int = 40
    ):
        super().__init__(Omega=Omega, cache_length=cache_length)
        
        if not isinstance(self.Omega, Real):
            raise ValueError(f"For Morlet, the carrier frequency {self.Omega=} should be a real number, not {type(self.Omega)}.")

        self.A = A
        self.t0 = t0
        self.tau = tau
        self.phi = phi

    def __repr__(self) -> str:
        return f"Morlet(A={self.A}, t0={self.t0}, tau={self.tau}, Omega={self.Omega}, phi={self.phi})"

    def __call__(self, time: float):
        return super().__call__(time)

    def _pulse_func(self, time: float) -> Complex:
        return Morlet.morlet_pulse(self.A, self.t0, self.tau, self.Omega, self.phi, time)

    @staticmethod
    def morlet_pulse(
        A: AnyNumber,
        t0: RealNumber,
        tau: RealNumber,
        Omega: RealNumber,
        phi: RealNumber,
        time: RealNumber
    ) -> Complex:
        return A * np.exp(-1j * (Omega * (time - t0) + phi)) * np.exp(-0.5 * (time - t0)**2 / tau**2)


# %% the temperary testting/debugging code
def _debug_test():
    import numpy as np
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')
    
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
