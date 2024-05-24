# %% The package
import attr
from attrs import define, field
import numpy as np

from pymddrive.my_types import AnyNumber, RealNumber
from pymddrive.pulses.pulse_base import PulseBase

@define
class MorletReal(PulseBase):
    A: AnyNumber = field(on_setattr=attr.setters.frozen)
    t0: RealNumber = field(on_setattr=attr.setters.frozen)
    tau: RealNumber = field(on_setattr=attr.setters.frozen)
    Omega: RealNumber = field(on_setattr=attr.setters.frozen)
    phi: RealNumber = field(on_setattr=attr.setters.frozen)
    

    def _pulse_func(self, time: RealNumber) -> AnyNumber:
        return MorletReal.real_morlet_pulse(self.A, self.t0, self.tau, self.Omega, self.phi, time)
    
    def _gradient_func(self, time: RealNumber) -> AnyNumber:
        return MorletReal.real_morlet_pulse_gradient(self.A, self.t0, self.tau, self.Omega, self.phi, time)

    @staticmethod
    def real_morlet_pulse(
        A: AnyNumber,
        t0: RealNumber,
        tau: RealNumber,
        Omega: RealNumber,
        phi: RealNumber,
        time: RealNumber
    ) -> RealNumber:
        # fully real-valued Morlet wavelet
        return A * np.cos(Omega * (time - t0) + phi) * np.exp(-0.5 * (time - t0)**2 / tau**2)
        
        # real-valued Morlet wavelet without the phase factor
        # return A * np.cos(Omega * time) * np.exp(-0.5 * (time - t0)**2 / tau**2)
        
    @staticmethod
    def real_morlet_pulse_gradient(
        A: AnyNumber,
        t0: RealNumber,
        tau: RealNumber,
        Omega: RealNumber,
        phi: RealNumber,
        time: RealNumber
    ) -> AnyNumber:
        return -A * Omega * np.sin(Omega * (time - t0) + phi) * np.exp(-0.5 * (time - t0)**2 / tau**2) - A * (time - t0) / tau**2 * np.cos(Omega * (time - t0) + phi) * np.exp(-0.5 * (time - t0)**2 / tau**2)
    
# %% The temperary testting/debugging code
def _test_debug_morlet_real():
    import matplotlib.pyplot as plt
    p1 = MorletReal(A=1, t0=4, tau=1, Omega=10, phi=0)
    t = np.linspace(-0, 12, 3000)
    sig = [p1(tt) for tt in t]
    fig = plt.figure(figsize=(3, 2), dpi=200)

    ax = fig.add_subplot(111)
    ax.plot(t, sig, lw=.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    plt.show()
    
# %% the __main__ testing/debugging code
if __name__ == "__main__":
    _test_debug_morlet_real()

# %%
