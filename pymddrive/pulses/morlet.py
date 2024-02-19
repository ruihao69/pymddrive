# %% The package
import numpy as np

from numpy.typing import ArrayLike
from typing import Union

from pymddrive.pulses.pulses import Pulse

class Morlet(Pulse):
    def __init__(
        self,
        A: float = 1,
        t0: float = 0,
        tau: float = 1,
        Omega: float = 1,
        phi: float = 0,
    ):
        self.A = A
        self.t0 = t0
        self.tau = tau
        self.Omega = Omega  
        self.phi = phi
        
    def __repr__(self) -> str:
        return f"Morlet(A={self.A}, t0={self.t0}, tau={self.tau}, Omega={self.Omega}, phi={self.phi})"
        
    def __call__(
        self, 
        time: Union[float, ArrayLike]
    ):
        return Morlet.morlet_pulse(self.A, self.t0, self.tau, self.Omega, self.phi, time)
    
    @staticmethod
    def morlet_pulse(
        A: float,
        t0: float,
        tau: float,
        Omega: float,
        phi: float,
        time: Union[float, ArrayLike]
    ):
        return A * np.exp(-1j * (Omega * (time - t0) + phi)) * np.exp(-0.5 * (time - t0)**2 / tau**2)
    
class MorletReal(Pulse):
    def __init__(
        self,
        A: float = 1,
        t0: float = 0,
        tau: float = 1,
        Omega: float = 1,
        phi: float = 0,
    ):
        self.A = A
        self.t0 = t0
        self.tau = tau
        self.Omega = Omega  
        self.phi = phi
        
    def __repr__(self) -> str:  
        return f"MorletReal(A={self.A}, t0={self.t0}, tau={self.tau}, Omega={self.Omega}, phi={self.phi})"
        
    def __call__(
        self, 
        time: Union[float, ArrayLike]
    ):
        return MorletReal.real_morlet_pulse(self.A, self.t0, self.tau, self.Omega, self.phi, time)
    
    @staticmethod
    def real_morlet_pulse(
        A: float,
        t0: float,
        tau: float,
        Omega: float,
        phi: float,
        time: Union[float, ArrayLike]
    ):
        return A * np.cos(Omega * (time - t0) + phi) * np.exp(-0.5 * (time - t0)**2 / tau**2)
    
# %% the temperary test code
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from pymddrive.pulses.pulses import MultiPulse
    
    print("Hello.")
    p1 = MorletReal(A=1, t0=4, tau=1, Omega=10, phi=0)
    # p2 = MorletReal(A=1, t0=-10, tau=2, Omega=10, phi=0)
    # p3 = MorletReal(A=1, t0=+10, tau=2, Omega=10, phi=0)
    # p4 = MorletReal(A=1, t0=+20, tau=2, Omega=10, phi=0)
    
    # p = MultiPulse(p1, p2, p3, p4)
    p = p1
    
    t = np.linspace(-0, 10, 1000)
    sig = p(t)
    fig = plt.figure(figsize=(3, 2), dpi=200)
    
    ax = fig.add_subplot(111)
    ax.plot(t, sig, lw=.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")

# %%
