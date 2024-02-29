# %% The package
import numpy as np

from numbers import Complex
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
        cache_length: int = 40
    ):
        super().__init__(Omega=Omega, cache_length=cache_length)

        self.A = A
        self.t0 = t0
        self.tau = tau
        self.phi = phi

    def __repr__(self) -> str:
        return f"Morlet(A={self.A}, t0={self.t0}, tau={self.tau}, Omega={self.Omega}, phi={self.phi})"

    def __call__(self, time: float):
        return super().__call__(time)

    def _pulse_func(self, time: float):
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
        cache_length: int = 40
    ):
        super().__init__(Omega=Omega, cache_length=cache_length)
        self.A = A
        self.t0 = t0
        self.tau = tau
        self.phi = phi

    def __repr__(self) -> str:
        return f"MorletReal(A={self.A}, t0={self.t0}, tau={self.tau}, Omega={self.Omega}, phi={self.phi})"

    def __call__(self, time: float):
        return super().__call__(time)

    def _pulse_func(self, time: float):
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

class Gaussian(Pulse):
    def __init__(
        self, 
        A: float = 1,
        t0: float = 0,
        tau: float = 1, 
        cache_length: int = 30
    )->None:
        super().__init__(None, cache_length)
        self.A = A  
        self.tau = tau
        self.t0 = t0
        
    def __repr__(self) -> str:
        return f"Gaussian(A={self.A}, t0={self.t0}, tau={self.tau}, Omega={self.Omega})"
    
    def _pulse_func(self, time: float):
        return Gaussian.gaussian_pulse(self.A, self.t0, self.tau, time)
        
    @staticmethod
    def gaussian_pulse(
        A: Union[float, Complex],
        t0: float, 
        tau: float, 
        time: Union[float, ArrayLike]
    )->Union[float, ArrayLike]:
        return A * np.exp(-0.5 * (time - t0)**2 / tau**2) 
    
    @classmethod
    def from_quasi_floquet_morlet_real(cls, morlet: MorletReal)->"Gaussian":
        """Convert a <MorletReal> pulse to a <Gaussian> pulse by taking the quasi-Floquet transform.
        A <MorletReal> pulse is a real-valued Morlet wavelet, which is a real-valued Gaussian wavelet modulated by a cosine function.
        The cosine modulation can be removed by taking the quasi-Floquet transform, which results in a <Gaussian> pulse.
        Here, we use a convention that the returned <Gaussian> pulse modulates the upper right non-diagonal quadrant of the quasi-Floquet Hamiltonian.

        Args:
            morlet (MorletReal): A real-valued Morlet wavelet.

        Returns:
            Gaussian: A <Gaussian> pulse resides in the upper right non-diagonal quadrant of the quasi-Floquet Hamiltonian.
        """
        t0 = morlet.t0; Omega = morlet.Omega; phi = morlet.phi
        phase_factor = np.exp(-1.0j * (Omega * t0 - phi))
        gaussian_complex_A: Complex = morlet.A * phase_factor
        return cls(A=gaussian_complex_A, t0=t0, tau=morlet.tau)
    
    @classmethod
    def from_quasi_floquet_morlet(cls, morlet: Morlet)->"Gaussian":
        raise NotImplementedError("The method <from_quasi_floquet_morlet> is not implemented yet.")

# %% the temperary testting/debugging code
def _debug_test_from_quasi_floquet_morlet_real():
    pulse_morletreal = MorletReal(A=1, t0=4, tau=1, Omega=10, phi=0)
    pulse_gaussian = Gaussian.from_quasi_floquet_morlet_real(pulse_morletreal)
    
    import numpy as np
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')
    
    t = np.linspace(-0, 12, 3000)
    fig = plt.figure(figsize=(3, 2), dpi=200)
    ax = fig.add_subplot(111)
    dat_morlet = np.array([pulse_morletreal(tt) for tt in t])
    dat_gaussian = np.array([pulse_gaussian(tt) for tt in t])
    ax.plot(t, dat_morlet, lw=.5, label="MorletReal")
    ax.plot(t, np.abs(dat_gaussian), lw=.5, label=r"Abs Gaussian")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    ax.legend()
    plt.show()
    
    fig = plt.figure(figsize=(3, 2), dpi=200)   
    ax = fig.add_subplot(111)
    ax.plot(t, dat_gaussian.real, lw=.5, label=r"$\Re$ Gaussian")
    ax.plot(t, dat_gaussian.imag, lw=.5, label=r"$\Im$ Gaussian")
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    ax.legend()
    plt.show()
    
    
     
def _debug_test():
    import numpy as np
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')
    from pymddrive.pulses.pulses import MultiPulse
    # Test for single pulse
    p1 = MorletReal(A=1, t0=4, tau=1, Omega=10, phi=0)
    # p2 = MorletReal(A=1, t0=-10, tau=2, Omega=10, phi=0)
    # p3 = MorletReal(A=1, t0=+10, tau=2, Omega=10, phi=0)
    # p4 = MorletReal(A=1, t0=+20, tau=2, Omega=10, phi=0)

    # p = MultiPulse(p1, p2, p3, p4)
    p = p1

    t = np.linspace(-0, 12, 3000)
    sig = [p(tt) for tt in t]
    fig = plt.figure(figsize=(3, 2), dpi=200)

    ax = fig.add_subplot(111)
    ax.plot(t, sig, lw=.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pulse Signal")
    plt.show()

    # Test for multi pulse
    p2 = MorletReal(A=1, t0=8, tau=1, Omega=10, phi=0)
    p = MultiPulse(p1, p2)
    sig = [p(tt) for tt in t]
    fig = plt.figure(figsize=(3, 2), dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(t, sig, lw=.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Multi Pulse Signal")
    
    _debug_test_from_quasi_floquet_morlet_real()

# %% the __main__ test code
if __name__ == "__main__":
    _debug_test()

# %%
