# %%
import attr
from attrs import define, field
import numpy as np

from pymddrive.pulses.pulse_base import PulseBase
from pymddrive.my_types import AnyNumber, RealNumber
from pymddrive.pulses.morlet import Morlet
from pymddrive.pulses.morlet_real import MorletReal

@define
class Gaussian(PulseBase):
    A: AnyNumber = field(default=float('nan'), on_setattr=attr.setters.frozen)
    t0: RealNumber = field(default=float('nan'), on_setattr=attr.setters.frozen)
    tau: RealNumber = field(default=float('nan'), on_setattr=attr.setters.frozen)

    def _pulse_func(self, time: RealNumber) -> AnyNumber:
        return Gaussian.gaussian_pulse(self.A, self.t0, self.tau, time)
        
    @staticmethod
    def gaussian_pulse(
        A: AnyNumber,
        t0: RealNumber,
        tau: RealNumber,
        time: RealNumber
    ) -> AnyNumber:
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
        # with phase factor
        t0 = morlet.t0; Omega = morlet.Omega; phi = morlet.phi
        # phase_factor = np.exp(-1.0j * (Omega * t0 - phi))
        phase_factor = np.exp(1.0j * (Omega * t0 - phi)) 
        gaussian_complex_A: complex = morlet.A * phase_factor
        return cls(A=gaussian_complex_A, t0=t0, tau=morlet.tau)
    
        # without phase factor
        # t0 = morlet.t0
        # return cls(A=morlet.A, t0=t0, tau=morlet.tau)
    
    @classmethod
    def from_quasi_floquet_morlet(cls, morlet: Morlet)->"Gaussian":
        raise NotImplementedError("The method <from_quasi_floquet_morlet> is not implemented yet.")
    
# %% the temperary testting/debugging code
# def _debug_test_from_quasi_floquet_morlet_real():
#     pulse_morletreal = MorletReal(A=1, t0=4, tau=1, Omega=10, phi=0)
#     pulse_gaussian = Gaussian.from_quasi_floquet_morlet_real(pulse_morletreal)
#     
#     import numpy as np
#     import matplotlib.pyplot as plt
#     
#     t = np.linspace(-0, 12, 3000)
#     fig = plt.figure(figsize=(3, 2), dpi=200)
#     ax = fig.add_subplot(111)
#     dat_morlet = np.array([pulse_morletreal(tt) for tt in t])
#     dat_gaussian = np.array([pulse_gaussian(tt) for tt in t])
#     ax.plot(t, dat_morlet, lw=.5, label="MorletReal")
#     ax.plot(t, np.abs(dat_gaussian), lw=.5, label=r"Abs Gaussian")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Pulse Signal")
#     ax.legend()
#     plt.show()
#     
#     fig = plt.figure(figsize=(3, 2), dpi=200)   
#     ax = fig.add_subplot(111)
#     ax.plot(t, dat_gaussian.real, lw=.5, label=r"$\Re$ Gaussian")
#     ax.plot(t, dat_gaussian.imag, lw=.5, label=r"$\Im$ Gaussian")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Pulse Signal")
#     ax.legend()
#     plt.show()
#     
# # %% the __main__ test code
# if __name__ == "__main__":
#     _debug_test_from_quasi_floquet_morlet_real() 
# %%

# %%
