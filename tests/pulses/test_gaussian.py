import unittest
import numpy as np
from pymddrive.pulses.pulse_base import PulseBase
from pymddrive.pulses.gaussian import Gaussian

class TestGaussian(unittest.TestCase):
    def test_gaussian_pulse(self):
        A = 1.0
        t0 = 0.0
        tau = 1.0
        time = 2.0
        expected_result = A * np.exp(-0.5 * (time - t0)**2 / tau**2)
        gaussian = Gaussian(A=A, t0=t0, tau=tau)
        result = gaussian._pulse_func(time)
        self.assertAlmostEqual(result, expected_result)

    def test_from_quasi_floquet_morlet_real(self):
        from pymddrive.pulses.morlet_real import MorletReal
        morlet = MorletReal(A=1, t0=4, tau=1, Omega=10, phi=0)
        expected_A = morlet.A * np.exp(1.0j * (morlet.Omega * morlet.t0 - morlet.phi))
        expected_t0 = morlet.t0
        expected_tau = morlet.tau
        gaussian = Gaussian.from_quasi_floquet_morlet_real(morlet)
        self.assertAlmostEqual(gaussian.A, expected_A)
        self.assertAlmostEqual(gaussian.t0, expected_t0)
        self.assertAlmostEqual(gaussian.tau, expected_tau)

    def test_from_quasi_floquet_morlet(self):
        from pymddrive.pulses.morlet import Morlet
        morlet = Morlet(A=1, t0=4, tau=1, Omega=10, phi=0)
        with self.assertRaises(NotImplementedError):
            Gaussian.from_quasi_floquet_morlet(morlet)
            
    def test_gaussian_pulse_gradient(self):
        A = 1.0
        t0 = 0.0
        tau = 1.0
        time = np.random.uniform(-3*tau, 3*tau)
        gaussian = Gaussian(A=A, t0=t0, tau=tau)
        expected_result = PulseBase.finite_deference_gradient(gaussian, time)
        result = gaussian.gradient(time)
        # result = gaussian._gradient_func(time)
        self.assertAlmostEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()