import unittest
import numpy as np
from pymddrive.pulses.morlet import Morlet, MorletReal

class TestMorlet(unittest.TestCase):
    def test_call(self):
        A = 1.5; t0 = 0.5; tau = 0.2; Omega = 0.5; phi = 0.1; t = 0.5
        pulse = Morlet(A, t0, tau, Omega, phi)
        result = pulse(t)
        expected_result = A * np.exp(-1j * (Omega * (t - t0) + phi)) * np.exp(-0.5 * (t - t0)**2 / tau**2)
        self.assertAlmostEqual(result, expected_result)

class TestMorletReal(unittest.TestCase):
    def test_call(self):
        A = 1.5; t0 = 0.5; tau = 0.2; Omega = 0.5; phi = 0.1; t = 0.5
        pulse = MorletReal(A, t0, tau, Omega, phi)
        result = pulse(t)
        expected_result = A * np.cos(Omega * (t - t0) + phi) * np.exp(-0.5 * (t - t0)**2 / tau**2)
        self.assertAlmostEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()