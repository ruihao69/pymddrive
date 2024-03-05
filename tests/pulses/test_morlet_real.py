import unittest
import numpy as np
from pymddrive.pulses.morlet_real import MorletReal

class TestMorletReal(unittest.TestCase):
    def test_pulse_func(self):
        pulse = MorletReal(A=1, t0=0, tau=1, Omega=1, phi=0)
        self.assertAlmostEqual(pulse._pulse_func(0), 1)
    
    def test_default_values(self):
        pulse = MorletReal()
        self.assertEqual(pulse.A, 1)
        self.assertEqual(pulse.t0, 0)
        self.assertEqual(pulse.tau, 1)
        self.assertEqual(pulse.Omega, 1)
        self.assertEqual(pulse.phi, 0)
        self.assertEqual(pulse._cache_length, 40)

if __name__ == '__main__':
    unittest.main()