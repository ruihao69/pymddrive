import unittest
import numpy as np
from pymddrive.pulses.morlet import Morlet

class TestMorletPulse(unittest.TestCase):
    def test_pulse_func(self):
        pulse = Morlet(A=2, Omega=2)
        self.assertAlmostEqual(pulse._pulse_func(0), 2)
        # self.assertAlmostEqual(pulse._pulse_func(np.pi/2), -2)
        # self.assertAlmostEqual(pulse._pulse_func(np.pi), 2)
    
    def test_invalid_omega(self):
        with self.assertRaises(ValueError):
            pulse = Morlet(A=1, Omega='invalid')
    
    def test_default_values(self):
        pulse = Morlet()
        self.assertEqual(pulse.A, 1)
        self.assertEqual(pulse.Omega, 1)
        self.assertEqual(pulse._cache_length, 40)

if __name__ == '__main__':
    unittest.main()