import unittest
import numpy as np
from pymddrive.pulses.sine_pulse import SinePulse

class TestSinePulse(unittest.TestCase):
    def test_pulse_func(self):
        pulse = SinePulse(A=2, Omega=2)
        self.assertAlmostEqual(pulse._pulse_func(0), 0)
        self.assertAlmostEqual(pulse._pulse_func(np.pi/4), 2)
        self.assertAlmostEqual(pulse._pulse_func(np.pi), 0)
    
    def test_invalid_omega(self):
        with self.assertRaises(ValueError):
            pulse = SinePulse(A=1, Omega='invalid')
    
    def test_default_values(self):
        pulse = SinePulse()
        self.assertEqual(pulse.A, 1)
        self.assertEqual(pulse.Omega, 1)
        self.assertEqual(pulse._cache_length, 40)

if __name__ == '__main__':
    unittest.main()