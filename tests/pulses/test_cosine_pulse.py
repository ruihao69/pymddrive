import unittest
import numpy as np
from pymddrive.pulses.cosine_pulse import CosinePulse

class TestCosinePulse(unittest.TestCase):
    def test_pulse_func(self):
        pulse = CosinePulse(A=2, Omega=2)
        self.assertEqual(pulse._pulse_func(0), 2)
        self.assertAlmostEqual(pulse._pulse_func(np.pi/2), -2)
        self.assertAlmostEqual(pulse._pulse_func(np.pi), 2)
    
    def test_invalid_omega(self):
        with self.assertRaises(ValueError):
            pulse = CosinePulse(A=1, Omega='invalid')
    
    def test_default_values(self):
        pulse = CosinePulse()
        self.assertEqual(pulse.A, 1)
        self.assertEqual(pulse.Omega, 1)
        self.assertEqual(pulse._cache_length, 40)

if __name__ == '__main__':
    unittest.main()