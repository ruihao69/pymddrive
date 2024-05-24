import unittest
import numpy as np
from pymddrive.pulses.sine_pulse import SinePulse

class TestSinePulse(unittest.TestCase):
    def test_pulse_func(self):
        pulse = SinePulse(A=2, Omega=2)
        self.assertAlmostEqual(pulse._pulse_func(0), 0)
        self.assertAlmostEqual(pulse._pulse_func(np.pi/4), 2)
        self.assertAlmostEqual(pulse._pulse_func(np.pi), 0)
        
    def test_gradient(self):
        A = 2
        Omega = 2
        pulse = SinePulse(A=A, Omega=Omega)
        T = 2*np.pi/Omega
        
        # gradient should be periodic
        self.assertAlmostEqual(pulse.gradient(0), pulse.gradient(T))
        # gradient should be zero at the peak
        self.assertAlmostEqual(pulse.gradient(T/4), 0.0)
        # gradient should be maximum at the zero crossing
        self.assertAlmostEqual(pulse.gradient(T/2), -A*Omega)

if __name__ == '__main__':
    unittest.main()