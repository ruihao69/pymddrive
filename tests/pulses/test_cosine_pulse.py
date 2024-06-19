import unittest
import numpy as np
from pymddrive.pulses.cosine_pulse import CosinePulse

class TestCosinePulse(unittest.TestCase):
    def test_pulse_func(self):
        pulse = CosinePulse(A=2, Omega=2)
        self.assertEqual(pulse._pulse_func(0), 2)
        self.assertAlmostEqual(pulse._pulse_func(np.pi/2), -2)
        self.assertAlmostEqual(pulse._pulse_func(np.pi), 2)
        
    def test_gradient(self):
        A = 2
        Omega = 2
        pulse = CosinePulse(A=A, Omega=Omega)
        T = 2*np.pi/Omega
        
        # gradient should be periodic
        self.assertAlmostEqual(pulse.gradient(0), pulse.gradient(T))
        # gradient should be zero at the peak
        self.assertAlmostEqual(pulse.gradient(0), 0.0)
        # gradient should be maximum at the zero crossing
        self.assertAlmostEqual(pulse.gradient(T/4), -A*Omega)
    
    
    
if __name__ == '__main__':
    unittest.main()