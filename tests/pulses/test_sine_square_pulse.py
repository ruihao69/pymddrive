import unittest
import numpy as np
from pymddrive.pulses.pulse_base import PulseBase
from pymddrive.pulses.sine_square_pulse import SineSquarePulse

class TestSinePulse(unittest.TestCase):
    def test_pulse_func(self):
        Omega = 2
        A = 1.5
        N = 8
        phi = np.pi/2
        pulse = SineSquarePulse(Omega=Omega, A=A, N=N, phi=phi)
        T = 2*np.pi/Omega*N
        
        self.assertAlmostEqual(pulse(0), pulse(T))
        
    def test_gradient(self):
        Omega = 2
        A = 1.5
        N = 8
        phi = np.pi/2
        pulse = SineSquarePulse(Omega=Omega, A=A, N=N, phi=phi)
        T = 2*np.pi/Omega*N
        
        t_rand = np.random.rand()    
        grad_finite_diff = PulseBase.finite_deference_gradient(pulse, t_rand)
        anlytical_grad = pulse.gradient(t_rand)
        
        self.assertAlmostEqual(grad_finite_diff, anlytical_grad, places=6)
        
        

if __name__ == '__main__':
    unittest.main()