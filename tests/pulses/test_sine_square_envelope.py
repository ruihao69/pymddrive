import unittest
import numpy as np
from pymddrive.pulses.sine_square_envelope import SineSquareEnvelope

class TestSineSquareEnvelope(unittest.TestCase):
    def test_pulse_func(self):
        Omega = 2
        A = 1.5
        N = 8
        pulse = SineSquareEnvelope(Omega=Omega, A=A, N=N)
        T = 2*np.pi/Omega*N
        
        self.assertAlmostEqual(pulse(0), pulse(T))
        
    def test_gradient(self):
        Omega = 2
        A = 1.5
        N = 8
        pulse = SineSquareEnvelope(Omega=Omega, A=A, N=N)
        T = 2*np.pi/Omega*N
        
        t_rand = np.random.rand()    
        grad_finite_diff = SineSquareEnvelope.finite_deference_gradient(pulse, t_rand)
        anlytical_grad = pulse.gradient(t_rand)
        
        self.assertAlmostEqual(grad_finite_diff, anlytical_grad, places=6)
        
    def test_from_quasi_floquet_sine_square_pulse(self):
        from pymddrive.pulses.sine_square_pulse import SineSquarePulse
        p = SineSquarePulse(A=1, N=8, Omega=10, phi=np.pi) 
        e = SineSquareEnvelope.from_quasi_floquet_sine_square_pulse(p)
        
        self.assertAlmostEqual(e.A, 1j)
        self.assertAlmostEqual(e.Omega, 10)
        self.assertAlmostEqual(e.N, 8)  

if __name__ == '__main__':
    unittest.main()