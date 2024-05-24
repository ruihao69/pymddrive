import unittest
import numpy as np
from pymddrive.pulses.pulse_base import PulseBase
from pymddrive.pulses.morlet import Morlet

class TestMorletPulse(unittest.TestCase):
    def test_pulse_func(self):
        pulse = Morlet(A=2, Omega=2, t0=0, tau=1, phi=0)
        self.assertAlmostEqual(pulse._pulse_func(0), 2)
        
    def test_gradient(self):
        pulse = Morlet(A=1, t0=0, tau=1, Omega=1, phi=0)
        t_rand = np.random.rand()
        grad_finite_diff = PulseBase.finite_deference_gradient(pulse, t_rand)
        anlytical_grad = pulse.gradient(t_rand)
        # finite difference should be close to the analytical gradient
        self.assertAlmostEqual(grad_finite_diff, anlytical_grad, places=6)
    

if __name__ == '__main__':
    unittest.main()