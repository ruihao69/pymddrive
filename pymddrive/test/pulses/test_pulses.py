import unittest
import numpy as np
from pymddrive.pulses.pulses import Pulse, get_carrier_frequency, SinePulse, CosinePulse, UnitPulse

class TestPulse(unittest.TestCase):
    def test_call(self):
        pulse = Pulse()
        result = pulse(0.5)
        self.assertEqual(result, 0.0)

    def test_set_Omega(self):
        pulse = Pulse()
        pulse.set_Omega(2)
        self.assertEqual(pulse.Omega, 2)

class TestGetCarrierFrequency(unittest.TestCase):
    def test_get_carrier_frequency(self):
        p = Pulse(Omega=5)
        frequency = get_carrier_frequency(p)
        self.assertEqual(frequency, 5)
        cp = CosinePulse(Omega=10)
        frequency = get_carrier_frequency(cp)
        self.assertEqual(frequency, 10)

class TestSinePulse(unittest.TestCase):
    def test_call(self):
        A = 1.5; Omega = 0.5; t = 0.5
        pulse = SinePulse(A, Omega)
        result = pulse(t)
        self.assertAlmostEqual(result, A*np.sin(Omega*t))

class TestCosinePulse(unittest.TestCase):
    def test_call(self):
        A = 1.5; Omega = 0.5; t = 0.5
        pulse = CosinePulse(A=A, Omega=Omega)
        result = pulse(t)   
        self.assertAlmostEqual(result, A*np.cos(Omega*t))

class TestUnitPulse(unittest.TestCase):
    def test_call(self):
        pulse = UnitPulse(A=0.21)
        result = pulse(0.5)
        self.assertEqual(result, 0.21)

if __name__ == '__main__':
    unittest.main()