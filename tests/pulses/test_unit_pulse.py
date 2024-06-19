import unittest
from pymddrive.pulses.unit_pulse import UnitPulse

class TestUnitPulse(unittest.TestCase):
    def test_pulse_func(self):
        pulse = UnitPulse(A=5)
        t = 0.0
        expected_result = 5
        self.assertEqual(pulse._pulse_func(t), expected_result)
        
    def test_gradient(self):
        pulse = UnitPulse(A=5)
        t = 0.0
        expected_result = 0
        self.assertEqual(pulse.gradient(t), expected_result)

if __name__ == '__main__':
    unittest.main()