import numpy as np
import unittest
from pymddrive.integrators.state import State, StateType, zeros_like

class TestState(unittest.TestCase):
    def test_state_creation(self):
        # Test classical state creation
        r = 3
        p = 3.5
        state_cl = State.from_variables(R=r, P=p)
        self.assertEqual(state_cl.stype, StateType.CLASSICAL)
        self.assertEqual(state_cl.data["R"], r)
        self.assertEqual(state_cl.data["P"], p)
        
        # Test quantum state creation
        rho = np.array([[1, 0], [0, 0]])
        state_q = State.from_variables(rho=rho)
        self.assertEqual(state_q.stype, StateType.QUANTUM)
        np.testing.assert_array_equal(state_q.data["rho"], rho)
        
        # Test mixed quantum classical state creation
        state_mqc = State.from_variables(R=r, P=p, rho=rho)
        self.assertEqual(state_mqc.stype, StateType.MQC)
        self.assertEqual(state_mqc.data["R"], r)
        self.assertEqual(state_mqc.data["P"], p)
        np.testing.assert_array_equal(state_mqc.data["rho"], rho)
        
    def test_state_operations(self):
        # Test addition
        state1 = State.from_variables(R=1, P=2)
        state2 = State.from_variables(R=3, P=4)
        result = state1 + state2
        self.assertEqual(result.stype, StateType.CLASSICAL)
        self.assertEqual(result.data["R"], 4)
        self.assertEqual(result.data["P"], 6)
        
        # Test multiplication by scalar
        scalar = 2
        result = state1 * scalar
        self.assertEqual(result.stype, StateType.CLASSICAL)
        self.assertEqual(result.data["R"], 2)
        self.assertEqual(result.data["P"], 4)
        
        # Test negation
        result = -state1
        self.assertEqual(result.stype, StateType.CLASSICAL)
        self.assertEqual(result.data["R"], -1)
        self.assertEqual(result.data["P"], -2)
        
        # Test subtraction
        result = state1 - state2
        self.assertEqual(result.stype, StateType.CLASSICAL)
        self.assertEqual(result.data["R"], -2)
        self.assertEqual(result.data["P"], -2)
        
        # Test division
        scalar = 2
        result = state1 / scalar
        self.assertEqual(result.stype, StateType.CLASSICAL)
        self.assertEqual(result.data["R"], 0.5)
        self.assertEqual(result.data["P"], 1)
        
        # Test in-place multiplication
        state1 *= scalar
        self.assertEqual(state1.stype, StateType.CLASSICAL)
        self.assertEqual(state1.data["R"], 2)
        self.assertEqual(state1.data["P"], 4)
        
        # Test in-place addition
        state1 += state2
        self.assertEqual(state1.stype, StateType.CLASSICAL)
        self.assertEqual(state1.data["R"], 5)
        self.assertEqual(state1.data["P"], 8)
        
        # Test in-place subtraction
        state1 -= state2
        self.assertEqual(state1.stype, StateType.CLASSICAL)
        self.assertEqual(state1.data["R"], 2)
        self.assertEqual(state1.data["P"], 4)
        
        # Test zeros_like
        state = State.from_variables(R=1, P=2)
        result = zeros_like(state)
        self.assertEqual(result.stype, StateType.CLASSICAL)
        self.assertEqual(result.data["R"], 0)
        self.assertEqual(result.data["P"], 0)
        
if __name__ == '__main__':
    unittest.main()