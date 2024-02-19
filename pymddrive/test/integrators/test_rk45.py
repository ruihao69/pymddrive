import unittest
import numpy as np
from pymddrive.integrators.rungekutta import tsit5_tableau

class TestTsit5Tableau(unittest.TestCase):
    def test_coefficients(self):
        self.assertTrue(np.all(tsit5_tableau.a[0] == 0))
        self.assertEqual(tsit5_tableau.c[1], tsit5_tableau.a[1, 0])
        self.assertEqual(tsit5_tableau.c[2] - tsit5_tableau.a[2, 1], tsit5_tableau.a[2, 0])
        self.assertEqual(tsit5_tableau.c[3] - tsit5_tableau.a[3, 1] - tsit5_tableau.a[3, 2], tsit5_tableau.a[3, 0])
        self.assertEqual(tsit5_tableau.c[4] - tsit5_tableau.a[4, 1] - tsit5_tableau.a[4, 2] - tsit5_tableau.a[4, 3], tsit5_tableau.a[4, 0])
        self.assertEqual(tsit5_tableau.c[5] - tsit5_tableau.a[5, 1] - tsit5_tableau.a[5, 2] - tsit5_tableau.a[5, 3] - tsit5_tableau.a[5, 4], tsit5_tableau.a[5, 0])

if __name__ == '__main__':
    unittest.main()
