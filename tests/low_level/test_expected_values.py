import unittest
import numpy as np

from pymddrive.low_level._low_level.states import get_expected_value
from tests.test_utils import get_random_O, get_random_vO, get_random_rho, get_random_psi

class TestExpectedValues(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = 2
        self.n_nuclear = 3
        self.psi = get_random_psi(self.dim)
        self.rho = get_random_rho(self.dim)
        self.H_real = get_random_O(self.dim, is_complex=False)
        self.H_complex = get_random_O(self.dim, is_complex=True)
        self.dHdR_real = get_random_vO(self.dim, self.n_nuclear, is_complex=False)
        self.dHdR_complex = get_random_vO(self.dim, self.n_nuclear, is_complex=True)

    def test_real_density_matrix(self):
        expected_value = get_expected_value(self.H_real, self.rho)
        expected_value_expected = np.trace(self.rho @ self.H_real)
        np.testing.assert_allclose(expected_value, expected_value_expected)

    def test_complex_density_matrix(self):
        expected_value = get_expected_value(self.H_complex, self.rho)
        expected_value_expected = np.trace(self.rho @ self.H_complex)
        np.testing.assert_allclose(expected_value, expected_value_expected)

    def test_real_wavefunction(self):
        expected_value = get_expected_value(self.H_real, self.psi)
        expected_value_expected = np.real(self.psi.conj().T @ self.H_real @ self.psi)
        np.testing.assert_allclose(expected_value, expected_value_expected)

    def test_complex_wavefunction(self):
        expected_value = get_expected_value(self.H_complex, self.psi)
        expected_value_expected = np.real(self.psi.conj().T @ self.H_complex @ self.psi)
        np.testing.assert_allclose(expected_value, expected_value_expected)

    def test_real_derivative_density_matrix(self):
        expected_value = get_expected_value(self.dHdR_real, self.rho)
        expected_value_expected = []
        for i in range(self.n_nuclear):
            expected_value_expected.append(np.trace(self.rho @ self.dHdR_real[:, :, i]))
        np.testing.assert_allclose(expected_value, expected_value_expected)

    def test_complex_derivative_density_matrix(self):
        expected_value = get_expected_value(self.dHdR_complex, self.rho)
        expected_value_expected = []
        for i in range(self.n_nuclear):
            expected_value_expected.append(np.trace(self.rho @ self.dHdR_complex[:, :, i]))
        np.testing.assert_allclose(expected_value, expected_value_expected)

    def test_real_derivative_wavefunction(self):
        expected_value = get_expected_value(self.dHdR_real, self.psi)
        expected_value_expected = []
        for i in range(self.n_nuclear):
            expected_value_expected.append(np.real(self.psi.conj().T @ self.dHdR_real[:, :, i] @ self.psi))
        np.testing.assert_allclose(expected_value, expected_value_expected)

    def test_complex_derivative_wavefunction(self):
        expected_value = get_expected_value(self.dHdR_complex, self.psi)
        expected_value_expected = []
        for i in range(self.n_nuclear):
            expected_value_expected.append(np.real(self.psi.conj().T @ self.dHdR_complex[:, :, i] @ self.psi))
        np.testing.assert_allclose(expected_value, expected_value_expected)

if __name__ == '__main__':
    unittest.main()

