# %%
import unittest
import numpy as np
from pymddrive.low_level._low_level.floquet import get_HF_cos as get_HF_cos_cpp
from pymddrive.low_level._low_level.floquet import get_dHF_dR_cos as get_dHF_dR_cos_cpp
from pymddrive.models.floquet import get_HF_cos as get_HF_cos_python

class TestFloquet(unittest.TestCase):
    @staticmethod
    def get_random_H(dim, is_complex=False):
        H = np.random.rand(dim, dim)
        if is_complex:
            H = H + 1.j * np.random.rand(dim, dim)
        H = H + H.conjugate().T
        return H

    def test_get_HF_cos(self):
        # Test parameters
        dim = 2
        is_complex = False
        H0 = self.get_random_H(dim, is_complex=is_complex)
        V = self.get_random_H(dim, is_complex=is_complex)

        Omega = 0.1
        NF = 2

        # Get the Floquet Hamiltonian using the C++ implementation
        HF_cos_cpp = get_HF_cos_cpp(H0, V, Omega, NF)

        # Get the Floquet Hamiltonian using the pure Python implementation
        HF_cos_python = get_HF_cos_python(H0, V, Omega, NF, is_gradient=False)

        # Compare the results
        np.testing.assert_allclose(HF_cos_cpp, HF_cos_python.toarray())

    def test_get_dHF_dR_cos(self):
        # Test parameters
        dim = 2
        is_complex = False
        dH0dR = self.get_random_H(dim, is_complex=is_complex)[:, :, np.newaxis]
        dVdR = self.get_random_H(dim, is_complex=is_complex)[:, :, np.newaxis]
        NF = 2
        Omega = 0.1

        # Get the gradient of the Floquet Hamiltonian using the C++ implementation
        dHF_dR_cos_cpp = get_dHF_dR_cos_cpp(dH0dR, dVdR, NF)

        # Get the gradient of the Floquet Hamiltonian using the pure Python implementation
        dHF_dR_cos_python = get_HF_cos_python(dH0dR[:, :, 0], dVdR[:, :, 0], Omega, NF, is_gradient=True)

        # Compare the results
        np.testing.assert_allclose(dHF_dR_cos_cpp[:, :, 0], dHF_dR_cos_python.toarray())

if __name__ == '__main__':
    unittest.main()

# %%

