# %%
import numpy as np
from numpy.typing import NDArray
from numba import njit

from pymddrive.low_level._low_level.ehrenfest import ehrenfest_meanF_diabatic as get_meanF_diabatic_cpp
from pymddrive.low_level._low_level.ehrenfest import ehrenfest_meanF_adiabatic as get_meanF_adiabatic_cpp

from tests.test_utils import get_random_O, get_random_vO, get_random_rho, get_random_psi, compute_dc

import unittest

class TestEhrenfest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = 2
        self.n_nuclear = 3

        self.rho = get_random_rho(self.dim)
        self.psi = get_random_psi(self.dim)

        self.H_real = get_random_O(self.dim, is_complex=False)
        self.H_complex = get_random_O(self.dim, is_complex=True)

        self.dHdR_real = get_random_vO(self.dim, self.n_nuclear, is_complex=False)
        self.dHdR_complex = get_random_vO(self.dim, self.n_nuclear, is_complex=True)

        self.dc_real, self.F_real, self.evals_real = compute_dc(self.H_real, self.dHdR_real)
        self.dc_complex, self.F_complex, self.evals_complex = compute_dc(self.H_complex, self.dHdR_complex)


        # run everything numba once to compile the functions
        get_meanF_adiabatic_py(self.F_real, self.evals_real, self.dc_real, self.rho)
        get_meanF_adiabatic_py(self.F_real, self.evals_real, self.dc_real, self.psi)
        get_meanF_adiabatic_py(self.F_complex, self.evals_complex, self.dc_complex, self.rho)
        get_meanF_adiabatic_py(self.F_complex, self.evals_complex, self.dc_complex, self.psi)

    def test_real_diabatic_wavefunction(self):
        meanF_cpp = get_meanF_diabatic_cpp(self.dHdR_real, self.psi)
        meanF_py = get_meanF_diabatic_py(self.dHdR_real, self.psi)
        np.testing.assert_allclose(meanF_cpp, meanF_py)

    def test_real_diabatic_density_matrix(self):
        meanF_cpp = get_meanF_diabatic_cpp(self.dHdR_real, self.rho)
        meanF_py = get_meanF_diabatic_py(self.dHdR_real, self.rho)
        np.testing.assert_allclose(meanF_cpp, meanF_py)

    def test_complex_diabatic_wavefunction(self):
        meanF_cpp = get_meanF_diabatic_cpp(self.dHdR_complex, self.psi)
        meanF_py = get_meanF_diabatic_py(self.dHdR_complex, self.psi)
        np.testing.assert_allclose(meanF_cpp, meanF_py)

    def test_complex_diabatic_density_matrix(self):
        meanF_cpp = get_meanF_diabatic_cpp(self.dHdR_complex, self.rho)
        meanF_py = get_meanF_diabatic_py(self.dHdR_complex, self.rho)
        np.testing.assert_allclose(meanF_cpp, meanF_py)

    def test_real_adiabatic_wavefunction(self):
        meanF_cpp = get_meanF_adiabatic_cpp(self.F_real, self.evals_real, self.dc_real, self.psi)
        meanF_py = get_meanF_adiabatic_py(self.F_real, self.evals_real, self.dc_real, self.psi)
        np.testing.assert_allclose(meanF_cpp, meanF_py)

    def test_real_adiabatic_density_matrix(self):
        meanF_cpp = get_meanF_adiabatic_cpp(self.F_real, self.evals_real, self.dc_real, self.rho)
        meanF_py = get_meanF_adiabatic_py(self.F_real, self.evals_real, self.dc_real, self.rho)
        np.testing.assert_allclose(meanF_cpp, meanF_py)

    def test_complex_adiabatic_wavefunction(self):
        meanF_cpp = get_meanF_adiabatic_cpp(self.F_complex, self.evals_complex, self.dc_complex, self.psi)
        meanF_py = get_meanF_adiabatic_py(self.F_complex, self.evals_complex, self.dc_complex, self.psi)
        np.testing.assert_allclose(meanF_cpp, meanF_py)

    def test_complex_adiabatic_density_matrix(self):
        meanF_cpp = get_meanF_adiabatic_cpp(self.F_complex, self.evals_complex, self.dc_complex, self.rho)
        meanF_py = get_meanF_adiabatic_py(self.F_complex, self.evals_complex, self.dc_complex, self.rho)
        np.testing.assert_allclose(meanF_cpp, meanF_py)

    def test_adiabatic_equivalent_to_diabatic_real(self):
        meanF_cpp_diabatic = get_meanF_diabatic_cpp(self.dHdR_real, self.psi)

        rho_diab = np.outer(self.psi, self.psi.conjugate())
        _, evecs = np.linalg.eigh(self.H_real)
        rho_adiab = evecs.T.conj() @ rho_diab @ evecs
        meanF_cpp_adiabatic = get_meanF_adiabatic_cpp(self.F_real, self.evals_real, self.dc_real, rho_adiab)
        np.testing.assert_allclose(meanF_cpp_diabatic, meanF_cpp_adiabatic)


    def test_adiabatic_equivalent_to_diabatic_complex(self):
        meanF_cpp_diabatic = get_meanF_diabatic_cpp(self.dHdR_complex, self.psi)

        rho_diab = np.outer(self.psi, self.psi.conjugate())
        _, evecs = np.linalg.eigh(self.H_complex)
        rho_adiab = evecs.T.conj() @ rho_diab @ evecs
        meanF_cpp_adiabatic = get_meanF_adiabatic_cpp(self.F_complex, self.evals_complex, self.dc_complex, rho_adiab)
        np.testing.assert_allclose(meanF_cpp_diabatic, meanF_cpp_adiabatic)


def get_meanF_diabatic_py(dHdR, rho_or_psi):
    if rho_or_psi.ndim == 1:
        return -np.einsum('jki,j,k->i', dHdR, rho_or_psi.conjugate(), rho_or_psi).real
    elif rho_or_psi.ndim == 2:
        return -np.einsum('jki,kj->i', dHdR, rho_or_psi).real
    else:
        raise ValueError('Invalid shape of rho_or_psi')


@njit
def get_meanF_adiabatic_wavefunction(
    F: NDArray[np.float64],
    evals: NDArray[np.float64],
    dc: NDArray,
    psi: NDArray[np.complex128]
):
    meanF = np.zeros(dc.shape[2], dtype=np.complex128)
    for i in range(dc.shape[2]):
        for j in range(dc.shape[0]):
            for k in range(dc.shape[1]):
                if j == k:
                    meanF[i] += np.real(F[j, i] * psi[j].conjugate() * psi[j])
                else:
                    meanF[i] += np.real(psi[j].conjugate() * dc[k, j, i] * psi[k] * (evals[k] - evals[j]))
    return np.real(meanF)

@njit
def get_meanF_adiabatic_density_matrix(
    F: NDArray[np.float64],
    evals: NDArray[np.float64],
    dc: NDArray,
    rho: NDArray[np.complex128]
):
    meanF = np.zeros(dc.shape[2], dtype=np.float64)
    for i in range(dc.shape[2]):
        for j in range(dc.shape[0]):
            meanF[i] += np.real(F[j, i] * rho[j, j])
            for k in range(j+1, dc.shape[1]):
                meanF[i] += 2*np.real(rho[j, k] * dc[k, j, i]) * (evals[k] - evals[j])
    return meanF

def get_meanF_adiabatic_py(F, evals, dc, rho_or_psi):
    if rho_or_psi.ndim == 1:
        return get_meanF_adiabatic_wavefunction(F, evals, dc, rho_or_psi)
    elif rho_or_psi.ndim == 2:
        return get_meanF_adiabatic_density_matrix(F, evals, dc, rho_or_psi)
    else:
        raise ValueError('Invalid shape of rho_or_psi')

# %%
if __name__ == '__main__':
    unittest.main()
