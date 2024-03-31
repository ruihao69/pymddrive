# %%
import numpy as np
from numpy.typing import NDArray
from numba import njit

from pymddrive.low_level._low_level.ehrenfest import ehrenfest_meanF_diabatic as get_meanF_diabatic_cpp
from pymddrive.low_level._low_level.ehrenfest import ehrenfest_meanF_adiabatic as get_meanF_adiabatic_cpp
from tests.test_utils import get_random_O, get_random_vO, get_random_rho, get_random_psi, compute_dc

import time


class BenchmarkEhrenfest:

    def __init__(self, dim: int, n_nuc: int, n_repeats: int) -> None:
        self.dim = 2
        self.n_nuclear = 3
        self.n_repeats = n_repeats

        self.rho = get_random_rho(self.dim)
        self.psi = get_random_psi(self.dim)

        self.H_real = get_random_O(self.dim, is_complex=False)
        self.H_complex = get_random_O(self.dim, is_complex=True)

        self.dHdR_real = get_random_vO(self.dim, self.n_nuclear, is_complex=False)
        self.dHdR_complex = get_random_vO(self.dim, self.n_nuclear, is_complex=True)

        self.dc_real, self.F_real, self.evals_real, _ = compute_dc(self.H_real, self.dHdR_real)
        self.dc_complex, self.F_complex, self.evals_complex, _ = compute_dc(self.H_complex, self.dHdR_complex)

        # run everything numba once to compile the functions
        get_meanF_adiabatic_py(self.F_real, self.evals_real, self.dc_real, self.rho)
        get_meanF_adiabatic_py(self.F_real, self.evals_real, self.dc_real, self.psi)
        get_meanF_adiabatic_py(self.F_complex, self.evals_complex, self.dc_complex, self.rho)
        get_meanF_adiabatic_py(self.F_complex, self.evals_complex, self.dc_complex, self.psi)

    def benchmark_one(self, cpp_function, python_function, label:str):
        start = time.perf_counter()
        for _ in range(self.n_repeats):
            python_function()
        end = time.perf_counter()
        python_time = end - start

        start = time.perf_counter()
        for _ in range(self.n_repeats):
            cpp_function()
        end = time.perf_counter()
        cpp_time = end - start
        n = len("=====================================")
        nlabel = len(label)
        nleft = (n - nlabel) // 2 - 1
        nright = n - nlabel - nleft - 2
        print(f"{'='*nleft} {label} {'='*nright}")
        print(f"Python (numpy) time: {python_time:.2e} s")
        print(f"C++ time: {cpp_time:.2e} s")
        print(f"Speedup: {python_time / cpp_time:.2f}")
        print("=====================================")
        print()

    def benchmark_real_diabatic_wavefunction(self):
        def cpp_function():
            return get_meanF_diabatic_cpp(self.dHdR_real, self.psi)

        def python_function():
            return get_meanF_diabatic_py(self.dHdR_real, self.psi)

        self.benchmark_one(cpp_function, python_function, label="real_diabatic_wavefunction")

    def benchmark_complex_diabatic_wavefunction(self):
        def cpp_function():
            return get_meanF_diabatic_cpp(self.dHdR_complex, self.psi)

        def python_function():
            return get_meanF_diabatic_py(self.dHdR_complex, self.psi)

        self.benchmark_one(cpp_function, python_function, label="complex_diabatic_wavefunction")

    def benchmark_real_diabatic_density_matrix(self):
        def cpp_function():
            return get_meanF_diabatic_cpp(self.dHdR_real, self.rho)

        def python_function():
            return get_meanF_diabatic_py(self.dHdR_real, self.rho)

        self.benchmark_one(cpp_function, python_function, label="real_diabatic_density_matrix")

    def benchmark_complex_diabatic_density_matrix(self):
        def cpp_function():
            return get_meanF_diabatic_cpp(self.dHdR_complex, self.rho)

        def python_function():
            return get_meanF_diabatic_py(self.dHdR_complex, self.rho)

        self.benchmark_one(cpp_function, python_function, label="complex_diabatic_density_matrix")

    def benchmark_real_adiabatic_wavefunction(self):
        def cpp_function():
            return get_meanF_adiabatic_cpp(self.F_real, self.evals_real, self.dc_real, self.psi)

        def python_function():
            return get_meanF_adiabatic_py(self.F_real, self.evals_real, self.dc_real, self.psi)

        self.benchmark_one(cpp_function, python_function, label="real_adiabatic_wavefunction")

    def benchmark_real_adiabatic_density_matrix(self):
        def cpp_function():
            return get_meanF_adiabatic_cpp(self.F_real, self.evals_real, self.dc_real, self.rho)

        def python_function():
            return get_meanF_adiabatic_py(self.F_real, self.evals_real, self.dc_real, self.rho)

        self.benchmark_one(cpp_function, python_function, label="real_adiabatic_density_matrix")

    def benchmark_complex_adiabatic_wavefunction(self):
        def cpp_function():
            return get_meanF_adiabatic_cpp(self.F_complex, self.evals_complex, self.dc_complex, self.psi)

        def python_function():
            return get_meanF_adiabatic_py(self.F_complex, self.evals_complex, self.dc_complex, self.psi)

        self.benchmark_one(cpp_function, python_function, label="complex_adiabatic_wavefunction")

    def benchmark_complex_adiabatic_density_matrix(self):
        def cpp_function():
            return get_meanF_adiabatic_cpp(self.F_complex, self.evals_complex, self.dc_complex, self.rho)

        def python_function():
            return get_meanF_adiabatic_py(self.F_complex, self.evals_complex, self.dc_complex, self.rho)

        self.benchmark_one(cpp_function, python_function, label="complex_adiabatic_density_matrix")

    @staticmethod
    def get_random_H(dim, is_complex=False):
        H = np.random.rand(dim, dim)
        if is_complex:
            H = H + 1.j * np.random.rand(dim, dim)
        H = H + H.conjugate().T
        return H

    @staticmethod
    def get_random_dHdR(dim, n_nuc, is_complex=False):
        return np.array([BenchmarkEhrenfest.get_random_H(dim, is_complex=is_complex) for _ in range(n_nuc)]).transpose(1, 2, 0)

    @staticmethod
    def get_random_rho(dim):
        rho = np.random.rand(dim, dim)
        rho = rho + 1.j * np.random.rand(dim, dim)
        rho = rho + rho.conjugate().T
        rho /= np.trace(rho)
        return rho

    @staticmethod
    def get_random_psi(dim):
        psi = np.random.rand(dim)
        psi = psi + 1.j * np.random.rand(dim)
        psi /= np.linalg.norm(psi)
        return psi

    @staticmethod
    def get_random_dc(dim, n_nuc, is_complex=False):
        def assert_dc(dc):
            dc_conj_T = dc.conjugate().transpose(1, 0, 2)
            assert np.allclose(dc, -dc_conj_T), 'dc is not skew-Hermitian'

        H = BenchmarkEhrenfest.get_random_H(dim, is_complex=is_complex)
        dHdR = BenchmarkEhrenfest.get_random_dHdR(dim, n_nuc, is_complex=is_complex)
        evals, evecs = np.linalg.eigh(H)
        dc = np.zeros_like(dHdR)
        F = np.zeros((dim, n_nuc))
        for i in range(n_nuc):
            dc[:, :, i] = evecs.T.conj() @ dHdR[:, :, i] @ evecs

        for i in range(dim):
            F[i, :] = -dc[i, i, :].real
            dc[i, i, :] = 0
            for j in range(i+1, dim):
                dc[i, j, :] /= evals[j] - evals[i]
                dc[j, i, :] = -dc[i, j, :].conjugate()
        assert_dc(dc)
        return H, dHdR, dc, F, evals

def get_meanF_diabatic_py(dHdR, rho_or_psi):
    # dHdR (dim, dim, n_nuc)
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
    b = BenchmarkEhrenfest(dim=60, n_nuc=5, n_repeats=100000)
    b.benchmark_real_diabatic_wavefunction()
    b.benchmark_complex_diabatic_wavefunction()
    b.benchmark_real_diabatic_density_matrix()
    b.benchmark_complex_diabatic_density_matrix()
    b.benchmark_real_adiabatic_wavefunction()
    b.benchmark_complex_adiabatic_wavefunction()
    b.benchmark_real_adiabatic_density_matrix()
    b.benchmark_complex_adiabatic_density_matrix()


# %%
