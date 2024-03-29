# %%
import numpy as np

from pymddrive.dynamics.math_utils import expected_value as get_expected_value_python
from pymddrive.low_level._low_level.states import get_expected_value

from tests.test_utils import get_random_O, get_random_vO, get_random_rho, get_random_psi, compute_dc

import time

class BenchmarkExpectationValue:
    def __init__(self, n_electronic: int=2, n_nuclear: int=3, n_repeats: int=10000) -> None:
        self.dim = n_electronic
        self.n_nuclear = n_nuclear
        self.n_repeats = n_repeats

        self.rho = get_random_rho(self.dim)
        self.psi = get_random_psi(self.dim)

        self.H_real = get_random_O(self.dim, is_complex=False)
        self.H_complex = get_random_O(self.dim, is_complex=True)

        self.dHdR_real = get_random_vO(self.dim, self.n_nuclear, is_complex=False)
        self.dHdR_complex = get_random_vO(self.dim, self.n_nuclear, is_complex=True)

        self.dc_real, self.F_real, self.evals_real = compute_dc(self.H_real, self.dHdR_real)
        self.dc_complex, self.F_complex, self.evals_complex = compute_dc(self.H_complex, self.dHdR_complex)

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

    def benchmark_real_density_matrix(self,):
        def cpp_function()->float:
            return get_expected_value(self.H_real, self.rho)

        def python_function()->float:
            # return np.trace(self.rho @ self.H_real).real
            return get_expected_value_python(O=self.H_real, qm=self.rho)
        self.benchmark_one(cpp_function, python_function, label="real_density_matrix")

    def benchmark_complex_density_matrix(self):
        def cpp_function()->float:
            return get_expected_value(self.H_complex, self.rho)

        def python_function()->float:
            return get_expected_value_python(O=self.H_complex, qm=self.rho)
        self.benchmark_one(cpp_function, python_function, label="complex_density_matrix")

    def benchmark_real_wavefunction(self):
        def cpp_function()->float:
            return get_expected_value(self.H_real, self.psi)

        def python_function()->float:
            # return np.real(self.psi.conj().T @ self.H_real @ self.psi)
            return get_expected_value_python(O=self.H_real, qm=self.psi)
        self.benchmark_one(cpp_function, python_function, label="real_wavefunction")

    def benchmark_complex_wavefunction(self):
        def cpp_function()->float:
            return get_expected_value(self.H_complex, self.psi)

        def python_function()->float:
            # return np.real(self.psi.conj().T @ self.H_complex @ self.psi)
            return get_expected_value_python(O=self.H_complex, qm=self.psi)
        self.benchmark_one(cpp_function, python_function, label="complex_wavefunction")

    def benchmark_real_derivative_density_matrix(self):
        dHdR_py = self.dHdR_real.transpose(2, 0, 1)
        def cpp_function()->float:
            return get_expected_value(self.dHdR_real, self.rho)

        def python_function()->float:
            # return [np.trace(self.rho @ self.dHdR_real[:, :, i]).real for i in range(self.n_nuclear)]
            return get_expected_value_python(O=dHdR_py, qm=self.rho)
        self.benchmark_one(cpp_function, python_function, label="real_derivative_density_matrix")

    def benchmark_complex_derivative_density_matrix(self):
        dHdR_py = self.dHdR_complex.transpose(2, 0, 1)
        def cpp_function()->float:
            return get_expected_value(self.dHdR_complex, self.rho)

        def python_function()->float:
            # return [np.trace(self.rho @ self.dHdR_complex[:, :, i]).real for i in range(self.n_nuclear)]
            return get_expected_value_python(O=dHdR_py, qm=self.rho)
        self.benchmark_one(cpp_function, python_function, label="complex_derivative_density_matrix")

    def benchmark_real_derivative_wavefunction(self):
        dHdR_py = self.dHdR_real.transpose(2, 0, 1)
        def cpp_function()->float:
            return get_expected_value(self.dHdR_real, self.psi)

        def python_function()->float:
            # return [np.real(self.psi.conj().T @ self.dHdR_real[:, :, i] @ self.psi) for i in range(self.n_nuclear)]
            return get_expected_value_python(O=dHdR_py, qm=self.psi)
        self.benchmark_one(cpp_function, python_function, label="real_derivative_wavefunction")

    def benchmark_complex_derivative_wavefunction(self):
        dHdR_py = self.dHdR_complex.transpose(2, 0, 1)
        def cpp_function()->float:
            return get_expected_value(self.dHdR_complex, self.psi)

        def python_function()->float:
            # return [np.real(self.psi.conj().T @ self.dHdR_complex[:, :, i] @ self.psi) for i in range(self.n_nuclear)]
            return get_expected_value_python(O=dHdR_py, qm=self.psi)
        self.benchmark_one(cpp_function, python_function, label="complex_derivative_wavefunction")

# %%
if __name__ == '__main__':
    nelectronic = 2
    nnuclear = 1
    nrepeats = 300000

    benchmark = BenchmarkExpectationValue(n_electronic=nelectronic, n_nuclear=nnuclear, n_repeats=nrepeats)
    benchmark.benchmark_real_density_matrix()
    benchmark.benchmark_complex_density_matrix()
    benchmark.benchmark_real_wavefunction()
    benchmark.benchmark_complex_wavefunction()
    benchmark.benchmark_real_derivative_density_matrix()
    benchmark.benchmark_complex_derivative_density_matrix()
    benchmark.benchmark_real_derivative_wavefunction()
    benchmark.benchmark_complex_derivative_wavefunction()

# %%
