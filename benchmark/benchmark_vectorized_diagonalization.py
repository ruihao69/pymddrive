# %%
import numpy as np
from numba import njit

from pymddrive.my_types import RealVector, GenericOperator, GenericVectorOperator, GenericDiagonalVectorOperator
from tests.test_utils import get_random_vO

import time
from typing import Tuple

def vectorized_diagonalization_tr(
    Hv: GenericVectorOperator,
) -> Tuple[RealVector, GenericOperator]:
    evals_tmp, evecs_tmp = np.linalg.eigh(Hv.transpose(2, 0, 1))
    return evals_tmp.T, evecs_tmp.transpose(1, 2, 0)

@njit
def vectorized_diagonalization_numba(
    Hv: GenericVectorOperator,
):
    dim: int = Hv.shape[0]
    reps: int = Hv.shape[-1]

    evals = np.zeros((dim, reps), dtype=np.float64)
    evecs = np.zeros((dim, dim, reps), dtype=Hv.dtype)

    _op = np.zeros((dim, dim), dtype=Hv.dtype)

    for kk in range(reps):
        _op = np.ascontiguousarray(Hv[:, :, kk])
        evals[:, kk], evecs[:, :, kk] = np.linalg.eigh(_op)

    return evals, evecs

def benchmark(dim_e: int, dim_v: int, num_reps: int):
    Hv = get_random_vO(dim_e, dim_v)
    start = time.perf_counter()
    for _ in range(num_reps):
        evals_list, evecs_list = vectorized_diagonalization_tr(Hv)
    end = time.perf_counter()
    print(f"Time taken for transpose: {end - start}")

    _, _ = vectorized_diagonalization_numba(Hv)
    start = time.perf_counter()
    for _ in range(num_reps):
        evals_list, evecs_list = vectorized_diagonalization_numba(Hv)
    end = time.perf_counter()
    print(f"Time taken for numba: {end - start}")

# %%
if __name__ == "__main__":
    dim_e = 2
    dim_v = 100

    num_reps = 1000
    benchmark(dim_e, dim_v, num_reps)

# %%
