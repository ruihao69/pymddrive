import numpy as np
from typing import Tuple

def get_floquet_space_dim(dim: int, NF: int) -> int:
    return (2 * NF + 1) * dim

def get_floquet_index(i: int, j: int, m: int, n: int, dim: int, NF: int) -> Tuple[int, int]:
    return (m + NF) * dim + i, (n + NF) * dim + j