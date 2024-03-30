import numpy as np

def get_floquet_space_dim(dim: int, NF: int) -> int:
    return (2 * NF + 1) * dim