import numpy as np
from numpy.typing import ArrayLike


def get_block_representation(matrix: np.ndarray, m: int, n: int) -> np.ndarray:
    """Block representation of a square matrix.

    Args:
        matrix (np.ndarray): a square matrix of shape (m*n, m*n)
        m (int): the dimension of the block square matrix
        n (int): the dimension of the matrix elements in the block square matrix

    Returns:
        np.ndarray: a block square matrix of shape (m, m, n, n)
    """
    return matrix.reshape(m, n, m, n).swapaxes(1, 2)

def get_floquet_multipliers(
    t: float, # time
    Omega: float, # driving frequency
    NF: int, # floquet levels cutoff
) -> ArrayLike: # return floquet multipliers
    return np.exp(1.j * np.arange(-NF, NF + 1) * Omega * t)

def get_Op_from_OpF(
    op_f: ArrayLike, # operator in F-space
    t: float,        # time
    Omega: float,    # driving frequency
    NF: int,         # floquet levels cutoff
):
    dim: int = op_f.shape[0] // (2 * NF + 1)
    blocked_op_f = get_block_representation(op_f, 2 * NF + 1, dim)
    center_col_blocked_op_f = blocked_op_f[:, NF, :, :]
    return np.tensordot(
        center_col_blocked_op_f, 
        get_floquet_multipliers(t, Omega, NF),
        axes=(0, 0)
    )
