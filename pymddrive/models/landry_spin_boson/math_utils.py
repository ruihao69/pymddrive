# %%
import numpy as np
from numpy.typing import ArrayLike

def get_sigma_z() -> ArrayLike:
    """Return the Pauli Z matrix for a given dimension."""
    return np.diag([1, -1])

def get_sigma_x() -> ArrayLike:
    """Return the Pauli X matrix for a given dimension."""
    return np.array([[0, 1], [1, 0]])

def get_sigma_y() -> ArrayLike:
    """Return the Pauli Y matrix for a given dimension."""
    return np.array([[0, -1j], [1j, 0]])