# %%
import numpy as np
from pymddrive.my_types import GenericOperator

def get_sigma_z() -> GenericOperator:
    """Return the Pauli Z matrix for a given dimension."""
    return np.diag([1, -1])

def get_sigma_x() -> GenericOperator:
    """Return the Pauli X matrix for a given dimension."""
    return np.array([[0, 1], [1, 0]])

def get_sigma_y() -> GenericOperator:
    """Return the Pauli Y matrix for a given dimension."""
    return np.array([[0, -1j], [1j, 0]])