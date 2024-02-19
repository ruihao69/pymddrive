import numpy as np
from numpy.typing import ArrayLike
from typing import Union

# Special arries
def zeros(t: Union[float, complex, ArrayLike]) -> Union[float, complex, ArrayLike]:
    if isinstance(t, float) or isinstance(t, complex):
        return 0
    elif isinstance(t, np.ndarray):
        return np.zeros(t.shape, dtype=t.dtype)
    else:
        raise TypeError(f"The input type is {type(t)}, which is not supported at this time.")
    
# Special matrices
def is_symmetric_matrix(H: ArrayLike):
    if not np.allclose(H, H.T):
        return False
    
    return True

def is_hermitian_matrix(H: ArrayLike):
    if not np.allclose(H, H.T.conj()):
        return False
    
    return True

def is_real_symmetric_matrix(H: ArrayLike):
    return np.all(np.isreal(H)) and is_symmetric_matrix(H)