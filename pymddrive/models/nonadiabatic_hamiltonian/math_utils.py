import numpy as np
import scipy.linalg as LA
import scipy.sparse as sp
from numpy.typing import ArrayLike

from typing import Tuple, Union

def matrix_col_dot(A: ArrayLike, B: ArrayLike) -> ArrayLike:
    """ Compute the dot product of two matrices along the columns
    """
    return np.einsum('ij,ij->j', A, B)

def _enforce_gauge_from_last(evecs: ArrayLike, evecs_last: ArrayLike) -> ArrayLike:
    """Remove the gauge ambiguity by enforcing the continuity of the eigenvectors.
    Also known as the state-following, or the state tracking algorithm.

    Args:
        evecs (ArrayLike): the current eigenvectors
        evecs_last (ArrayLike): the previous eigenvectors

    Returns:
        ArrayLike: the gauge-consistent eigenvectors
        
    Reference:
        `mudslide` by Shane Parker. see the following link for original implementation:
        <https://github.com/smparker/mudslide>
    """
    signs = np.sign(matrix_col_dot(evecs, evecs_last.conjugate()))
    evecs *= signs
    return evecs

def diagonalize_hamiltonian_history(hamiltonian: ArrayLike, evec_last: Union[None, ArrayLike]=None) -> Tuple[ArrayLike, ArrayLike]:
    evals, evecs = LA.eigh(hamiltonian.toarray()) if sp.isspmatrix(hamiltonian) else LA.eigh(hamiltonian)
    
    # if evec_last is not None:
    #     evecs = _enforce_gauge_from_last(evecs, evec_last)
                
    return evals, evecs

def diagonalize_2d_real_symmetric(H: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    a = H[0, 0] 
    b = H[1, 1]
    c = H[0, 1]
        
    lambda1 = 0.5 * (a + b - np.sqrt((a - b)**2 + 4 * c**2))
    lambda2 = 0.5 * (a + b + np.sqrt((a - b)**2 + 4 * c**2))
        
    evals = np.array([lambda1, lambda2])
    theta = np.arctan2(2 * c, b - a) / 2
    evecs = np.array(
        [[ np.cos(theta), np.sin(theta)], 
         [-np.sin(theta), np.cos(theta)]]
    )
    return evals, evecs

def diabatic_to_adiabatic(
    O: ArrayLike,
    U: ArrayLike, 
    out: Union[ArrayLike, None]=None
)-> ArrayLike:
    if out is not None:
        np.dot(O, U, out=out)
        np.dot(U.conj().T, out, out=out)
    else:
        return np.dot(U.conj().T, np.dot(O, U)) 
    
def adiaobatic_to_diabatic(
    O: ArrayLike,
    U: ArrayLike, 
    out: Union[ArrayLike, None]=None
) -> ArrayLike:
    if out is not None:
        np.dot(U, O, out=out)
        np.dot(out, U.conj().T, out=out)
    else:
        return np.dot(U, np.dot(O, U.conj().T))