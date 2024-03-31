# %%
import unittest
import numpy as np
from numpy.typing import ArrayLike
import scipy.sparse as sp
from pymddrive.low_level._low_level.floquet import get_HF_cos as get_HF_cos_cpp
from pymddrive.low_level._low_level.floquet import get_dHF_dR_cos as get_dHF_dR_cos_cpp
# from pymddrive.models.floquet import get_HF_cos as get_HF_cos_python

from typing import List




def _dim_to_dimF(dim: int, NF: int) -> int:
    return dim * (2 * NF + 1)

def _get_Floquet_offset(dim_sys: int, NF: int, Omega: float) -> List:
    return [np.identity(dim_sys) * ii * Omega for ii in range(-NF, NF+1)]


def get_HF_cos_python(
    H0: ArrayLike, # The time-independent part of the Hamiltonian,
    V: ArrayLike, # The time-dependent part of the Hamiltonian (times cosine function),
    Omega: float, # The frequency of the driving field,
    NF: int, # The number of floquet levels to consider,
    is_gradient: bool = False,
    to_csr: bool = True
) -> sp.bsr_matrix:
    """ Suppose the Hamiltonian is given by H(t) = H0 + V(t) * cos(Omega * t). """
    dim = H0.shape[0]
    dimF = _dim_to_dimF(dim, NF)
    dtype = np.complex128 if np.iscomplexobj(H0) or np.iscomplexobj(V) else np.float64  
    
    if NF == 0:
        return sp.bsr_matrix(H0, dtype=dtype)
    
    offsets = _get_Floquet_offset(dim, NF, Omega)
    offsets = np.zeros_like(offsets) if is_gradient else offsets
    V_upper = V
    V_lower = V.transpose().conj()
    # V_upper = V.transpose().conj()
    # V_lower = V 
    
    
    data_first_row = (H0 + offsets[0], V_upper)
    data_middle = ((V_lower, H0+offsets[ii+1], V_upper) for ii in range(2*NF-1))
    data_last_row = (V_lower, H0 + offsets[-1])
    
    data = np.concatenate((data_first_row, *data_middle, data_last_row))
    
    indptr = np.concatenate([(0, ), 2+3*np.arange(0, 2*NF, dtype=int), (6*NF+1, )])
    indices = np.concatenate([(0, 1), *(i+np.arange(0, 3) for i in range(2*NF-1)), (2*NF-1, 2*NF)])
    
    HF = sp.bsr_matrix((data, indices, indptr), shape=(dimF, dimF), dtype=dtype) 
    # print(f"{LA.ishermitian(HF.toarray())=}")
    return HF.tocsr() if to_csr else HF


class TestFloquet(unittest.TestCase):
    @staticmethod
    def get_random_H(dim, is_complex=False):
        H = np.random.rand(dim, dim)
        if is_complex:
            H = H + 1.j * np.random.rand(dim, dim)
        H = H + H.conjugate().T
        return H

    def test_get_HF_cos_real(self):
        # Test parameters
        dim = 2
        is_complex = False
        H0 = self.get_random_H(dim, is_complex=is_complex)
        V = self.get_random_H(dim, is_complex=is_complex)

        Omega = 0.1
        NF = 2

        # Get the Floquet Hamiltonian using the C++ implementation
        HF_cos_cpp = get_HF_cos_cpp(H0, V, Omega, NF)

        # Get the Floquet Hamiltonian using the pure Python implementation
        HF_cos_python = get_HF_cos_python(H0, V, Omega, NF, is_gradient=False)

        # Compare the results
        np.testing.assert_allclose(HF_cos_cpp, HF_cos_python.toarray())
        
    def test_get_HF_cos_complex(self):
        # Test parameters
        dim = 2
        is_complex = True
        H0 = self.get_random_H(dim, is_complex=is_complex)
        V = self.get_random_H(dim, is_complex=is_complex)

        Omega = 0.1
        NF = 2

        # Get the Floquet Hamiltonian using the C++ implementation
        HF_cos_cpp = get_HF_cos_cpp(H0, V, Omega, NF)

        # Get the Floquet Hamiltonian using the pure Python implementation
        HF_cos_python = get_HF_cos_python(H0, V, Omega, NF, is_gradient=False)

        # Compare the results
        np.testing.assert_allclose(HF_cos_cpp, HF_cos_python.toarray())

    def test_get_dHF_dR_cos_real(self):
        # Test parameters
        dim = 2
        is_complex = False
        dH0dR = self.get_random_H(dim, is_complex=is_complex)[:, :, np.newaxis]
        dVdR = self.get_random_H(dim, is_complex=is_complex)[:, :, np.newaxis]
        NF = 2
        Omega = 0.1

        # Get the gradient of the Floquet Hamiltonian using the C++ implementation
        dHF_dR_cos_cpp = get_dHF_dR_cos_cpp(dH0dR, dVdR, NF)

        # Get the gradient of the Floquet Hamiltonian using the pure Python implementation
        dHF_dR_cos_python = get_HF_cos_python(dH0dR[:, :, 0], dVdR[:, :, 0], Omega, NF, is_gradient=True)

        # Compare the results
        np.testing.assert_allclose(dHF_dR_cos_cpp[:, :, 0], dHF_dR_cos_python.toarray())
        
    def test_get_dHF_dR_cos_complex(self):
        # Test parameters
        dim = 2
        is_complex = True
        dH0dR = self.get_random_H(dim, is_complex=is_complex)[:, :, np.newaxis]
        dVdR = self.get_random_H(dim, is_complex=is_complex)[:, :, np.newaxis]
        NF = 2
        Omega = 0.1

        # Get the gradient of the Floquet Hamiltonian using the C++ implementation
        dHF_dR_cos_cpp = get_dHF_dR_cos_cpp(dH0dR, dVdR, NF)

        # Get the gradient of the Floquet Hamiltonian using the pure Python implementation
        dHF_dR_cos_python = get_HF_cos_python(dH0dR[:, :, 0], dVdR[:, :, 0], Omega, NF, is_gradient=True)

        # Compare the results
        np.testing.assert_allclose(dHF_dR_cos_cpp[:, :, 0], dHF_dR_cos_python.toarray())

if __name__ == '__main__':
    unittest.main()

# %%

