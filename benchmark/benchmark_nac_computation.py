import numpy as np
from numba import njit

def diabatic_to_adiabatic(O, U):
    return np.dot(U.conjugate().T, np.dot(O, U))

def naive_nac_evaluation(evals, evecs, dHdR):
    d = np.zeros_like(dHdR)
    F = np.zeros((dHdR.shape[-2], dHdR.shape[-1]), dtype=dHdR.dtype)
    
    for kk in range(dHdR.shape[-1]):
        d[..., kk] = diabatic_to_adiabatic(dHdR[..., kk], evecs)
        
    for ii in range(dHdR.shape[0]):
        F[ii, :] = - d[ii, ii, :]
        d[ii, ii, :] = 0
        for jj in range(ii+1, dHdR.shape[0]):
            d[ii, jj, :] /= evals[jj] - evals[ii]
            d[jj, ii, :] = -d[ii, jj, :].conjugate()
    return d, F

@njit
def numba_nac_evaluation(evals, evecs, dHdR):
    d = np.zeros_like(dHdR)
    F = np.zeros((dHdR.shape[-2], dHdR.shape[-1]), dtype=dHdR.dtype)
    
    _op = np.zeros((dHdR.shape[0], dHdR.shape[0]), dtype=dHdR.dtype)
    for kk in range(dHdR.shape[-1]):
        _op = np.ascontiguousarray(dHdR[:, :, kk])
        d[:, :, kk] = np.dot(evecs.T.conjugate(), np.dot(_op, evecs))
    
    for ii in range(dHdR.shape[0]):
        F[ii, :] = - d[ii, ii, :]
        d[ii, ii, :] = 0
        for jj in range(ii+1, dHdR.shape[0]):
            d[ii, jj, :] /= evals[jj] - evals[ii]
            d[jj, ii, :] = -d[ii, jj, :].conjugate() 
    return d, F

def main():
    from tests.test_utils import get_random_vO, get_random_O
    import scipy.linalg as LA
    is_complex = False
    dim = 2
    nuc = 1
    
    dHdR = get_random_vO(dim, nuc, is_complex)
    H = get_random_O(dim, is_complex)
    evals, evecs = LA.eigh(H)
    
    d1, F1 = naive_nac_evaluation(evals, evecs, dHdR)
    d2, F2 = numba_nac_evaluation(evals, evecs, dHdR)
    
    np.testing.assert_allclose(d1, d2)
    np.testing.assert_allclose(F1, F2)
    
    import time
    N = 10000
    
    start = time.perf_counter() 
    for _ in range(N):
        d1, F1 = naive_nac_evaluation(evals, evecs, dHdR) 
    time_python_numpy = time.perf_counter() - start
    
    
    start = time.perf_counter() 
    for _ in range(N):
        d2, F2 = numba_nac_evaluation(evals, evecs, dHdR) 
    time_numba = time.perf_counter() - start
    
    print(f"{time_python_numpy=}")
    print(f"{time_numba=}")
    
if __name__ == "__main__":
    main()
    
    
    
        
        
