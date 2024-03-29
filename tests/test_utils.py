import numpy as np
import scipy.linalg as LA

def get_random_O(dim: int, is_complex=True) -> np.ndarray:
    op = np.random.rand(dim, dim)
    if is_complex:
        op = op + 1j*np.random.rand(dim, dim)
    op = op + op.T.conj()
    return op

def get_random_vO(dim: int, n_nuclei: int, is_complex=True) -> np.ndarray:
    return np.array([get_random_O(dim, is_complex) for _ in range(n_nuclei)]).transpose(1, 2, 0)

def get_random_rho(dim: int) -> np.ndarray:
    rho = np.zeros((dim, dim), dtype=np.complex128)
    rho[:] += np.random.rand(dim, dim) + 1j*np.random.rand(dim, dim)
    rho = rho + rho.T.conj()
    rho = rho / np.trace(rho)
    return rho

def get_random_psi(dim: int) -> np.ndarray:
    psi = np.zeros(dim, dtype=np.complex128)
    psi[:] += np.random.rand(dim) + 1j*np.random.rand(dim)
    psi = psi / np.linalg.norm(psi)
    return psi

def compute_dc(H: np.ndarray, dHdR: np.ndarray) -> tuple:
    evals, evecs = LA.eigh(H)

    nac = np.zeros_like(dHdR)
    # F = np.zeros_like(dHdR[0, ...])
    F = np.zeros((dHdR.shape[0], dHdR.shape[-1]), dtype=dHdR.dtype)

    for kk in range(dHdR.shape[2]):
        nac[:, :, kk] = evecs.T.conj() @ dHdR[:, :, kk] @ evecs

    for ii in range(H.shape[0]):
        F[ii, :] = -nac[ii, ii, :]
        nac[ii, ii, :] = 0
        for jj in range(ii+1, H.shape[0]):
            nac[ii, jj, :] /= evals[jj] - evals[ii]
            nac[jj, ii, :] = -nac[ii, jj, :].conjugate()

    return nac, F, evals, evecs

