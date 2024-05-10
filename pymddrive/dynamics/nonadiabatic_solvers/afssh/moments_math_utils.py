# %%
import numpy as np
from numba import njit

from pymddrive.my_types import ComplexVectorOperator, RealVector, GenericOperator, GenericVector, ComplexOperator, ActiveSurface, GenericDiagonalVectorOperator, GenericVectorOperator

from typing import Union

def dot_delta_R(
    evals: RealVector, # eigenvalues, V
    delta_R: ComplexVectorOperator, # delta_R
    delta_P: ComplexVectorOperator, # delta_P
    mass: Union[float, RealVector], # mass
    v_dot_d: GenericOperator, # v_dot_d
    active_surface: ActiveSurface, # active surface
) -> ComplexVectorOperator:
    ii = active_surface[0]
    T_R = evaluate_T_R(evals, delta_R, delta_P, mass, v_dot_d)
    T_R_ii = T_R[ii, ii, :]
    return T_R - T_R_ii[None, None, :]

def dot_delta_P(
    evals: RealVector, # eigenvalues, V
    delta_P: ComplexVectorOperator, # delta_P
    delta_F: ComplexVectorOperator, # delta_F
    v_dot_d: GenericOperator, # v_dot_d
    rho: ComplexOperator, # rho
    active_surface: ActiveSurface, # active surface
) -> ComplexVectorOperator:
    ii = active_surface[0]
    T_P = evaluate_T_P(evals, delta_P, delta_F, v_dot_d, rho)
    T_P_ii = T_P[ii, ii, :]
    return T_P - T_P_ii[None, None, :]


def evaluate_T_R(
    evals: RealVector, # eigenvalues, V
    delta_R: ComplexVectorOperator, # delta_R
    delta_P: ComplexVectorOperator, # delta_P
    mass: Union[float, RealVector], # mass
    v_dot_d: GenericOperator, # v_dot_d
) -> ComplexVectorOperator:
    T_R = np.zeros(shape=delta_R.shape, dtype=np.complex128)

    T_R[:] += -1.j * vectorized_diagonal_commutator(evals, delta_R)
    T_R[:] += delta_P  / mass
    T_R[:] += -vectorized_commutator(v_dot_d, delta_R)
    return T_R

def evaluate_T_P(
    evals: RealVector, # eigenvalues, V
    delta_P: ComplexVectorOperator, # delta_P
    delta_F: ComplexVectorOperator, # delta_F
    v_dot_d: GenericOperator, # v_dot_d
    rho: ComplexOperator, # rho
) -> ComplexVectorOperator:
    T_P = np.zeros(shape=delta_P.shape, dtype=np.complex128)

    T_P[:] += -1.j * vectorized_diagonal_commutator(evals, delta_P)
    T_P[:] += 0.5 * vectorized_anti_commutator(rho, delta_F)
    T_P[:] += -vectorized_commutator(v_dot_d, delta_P)
    return T_P

@njit
def delta_P_rescale(
    P: RealVector,
    mass: Union[float, RealVector],
    delta_P: ComplexVectorOperator,
    evals: RealVector,
    rho: ComplexOperator,
    active_surface: ActiveSurface,
    dc: GenericVectorOperator
) -> ComplexVectorOperator:
    delta_P_new = np.copy(delta_P)
    n_elec = delta_P.shape[0]
    n_nucl = delta_P.shape[2]
    
    direction = np.zeros(n_nucl, dtype=np.float64)
    jj = active_surface[0]
    for kk in range(n_elec):
        direction[:] = dc[jj, kk, :]
        gamma = rescale_single_delta_P(P, direction, evals[kk] - evals[jj], mass)
        delta_P_new[kk, kk, :] = -gamma * direction * rho[kk, kk]
        for ll in range(kk+1, n_elec):
            delta_P_new[kk, ll, :] = 0.0 # We set the off-diagonal elements to zero
            delta_P_new[ll, kk, :] = 0.0 # We set the off-diagonal elements to zero
    return delta_P_new
        

@njit
def rescale_single_delta_P(
    P: RealVector,
    direction: RealVector,
    dE: float,
    mass: Union[float, RealVector],
) -> float:
    a: float = 0.5 * np.dot(direction, direction) / mass
    b: float = np.dot(P / mass, direction)
    c: float = dE
    
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return 0.0
    elif a == 0:
        return 0.0
    else:
        gamma = (b + np.sqrt(discriminant)) / (2 * a) if b < 0 else (b - np.sqrt(discriminant)) / (2 * a)
        return gamma
        
    
    
    

    

@njit
def vectorized_diagonal_commutator(
    diag_O: GenericVector, # diagonal operator O represented by a vector
    vec: ComplexVectorOperator, # vector of operators shaped as (n, n, m)
) -> ComplexVectorOperator:
    out = np.zeros_like(vec)
    dim_elec = vec.shape[0]
    dim_nucl = vec.shape[2]
    for kk in range(dim_nucl):
        for ii in range(dim_elec):
            for jj in range(dim_elec):
                out[ii, jj, kk] = diag_O[ii] * vec[ii, jj, kk] - vec[ii, jj, kk] * diag_O[jj]

    return out

@njit
def vectorized_commutator(
    O: GenericOperator, # operator O
    vec: ComplexVectorOperator, # vector of operators shaped as (n, n, m)
) -> ComplexVectorOperator:
    out = np.zeros(shape=vec.shape, dtype=np.complex128)
    dim_elec = vec.shape[0]
    dim_nucl = vec.shape[2]
    for kk in range(dim_nucl):
        for ii in range(dim_elec):
            for jj in range(dim_elec):
                for ll in range(dim_elec):
                    out[ii, jj, kk] += O[ii, ll] * vec[ll, jj, kk] - vec[ii, ll, kk] * O[ll, jj]

    return out

@njit
def vectorized_anti_commutator(
    O: GenericOperator, # operator O
    vec: ComplexVectorOperator, # vector of operators shaped as (n, n, m)
) -> ComplexVectorOperator:
    out = np.zeros(shape=vec.shape, dtype=np.complex128)
    dim_elec = vec.shape[0]
    dim_nucl = vec.shape[2]
    for kk in range(dim_nucl):
        for ii in range(dim_elec):
            for jj in range(dim_elec):
                for ll in range(dim_elec):
                    out[ii, jj, kk] += O[ii, ll] * vec[ll, jj, kk] + vec[ii, ll, kk] * O[ll, jj]
    return out

@njit
def evaluate_delta_vec_O(
    vec_O: GenericVectorOperator,
    vec: GenericVector
) -> GenericVectorOperator:
    delta_vec_O = np.copy(vec_O)
    dim_qm = vec_O.shape[0]
    dim_cl = vec_O.shape[2]
    for kk in range(dim_cl):
        for ii in range(dim_qm):
            delta_vec_O[ii, ii, kk] -= vec[kk]
    return delta_vec_O

@njit
def reconstruct_HF_force(
    F: GenericDiagonalVectorOperator,
    d: GenericVectorOperator,
    evals: RealVector,
) -> ComplexVectorOperator:
    hellmann_feynman = np.zeros(shape=d.shape, dtype=np.complex128)
    # fill the diagonal elements
    for ii in range(d.shape[0]):
        hellmann_feynman[ii, ii, :] = -F[ii, :]

    # fill the off-diagonal elements
    for ii in range(d.shape[0]):
        for jj in range(ii+1, d.shape[0]):
            hellmann_feynman[ii, jj, :] = d[ii, jj, :] * (evals[jj] - evals[ii])
            hellmann_feynman[jj, ii, :] = -d[ii, jj, :].conjugate()
    return hellmann_feynman

# %%
def test():
    def test_vectorized_commutator():
        O = np.random.rand(2, 2)
        vec = np.random.rand(2, 2, 100)

        expected = np.zeros(shape=vec.shape, dtype=np.complex128)
        for kk in range(100):
            expected[..., kk] = np.dot(O, vec[..., kk]) - np.dot(vec[..., kk], O)
        result = vectorized_commutator(O, vec)

        print(f"{np.allclose(result, expected)=}")

    def test_vectorized_diagonal_commutator():
        diag_O = np.random.rand(2)
        vec = np.random.rand(2, 2, 100)

        expected = np.zeros(shape=vec.shape, dtype=np.complex128)
        for kk in range(100):
            mat_diag_O = np.diagflat(diag_O)
            expected[..., kk] = np.dot(mat_diag_O, vec[..., kk]) - np.dot(vec[..., kk], mat_diag_O)
        result = vectorized_diagonal_commutator(diag_O, vec)

        print(f"{np.allclose(result, expected)=}")

    test_vectorized_commutator()
    test_vectorized_diagonal_commutator()

    def test_evaluate_dot_delta_R():
        from tests.test_utils import get_random_O, get_random_psi, get_random_vO
        from pymddrive.models.nonadiabatic_hamiltonian import evaluate_nonadiabatic_couplings
        n_elec = 2
        n_nucl = 5
        psi = get_random_psi(n_elec)
        rho = np.outer(psi, psi.conj())
        H = get_random_O(n_elec)
        dHdR = get_random_vO(n_elec, n_nucl)
        evals, evecs = np.linalg.eigh(H)
        d, F, F_hellmann_feynman = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)

        P = np.random.rand(n_nucl)
        mass = 1
        v = P / mass
        v_dot_d = np.tensordot(v, d, axes=(0, 2))

        delta_R = get_random_vO(n_elec, n_nucl)
        delta_P = get_random_vO(n_elec, n_nucl)
        F_HF = reconstruct_HF_force(F, d, evals)
        active_surface = 0
        delta_F = F_HF - F_HF[active_surface, active_surface, :]

        active_surface : ActiveSurface = np.array([0])
        dot_delta_R(evals, delta_R, delta_P, mass, v_dot_d, active_surface)
        dot_delta_P(evals, delta_P, delta_F, v_dot_d, rho, active_surface)

    test_evaluate_dot_delta_R()






if __name__ == "__main__":
    test()
# %%
