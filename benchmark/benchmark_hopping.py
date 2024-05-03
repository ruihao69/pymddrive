# %%
import numpy as np
from numba import njit
from numpy.typing import NDArray

# from _low_level.surface_hopping import fssh_surface_hopping
from pymddrive.my_types import ActiveSurface
from pymddrive.low_level._low_level.surface_hopping import fssh_surface_hopping
# from pymddrive.dynamics.tmp.fssh import hopping
from pymddrive.dynamics.misc_utils import HamiltonianRetureType

from typing import Optional, Tuple

# %%

def v_dot_d(v, d,):
    return np.tensordot(v, d, axes=(0, 0))

def momentum_rescale(
    from_: int,
    to_: int,
    evals: NDArray[np.float64],
    d: NDArray[np.complex128],
    P_current: NDArray[np.float64],
    mass: Optional[NDArray[np.float64]],
) -> Tuple[bool, NDArray[np.float64]]:
    # determine the rescaling direction
    d_component = d[:, to_, from_] # I use the convention (i, j, k) for the index
                                   # i: the index of the classical degree of freedom
                                   # j, k: the index of the electronic states
                                   
    normalized_direction = d_component 
    M_inv = 1.0 / mass 
    dE = evals[to_] - evals[from_]
    
    # solve the quadratic equation
    a = 0.5 * np.sum(M_inv * normalized_direction**2)
    b = np.vdot(P_current * M_inv, normalized_direction)
    c = dE
    b2_4ac = b**2 - 4 * a * c
    if b2_4ac < 0:
        return False, P_current
    elif b < 0:
        gamma: float = (b + np.sqrt(b2_4ac)) / (2 * a)
        P_rescaled = P_current - gamma * normalized_direction
        # dPE = np.sum((P_rescaled**2 - P_current**2) * M_inv) / 2
        # print(f"{dE=}, {dPE=}, {mass=}")
        return True, P_rescaled
    elif b >= 0:
        # print(f"the gamma is {gamma}")
        gamma: float = (b - np.sqrt(b2_4ac)) / (2 * a)
        P_rescaled = P_current - gamma * normalized_direction
        # dPE = np.sum((P_rescaled**2 - P_current**2) * M_inv) / 2
        # print(f"{dE=}, {dPE=}, {mass=}")
        return True, P_rescaled
    else:
        return False, P_current


def hopping(
    dt: float,
    rho: NDArray[np.complex128],
    hami_return: HamiltonianRetureType,
    P: NDArray[np.float64],
    mass: Optional[NDArray[np.float64]],
    active_surf: ActiveSurface,
) -> Tuple[bool, ActiveSurface, NDArray[np.float64]]:
    ############################
    # The surface hopping algorithm
    ############################
    
    # compute the hopping probabilities
    v = P / mass
    vdotd = v_dot_d(v, hami_return.d)
    prob_vec = np.zeros(rho.shape[0])
    prob_vec[active_surf[0]] = 1.0
    prob_vec = _evaluate_hopping_prob(dt, active_surf[0], rho, vdotd, prob_vec)
    
    # use random number to determine the hopping 
    to = _hopping(prob_vec)
    
    # if hopping happens, update the active surface, rescale the momentum
    if to == active_surf[0]:
        return False, active_surf, P
    else:
        evals, d = hami_return.evals, hami_return.d
        allowed_hopping, P_rescaled = momentum_rescale(active_surf[0], to, evals, d, P, mass)
        return allowed_hopping, np.array([to]), P_rescaled

@njit
def _hopping_prob(
    from_: int,
    to_: int, 
    dt: float,
    vdotd: NDArray[np.complex128],
    rho: NDArray[np.complex128],
) -> float:
    # prob: float = -2.0 * dt * np.real(rho[to_, from_] * vdotd[from_, to_] / rho[from_, from_])
    prob: float = 2.0 * dt * np.real(rho[to_, from_] * vdotd[from_, to_] / rho[from_, from_])
    # prob: float = 2.0 * dt * np.real(rho[to_, from_] * vdotd[to_, from_] / rho[from_, from_])
    return prob if prob > 0 else 0.0

@njit
def _evaluate_hopping_prob(
    dt: float, 
    from_: int,
    rho: NDArray[np.complex128],
    v_dot_d: NDArray[np.complex128],
    prob_vec: NDArray[np.float64],
) -> NDArray[np.float64]:
    # tr_rho: float = np.trace(rho).real
    for to_ in range(prob_vec.shape[0]): 
        if from_ == to_:
            continue
        # prob_vec[to_] = _hopping_prob(from_, to_, dt, v_dot_d, rho) / tr_rho
        prob_vec[to_] = _hopping_prob(from_, to_, dt, v_dot_d, rho)
        prob_vec[from_] -= prob_vec[to_]
    return prob_vec

@njit
def _hopping(
    prob_vec: NDArray[np.float64]
) -> int:
    # prepare the variables
    accumulative_prob: float = 0.0
    to: int = 0
    
    # generate a random number
    random_number = np.random.rand()
    while (to < prob_vec.shape[0]):
        accumulative_prob += prob_vec[to]
        if accumulative_prob > random_number:
            break
        to += 1
    return to

def get_random_H(dim, is_complex=False):
    H = np.random.rand(dim, dim)
    if is_complex:
        H = H + 1.j * np.random.rand(dim, dim)
    H = H + H.conjugate().T
    return H

def get_random_dHdR(dim, n_nuc, is_complex=False):
    return np.array([get_random_H(dim, is_complex=is_complex) for _ in range(n_nuc)]).transpose(1, 2, 0)

def get_random_rho(dim):
    rho = np.random.rand(dim, dim)
    rho = rho + 1.j * np.random.rand(dim, dim)
    rho = rho + rho.conjugate().T
    rho /= np.trace(rho)
    return rho

def get_random_psi(dim):
    psi = np.random.rand(dim)
    psi = psi + 1.j * np.random.rand(dim)
    psi /= np.linalg.norm(psi)
    return psi

def get_random_dc(dim, n_nuc, is_complex=False):
    def assert_dc(dc):
        dc_conj_T = dc.conjugate().transpose(1, 0, 2) 
        assert np.allclose(dc, -dc_conj_T), 'dc is not skew-Hermitian'

    H = get_random_H(dim, is_complex=is_complex)
    dHdR = get_random_dHdR(dim, n_nuc, is_complex=is_complex)
    evals, evecs = np.linalg.eigh(H)
    dc = np.zeros_like(dHdR)
    F = np.zeros((dim, n_nuc))
    for i in range(n_nuc):
        dc[:, :, i] = evecs.T.conj() @ dHdR[:, :, i] @ evecs

    for i in range(dim):
        F[i, :] = -dc[i, i, :].real
        dc[i, i, :] = 0
        for j in range(i+1, dim):
            dc[i, j, :] /= evals[j] - evals[i]
            dc[j, i, :] = -dc[i, j, :].conjugate()
    assert_dc(dc)
    return H, dHdR, dc, F, evals

def randomize_surface_hopping_inputs(n_nuc, n_elec, is_complex):
    H, dHdR, dc, F, evals = get_random_dc(n_elec, n_nuc, is_complex=is_complex) 
    rho = get_random_rho(n_elec)
    P_current = np.random.normal(0, 1, n_nuc)
    mass = 1
    dt = 1
    active_surface = np.random.randint(0, n_elec)
        
    v_dot_d = np.tensordot(P_current, dc, axes=[[0], [2]])
    # print(f"{v_dot_d.shape=}, {rho.shape=}")
    return dt, active_surface, P_current, rho, evals, v_dot_d, dc, mass
    
def randomize_surface_hopping_inpts_py(n_nuc, n_elec, is_complex):
    dt, active_surface, P_current, rho, evals, v_dot_d, dc, mass = randomize_surface_hopping_inputs(n_nuc, n_elec, is_complex)
    dHdR = np.zeros_like(dc)
    H = np.zeros((n_elec, n_elec))
    F = np.zeros((n_elec, n_nuc))
    dc = dc.transpose(2, 0, 1)
    hami_return: HamiltonianRetureType = HamiltonianRetureType(H=H, dHdR=dHdR, evals=evals, evecs=np.zeros_like(H), d=dc, F=F)
    return dt, rho, hami_return, P_current, mass, np.array([active_surface])
    
def benchmark(n_nuc=3, n_elec=2, is_complex=False, n_repeats=10000):
    python_input_tuples = randomize_surface_hopping_inpts_py(n_nuc, n_elec, is_complex) 
    cpp_input_tuples = randomize_surface_hopping_inputs(n_nuc, n_elec, is_complex)
    import time
    
    start = time.perf_counter()
    for _ in range(n_repeats):
        success, active_surface, P_new = hopping(*python_input_tuples)
    print(f"Python time: {time.perf_counter() - start}")
    
    start = time.perf_counter()
    for _ in range(n_repeats):
        success, active_surface, P_new = fssh_surface_hopping(*cpp_input_tuples)
    print(f"C++ time: {time.perf_counter() - start}")

# %%
if __name__ == "__main__":
    n_nuc = 1
    n_elec = 2
    is_complex = False
     
    n_test = 1000
    n_success = 0
    n_fail = 0
    for _ in range(n_test):
        success, active_surface, P_new = fssh_surface_hopping(*randomize_surface_hopping_inputs(n_nuc, n_elec, is_complex)) 
        # success, active_surface, P_new = fssh_surface_hopping(dt, active_surface, P_current, rho, evals, v_dot_d, dc, mass)
        n_success += success
        n_fail += not success
        # print(f"{success=}, {active_surface=}")
    # print(f"{success=}, {active_surface=}, {P_current=}, {P_new=}")
    print(f"Success rate: {n_success/n_test} c++")
   
    n_success = 0
    n_fail = 0 
    for _ in range(n_test):
        success, active_surface, P_new = hopping(*randomize_surface_hopping_inpts_py(n_nuc, n_elec, is_complex))
        n_success += success
        n_fail += not success
    print(f"Success rate: {n_success/n_test} python")
    
    benchmark(n_nuc, n_elec, is_complex, n_repeats=300000)
    
        
    
    # print(-0.2 * dt * (rho[0, 1] * v_dot_d[1, 0]).real / rho[0, 0].real)
    
    
    
# %%
