# %%
import numpy as np

# from _low_level.surface_hopping import fssh_surface_hopping
from pymddrive.low_level._low_level.surface_hopping import fssh_surface_hopping
from pymddrive.dynamics.fssh import hopping
from pymddrive.dynamics.misc_utils import HamiltonianRetureType

# %%

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
    from pymddrive.dynamics.fssh import ActiveSurface
    return dt, rho, hami_return, P_current, mass, ActiveSurface(active_surface)
    
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
