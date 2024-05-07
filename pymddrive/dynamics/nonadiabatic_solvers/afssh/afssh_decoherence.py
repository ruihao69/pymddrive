# %%

import numpy as np
from numba import njit

from pymddrive.my_types import RealDiagonalVectorOperator, ComplexOperator, RealVector, GenericDiagonalVectorOperator, ActiveSurface, GenericVectorOperator

from typing import Tuple 

@njit
def get_rho_collapse(
    rho: ComplexOperator, # density matrix 
    ll: int, # the index of the active surface
    nn: int, # the index of the collapsed index
) -> ComplexOperator:
    rho_collapse = rho.copy()
    dim = rho.shape[0]
    
    # collapse the density matrix for index nn
    for ii in range(dim):
        rho_collapse[ii, nn] = 0.0
        rho_collapse[nn, ii] = 0.0  
        
    # renormalize the density matrix
    rho_collapse[ll, ll] += rho[nn, nn]
    for kk in range(dim):
        if (kk == ll) or (kk == nn):
            continue    
        rho_collapse[ll, kk] *= np.sqrt(np.real((rho[ll, ll] + rho[nn, nn]) / rho[ll, ll]))
        rho_collapse[kk, ll] = np.conj(rho_collapse[ll, kk])      
    return rho_collapse

@njit   
def afssh_decoherence(
    rho: ComplexOperator, # density matrix  
    delta_F: GenericDiagonalVectorOperator, # aux force displacement
    delta_R: RealDiagonalVectorOperator, # aux position displacement
    delta_P: RealDiagonalVectorOperator, # aux momentum displacement
    active_surface: ActiveSurface, # the active surface
    d: GenericVectorOperator, # nonadiabatic couplings
    evals: RealVector, # eigenvalues
    dt: float, # time step  
) -> Tuple[ComplexOperator, RealDiagonalVectorOperator, RealDiagonalVectorOperator, bool]:
    tau_d_reciprocal, tau_r_reciprocal = evaluate_decoherence_rates(
        delta_F=delta_F, delta_R=delta_R, active_surface=active_surface, d=d, evals=evals
    )
    
    n_decoherence: int = 0

    dim = tau_d_reciprocal.shape[0] 
    for nn in range(dim):
        if nn == active_surface[0]:
            continue
        random_number = np.random.rand() # each state n  ̸= i we generate a random number ∈[0,1].
        if random_number < tau_d_reciprocal[nn] * dt:
            n_decoherence += 1
            rho[:] = get_rho_collapse(rho=rho, ll=active_surface[0], nn=nn) 
            delta_R[nn, :] = 0.0
            delta_P[nn, :] = 0.0
        elif random_number < tau_r_reciprocal[nn] * dt:
            delta_R[nn, :] = 0.0
            delta_P[nn, :] = 0.0
            
    return rho, delta_R, delta_P, (n_decoherence > 0)
        

@njit
def evaluate_decoherence_rates(
    delta_F: GenericDiagonalVectorOperator, # aux force displacement
    delta_R: RealDiagonalVectorOperator, # aux position displacement
    active_surface: ActiveSurface, # the active surface
    d: GenericVectorOperator, # nonadiabatic couplings
    evals: RealVector, # eigenvalues
) -> Tuple[RealVector, RealVector]:
    dim = evals.shape[0]
    nuclear_dim = d.shape[-1]
    lambd = active_surface[0]
    tau_d_reciprocal = np.zeros(dim, dtype=np.float64)
    tau_r_reciprocal = np.zeros(dim, dtype=np.float64)
    delta_F_nn = np.zeros(nuclear_dim, dtype=np.complex128)
    F_ln = np.zeros(nuclear_dim, dtype=np.complex128)
    delta_R_diff = np.zeros(nuclear_dim, dtype=np.complex128)
    for nn in range(dim):
        if nn == lambd:
            continue
        delta_F_nn[:] = delta_F[nn]
        F_ln[:] = d[lambd, nn, :] * (evals[nn] - evals[lambd])
        delta_R_diff[:] = delta_R[nn] - delta_R[lambd]
        tau_d_reciprocal[nn] += 0.5 * np.abs(np.dot(delta_F_nn, delta_R_diff))
        tau_r_reciprocal[nn] += tau_d_reciprocal[nn]
        tau_d_reciprocal[nn] -= 2.0 * np.abs(np.dot(F_ln, delta_R_diff))
        
    return tau_d_reciprocal, tau_r_reciprocal
        
# %%
def test():
    import scipy.linalg as LA
    from tests.test_utils import get_random_psi, get_random_O, get_random_vO
    
    # --- test get_rho_collapse 
    dim = 4
    psi = get_random_psi(dim)
    rho = np.round(np.outer(psi, psi.conj()), 4)
    active_surface = np.array([0])
    
    H = get_random_O(dim) 
    d = get_random_vO(dim, 1)
    print(f"{d.shape=}")
    evals, evecs = LA.eigh(H)
    print(f"{evals.dtype=}")
    print(f"{d.dtype=}")
    
    collapse_index = 1 
    rho_collapse = np.round(get_rho_collapse(rho=rho, ll=active_surface[0], nn=collapse_index), 4)
    
    # print(rho)
    # print(rho_collapse)
    
    # print(f"{LA.ishermitian(rho_collapse)=}")
    # print(f"{np.trace(rho_collapse)=}")
    
    # --- test afssh_decoherence 
    L = 10
    delta_R = np.array([np.random.uniform(-L, L) for _ in range(dim)])[..., np.newaxis]
    delta_P = np.array([np.random.uniform(-L, L) for _ in range(dim)])[..., np.newaxis]
    delta_F = np.array([np.random.uniform(-L, L) for _ in range(dim)])[..., np.newaxis]
    dt = 0.1
    print()
    print("=====================================================")
    print(f"before applying afssh_decoherence")
    print(f"{delta_R.T=}")
    print(f"{delta_P.T=}")
    print(f"{delta_F.T=}")
    print(f"{rho=}")
    
     
    rho, delta_R, delta_P, collapsed_flag = afssh_decoherence(
        rho=rho, delta_F=delta_F, delta_R=delta_R, delta_P=delta_P, active_surface=active_surface, d=d, evals=evals, dt=dt
    )
    
    print()
    print("=====================================================")
    print(f"after applying afssh_decoherence")
    print(f"{delta_R.T=}")
    print(f"{delta_P.T=}")
    print(f"{rho=}")
    
    
    
    
    
# %%
if __name__ == "__main__":
    test()

# %%
