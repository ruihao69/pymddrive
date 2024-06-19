# %%
import numpy as np
from numba import njit

from pymddrive.my_types import RealVector, ComplexVector, ComplexOperator, GenericOperator, ComplexVectorOperator


@njit
def evaluate_Fmag(
    v_dot_d: ComplexOperator, # dot product of velocity and derivative coupling 
    d: ComplexVectorOperator, # complex-valued derivative couplings
    active_surface: int, # active surface index
) -> RealVector:
    """
        by symmetry, if the nuclear is 1D, this term should give 0
    """
    dim_elc: int = v_dot_d.shape[0]
    dim_nuc: int = d.shape[-1]
    F_mag_active: RealVector = np.zeros(dim_nuc, dtype=np.float64)
    j: int = active_surface
    for k in range(dim_elc):
        if k == j:
            continue
        F_mag_active[:] += 2.0 * np.imag(d[j, k, ...] * v_dot_d[k, j])
    return F_mag_active

@njit
def get_optimal_phase_eta(
    complex_direction: ComplexVector,
) -> float:
    d_re = np.ascontiguousarray(np.real(complex_direction))
    d_im = np.ascontiguousarray(np.imag(complex_direction))
    
    dot_val: float = np.dot(d_re, d_im)
    norm_diff: float = np.linalg.norm(d_re)**2 - np.linalg.norm(d_im)**2
    tan_2eta: float = -2.0 * dot_val / norm_diff
    
    eta0: float = 0.5 * np.arctan(tan_2eta)
    eta1: float = eta0 + np.pi / 2
    
    if np.cos(2*eta0) * norm_diff > 0:
        return eta0
    elif np.cos(2*eta1) * norm_diff > 0:
        return eta1
    else:
        raise ValueError("No optimal phase found.")

@njit
def get_rescale_direction(
    complex_direction: ComplexVector, 
) -> RealVector:
    eta = get_optimal_phase_eta(complex_direction)
    return np.ascontiguousarray(np.real(np.exp(1j * eta) * complex_direction))


def testing():
    NTESTS = 100
    from tests.test_utils import get_random_vO
    from pymddrive.dynamics.nonadiabatic_solvers.math_utils import compute_v_dot_d
    
    def test_one_dim():
        dim_nuc = 1
        dim_elc = 2
        active_surface = np.random.randint(dim_elc)
        P = np.random.normal(size=dim_nuc)
        mass = 1.0
        dc = get_random_vO(dim=dim_elc, n_nuclei=dim_nuc, is_complex=True)
        v_dot_d = compute_v_dot_d(v=P/mass, dc=dc)
        
        F_mag = evaluate_Fmag(v_dot_d=v_dot_d, d=dc, P=P, active_surface=active_surface)
        assert F_mag.shape == (dim_nuc, )
        assert np.allclose(F_mag, np.zeros(dim_nuc))
        
    def test_two_dim():
        dim_nuc = 2
        dim_elc = 2
        active_surface = np.random.randint(dim_elc)
        P = np.random.normal(size=dim_nuc)
        mass = 1.0
        dc = get_random_vO(dim=dim_elc, n_nuclei=dim_nuc, is_complex=True)
        v_dot_d = compute_v_dot_d(v=P/mass, dc=dc)
        
        F_mag = evaluate_Fmag(v_dot_d=v_dot_d, d=dc, P=P, active_surface=active_surface)
        print("For 2d case, F_mag is not generally 0, ", f"{F_mag=}")
    
    def test_two_dim_pure_imag_dc():
        dim_nuc = 2
        dim_elc = 2
        active_surface = np.random.randint(dim_elc)
        P = np.random.normal(size=dim_nuc)
        mass = 1.0
        dc = get_random_vO(dim=dim_elc, n_nuclei=dim_nuc, is_complex=True)
        dc = np.ascontiguousarray(np.imag(dc))
        v_dot_d = compute_v_dot_d(v=P/mass, dc=1j * dc)
        
        F_mag = evaluate_Fmag(v_dot_d=v_dot_d, d=1j * dc, P=P, active_surface=active_surface)
        # print("For 2d pure imaginary deriv couping, F_mag is 0, ", f"{F_mag=}")
        assert np.allclose(F_mag, np.zeros(dim_nuc))
        
    def test_get_optimal_phase():
        dim_nuc = 2
        dim_elc = 2
        active_surface = np.random.randint(dim_elc)
        while (target_surface := np.random.randint(dim_elc)) == active_surface:
            pass
        assert target_surface != active_surface
        dc = get_random_vO(dim=dim_elc, n_nuclei=dim_nuc, is_complex=True)
        complex_direction = dc[active_surface, target_surface, ...]
        _ = get_optimal_phase_eta(complex_direction)
        direction = get_rescale_direction(complex_direction)
        # print("For 2d case, ", f"{complex_direction=}", f"{direction=}")
        
        
        
    for _ in range(NTESTS):
        # --- test one dim
        test_one_dim()
        test_two_dim_pure_imag_dc()
        test_get_optimal_phase()
        
    test_two_dim()
    
# %%
if __name__ == "__main__":
    testing()
        
    
# %%