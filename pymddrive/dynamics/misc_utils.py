# %%
import numpy as np
from numpy.typing import ArrayLike  
import scipy.sparse as sp

from pymddrive.low_level.states import State
from pymddrive.integrators.state import get_state
from pymddrive.integrators.rungekutta import evaluate_initial_dt
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, evaluate_hamiltonian, evaluate_nonadiabatic_couplings, nac_phase_following
from pymddrive.dynamics.options import BasisRepresentation

from typing import Tuple, Any
from numbers import Real
from collections import namedtuple


HamiltonianRetureType = namedtuple('HamiltonianRetureType', 'H, dHdR, evals, evecs, d, F')

def eval_nonadiabatic_hamiltonian(
    t: float, R: ArrayLike, hamiltonian: HamiltonianBase, 
    basis_rep: BasisRepresentation=BasisRepresentation.Adiabatic,
    eval_deriv_cp: bool=False,
) -> HamiltonianRetureType: 
    flag_reshape = False
    flag_evec_following = True if basis_rep == BasisRepresentation.Adiabatic else False
    # flag_evec_following = False
    if R.shape[0] == 1:
        H, dHdR, evals, evecs = evaluate_hamiltonian(t, R[0], hamiltonian, enable_evec_following=flag_evec_following)
        flag_reshape = True
    else:
        H, dHdR, evals, evecs = evaluate_hamiltonian(t, R, hamiltonian, enable_evec_following=flag_evec_following)
    H = H.toarray() if sp.issparse(H) else H
    dHdR = dHdR.toarray() if sp.issparse(dHdR) else dHdR
        
    if basis_rep == BasisRepresentation.Adiabatic or eval_deriv_cp:
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs,)
    else:
        d, F = None, None
        
    if flag_reshape:
        dHdR = dHdR[:, :, np.newaxis]
        d = d[np.newaxis, :, :] if d is not None else None
        F = F[:, np.newaxis] if F is not None else None
    if (d is not None) and (hamiltonian.last_deriv_couplings is not None):
        d = nac_phase_following(hamiltonian.last_deriv_couplings, d)
    # hamiltonian.update_last_deriv_couplings(d)
    # if has friction attribute, then update the friction
    # if hasattr(hamiltonian, 'frictional_force'):
    #     F_fric = hamiltonian.frictional_force(P)
    #     F += F_fric
    return HamiltonianRetureType(H=H, dHdR=dHdR, evals=evals, evecs=evecs, d=d, F=F)

def estimate_scatter_dt(
    deriv: callable, 
    r_bounds: tuple, 
    s0: State, 
    cache: Any,
    nsample: Real=30, 
    t_bounds: Tuple[Real]=None
) -> float:
    _, p0, rho0 = s0.get_variables()
    r_list = np.linspace(*r_bounds, nsample)
    if t_bounds is not None:
        t_list = np.random.uniform(*t_bounds, nsample)
    else:
        t_list = np.zeros(nsample)
    _dt = 99999999999
    mass = s0.get_mass()
    for i in range(nsample):
        s0 = get_state(mass, r_list[i], p0, rho0)
        _dt = min(_dt, evaluate_initial_dt(deriv, t_list[i], s0, order=4, atol=1e-8, rtol=1e-8, cache=cache))
    return _dt

def valid_real_positive_value(value: Real) -> bool:
    if (flag_pos := (value > 0)) and (flag_real := np.isreal(value)):
        return True, flag_pos, flag_real
    else:
        return False, flag_pos, flag_real
    
def assert_valid_real_positive_value(value: Real) -> None:
    flag, flag_pos, flag_real = valid_real_positive_value(value)
    if not flag:
        raise ValueError(f"The value {value} is not a valid real positive number. The flags are {flag_pos=}, {flag_real=}.")
# %%
