# %%
import numpy as np
from numpy.typing import ArrayLike  
import scipy.sparse as sp

from pymddrive.integrators.state import State
from pymddrive.integrators.rungekutta import evaluate_initial_dt
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase, evaluate_hamiltonian, evaluate_nonadiabatic_couplings
from pymddrive.dynamics.options import BasisRepresentation

from typing import Tuple
from numbers import Real
from collections import namedtuple


HamiltonianRetureType = namedtuple('HamiltonianRetureType', 'H, dHdR, evals, evecs, d, F')

def eval_nonadiabatic_hamiltonian(
    t: float, R: ArrayLike, hamiltonian: HamiltonianBase, 
    basis_rep: BasisRepresentation=BasisRepresentation.Adiabatic,
    eval_deriv_cp: bool=False,
) -> HamiltonianRetureType: 
    flag_reshape = False
    if R.shape[0] == 1:
        H, dHdR, evals, evecs = evaluate_hamiltonian(t, R[0], hamiltonian)
        flag_reshape = True
    else:
        H, dHdR, evals, evecs = evaluate_hamiltonian(t, R, hamiltonian)
    H = H.toarray() if sp.issparse(H) else H
    dHdR = dHdR.toarray() if sp.issparse(dHdR) else dHdR
        
    if basis_rep == BasisRepresentation.Adiabatic or eval_deriv_cp:
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
    else:
        d, F = None, None
        
    if flag_reshape:
        dHdR = dHdR[:, :, np.newaxis]
        d = d[np.newaxis, :, :] if d is not None else None
        F = F[:, np.newaxis] if F is not None else None
        
    return HamiltonianRetureType(H=H, dHdR=dHdR, evals=evals, evecs=evecs, d=d, F=F)

def estimate_scatter_dt(deriv: callable, r_bounds: tuple, s0: State, nsample: Real=30, t_bounds: Tuple[Real]=None) -> float:
    _, p0, rho0 = s0.get_variables()
    r_list = np.linspace(*r_bounds, nsample)
    if t_bounds is not None:
        t_list = np.random.uniform(*t_bounds, nsample)
    else:
        t_list = np.zeros(nsample)
    _dt = 99999999999
    for i in range(nsample):
        s0 = State.from_variables(R=r_list[i], P=p0, rho=rho0)
        _dt = min(_dt, evaluate_initial_dt(deriv, t_list[i], s0, order=4, atol=1e-8, rtol=1e-8,))
    return _dt
# %%
