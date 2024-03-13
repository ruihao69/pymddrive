import numpy as np
from numpy.typing import ArrayLike
import scipy.linalg as LA

from pymddrive.dynamics.options import BasisRepresentation
from pymddrive.models.nonadiabatic_hamiltonian import diabatic_to_adiabatic, adiabatic_to_diabatic
from pymddrive.dynamics.floquet.math_utils import (
    get_Op_from_OpF
)

from typing import Tuple

def get_rho_and_populations(
    time: float,
    active_surface: int,
    HF_diab: ArrayLike,
    evecs_F: ArrayLike,
    NF: int,
    Omega: float,
    F_basis: BasisRepresentation=BasisRepresentation.Adiabatic,
    target_basis: BasisRepresentation=BasisRepresentation.Adiabatic
) -> Tuple[ArrayLike, ArrayLike]:
    if F_basis == BasisRepresentation.Diabatic:
        raise ValueError(f"Invalid basis representation: {F_basis=}! We don't usually do FSSH in diabatic basis!")
    elif F_basis == BasisRepresentation.Adiabatic and target_basis == BasisRepresentation.Diabatic:
        return _diab_from_adiab_active_surface(time, active_surface, evecs_F, NF, Omega, HF_diab.shape[0])
    elif F_basis == BasisRepresentation.Adiabatic and target_basis == BasisRepresentation.Adiabatic:
        return _adiab_from_adiab_active_surface(time, active_surface, evecs_F, HF_diab, NF, Omega, HF_diab.shape[0])
    else:
        raise ValueError(f"Invalid basis representation: ({F_basis=}, {target_basis=})!")
    
def _diab_from_adiab_active_surface(
    time: float,
    active_surface: int,
    evecs_F: ArrayLike,
    NF: int,
    Omega: float,
    dim: int,
) -> Tuple[ArrayLike, ArrayLike]:
    # adiab_populations = np.zeros(dim)
    # adiab_populations[active_surface] = 1.0
    adiab_psiF = np.zeros(dim, dtype=complex)
    adiab_psiF[active_surface] = 1.0
    diab_psiF = evecs_F.conjugate().T @ adiab_psiF
    # diab_rhoF = adiabatic_to_diabatic(np.diag(adiab_populations), evecs_F)
    diab_rhoF = np.outer(diab_psiF, diab_psiF.conjugate())
    diab_rho = get_Op_from_OpF(diab_rhoF, time, Omega, NF)
    return diab_rho, diab_rho.diagonal().real

def _adiab_from_adiab_active_surface(
    time: float,
    active_surface: int,
    evecs_F: ArrayLike,
    HF_diab: ArrayLike,
    NF: int,
    Omega: float,
    dim: int,
) -> Tuple[ArrayLike, ArrayLike]: 
    diab_rho, _ = _diab_from_adiab_active_surface(time, active_surface, evecs_F, NF, Omega, dim)
    H = get_Op_from_OpF(HF_diab, time, Omega, NF)
    _, evecs = LA.eigh(H)
    adiab_rho = diabatic_to_adiabatic(diab_rho, evecs)
    return adiab_rho, adiab_rho.diagonal().real