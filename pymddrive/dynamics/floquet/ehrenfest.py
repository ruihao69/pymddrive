import numpy as np
from numpy.typing import ArrayLike

from pymddrive.models.nonadiabatic_hamiltonian import diabatic_to_adiabatic, adiabatic_to_diabatic
from pymddrive.dynamics.options import BasisRepresentation
from pymddrive.dynamics.floquet.math_utils import get_Op_from_OpF

from typing import Tuple


def get_rho_and_populations(
    time: float,
    rho_F: ArrayLike, 
    HF_diab: ArrayLike,
    evecs_F: ArrayLike,
    NF: int,
    Omega: float,
    F_basis: BasisRepresentation,    
    target_basis: BasisRepresentation
) -> Tuple[ArrayLike, ArrayLike]:
    if F_basis == BasisRepresentation.Diabatic and target_basis == BasisRepresentation.Diabatic:
        return _diab_from_diabrhoF(time, rho_F, NF, Omega)
    elif F_basis == BasisRepresentation.Diabatic and target_basis == BasisRepresentation.Adiabatic:
        return _adiab_from_diabrhoF(time, rho_F, HF_diab, NF, Omega)
    elif F_basis == BasisRepresentation.Adiabatic and target_basis == BasisRepresentation.Diabatic:
        return _diab_from_adiabrhoF(time, rho_F, evecs_F, NF, Omega)
    elif F_basis == BasisRepresentation.Adiabatic and target_basis == BasisRepresentation.Adiabatic:
        return _adiab_from_adiabrhoF(time, rho_F, evecs_F, HF_diab, NF, Omega)
    else:
        raise ValueError(f"Invalid basis representation: ({F_basis=}, {target_basis=})!")
        

# diab floquet rho to diab rho and diab populations

def _diab_from_diabrhoF(
    time: float,
    diab_rho_F: ArrayLike, 
    NF: int,
    Omega: float,
) -> Tuple[ArrayLike, ArrayLike]:
    diab_rho = get_Op_from_OpF(diab_rho_F, time, Omega, NF)
    return diab_rho, diab_rho.diagonal().real

# diab floquet rho to adiab rho and adiab populations
def _adiab_from_diabrhoF(
    time: float,
    diab_rho_F: ArrayLike,
    HF_diab: ArrayLike,
    NF: int,
    Omega: float, 
) -> ArrayLike:
    H = get_Op_from_OpF(HF_diab, time, Omega, NF)
    _, evecs = np.linalg.eigh(H)
    diab_rho = get_Op_from_OpF(diab_rho_F, time, Omega, NF)
    adiab_rho = diabatic_to_adiabatic(diab_rho, evecs)
    return adiab_rho, adiab_rho.diagonal().real

# adiab floquet rho to diab rho and diab populations
def _diab_from_adiabrhoF(
    time: float,
    adiab_rho_F: ArrayLike,
    evecs_F: ArrayLike,
    NF: int,
    Omega: float,
) -> Tuple[ArrayLike, ArrayLike]:
    diab_rho_F = adiabatic_to_diabatic(adiab_rho_F, evecs_F)
    diab_rho = get_Op_from_OpF(diab_rho_F, time, Omega, NF)
    return diab_rho, diab_rho.diagonal().real

# adiab floquet rho to adiab rho and adiab populations
def _adiab_from_adiabrhoF(
    time: float,
    adiab_rho_F: ArrayLike,
    evecs_F: ArrayLike,
    HF_diab: ArrayLike,
    NF: int,
    Omega: float,
) -> ArrayLike:
    diab_rho_F = adiabatic_to_diabatic(adiab_rho_F, evecs_F)
    diab_rho = get_Op_from_OpF(diab_rho_F, time, Omega, NF)
    diab_H = get_Op_from_OpF(HF_diab, time, Omega, NF)
    _, evecs = np.linalg.eigh(diab_H)
    adiab_rho = diabatic_to_adiabatic(diab_rho, evecs)
    return adiab_rho, adiab_rho.diagonal().real
    
    