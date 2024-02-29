# %%
import numpy as np
import scipy.linalg as LA
import scipy.sparse as sp

from numba import jit   

from abc import ABC, abstractmethod
from enum import Enum, unique

from typing import Tuple, Union
from numpy.typing import ArrayLike
from numbers import Real

from pymddrive.pulses.pulses import Pulse, get_carrier_frequency
from pymddrive.models.floquet import get_HF, FloquetType, _dim_to_dimF


class NonadiabaticHamiltonianBase(ABC):
    def __init__(
        self,
        dim: int,
    ) -> None:
        self.dim: int = dim
        self.evec_last: Union[ArrayLike, None] = None
        
    def __call__(self, t: float, r: Union[float, ArrayLike]) -> Tuple[ArrayLike, ArrayLike]:
        H = self.H(t, r)
        # print(self.evec_last)
        evals, evecs = diagonalize_hamiltonian_history(H, self.evec_last)
        # evals, evecs = diagonalize_hamiltonian(H, enforce_gauge=False)
        self.evec_last = evecs
        dHdR = self.dHdR(t, r)
        dHdR, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        return H, evals, evecs, dHdR, F   
     
    @abstractmethod
    def H(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        pass
    
    @abstractmethod
    def dHdR(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        pass
    
class TD_NonadiabaticHamiltonianBase(NonadiabaticHamiltonianBase):
    def __init__(
        self,
        dim: int,
        pulse: Pulse,
    ) -> None:
        """ Time-dependent nonadiabatic Hamiltonian. """
        """ The time dependence is defined by a 'Pulse' object. """
        """ The pulse consists of a carrier frequency <Omega> and an envelope <E(t)>. """
        super().__init__(dim)
        self.pulse = pulse
        
    def H(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        return self.H0(r) + self.H1(t, r)
    
    def dHdR(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        return self.dH0dR(r) + self.dH1dR(t, r)
    
    @abstractmethod 
    def H0(self, r: Union[float, ArrayLike]) -> ArrayLike:
        pass
    
    @abstractmethod 
    def H1(self, t: float, r: Union[float, ArrayLike], pulse: Pulse) -> ArrayLike:
        pass
    
    @abstractmethod
    def dH0dR(self, r: Union[float, ArrayLike]) -> ArrayLike:
        pass
    
    @abstractmethod 
    def dH1dR(self, t: float, r: Union[float, ArrayLike], pulse: Pulse) -> ArrayLike:
        pass

@unique
class FloquetablePulses(Enum):
    MORLET = "Morlet"
    MORLET_REAL = "MorletReal"
    COSINE = "CosinePulse"
    SINE = "SinePulse"
    EXPONENTIAL = "ExponentialPulse"

@unique 
class ValidQuasiFloqeuetPulses(Enum):
    GAUSSIAN = "Gaussian"
    UNIT = "UnitPulse"
    
def get_floquet_type_from_pulsetype(pulsetype: FloquetablePulses) -> FloquetType:
    if pulsetype == FloquetablePulses.MORLET_REAL:
        return FloquetType.COSINE
    elif pulsetype == FloquetablePulses.MORLET:
        return FloquetType.EXPONENTIAL
    elif pulsetype == FloquetablePulses.COSINE:
        return FloquetType.COSINE
    elif pulsetype == FloquetablePulses.SINE:
        return FloquetType.SINE
    elif pulsetype == FloquetablePulses.EXPONENTIAL:
        return FloquetType.EXPONENTIAL
    else:
        raise NotImplementedError(f"The quasi-floquet model for pulse type {pulsetype} is not implemented yet.")
    
def check_original_pulse(pulse: Pulse) -> FloquetType:
    try:
        pulse_type = FloquetablePulses(pulse.__class__.__name__)
    except ValueError:
        raise ValueError(f"The pulse {pulse.__class__.__name__} is not a Floquet-able pulse.")
    return get_floquet_type_from_pulsetype(pulse_type)

def check_validity_of_floquet_pulse(pulse: Pulse) -> None:
    try:
        ValidQuasiFloqeuetPulses(pulse.__class__.__name__)
    except ValueError:
        raise ValueError(f"The pulse {pulse.__class__.__name__} is not a valid quasi-Floquet pulse.")
    
    
class QuasiFloquetHamiltonian(TD_NonadiabaticHamiltonianBase):
    def __init__(
        self,
        dim: int,
        orig_pulse: Pulse,
        floq_pulse: Pulse,
        NF: int,
        Omega: Union[float, None]=None,
        floquet_type: Union[FloquetType, None]=None,
    ) -> None:
        """ Quasi-Floquet Hamiltonian for a time-dependent Hamiltonian """
        """ whose time dependence is definded by a 'Pulse' object. """
        if Omega is None:
            self.Omega = get_carrier_frequency(orig_pulse)
        else:
            assert Omega == get_carrier_frequency(orig_pulse)
            self.Omega = Omega
        assert self.Omega is not None 
        
        print(f"Omega: {self.Omega}")
        
        if floquet_type is None:
            self.floquet_type = check_original_pulse(orig_pulse)
        else:
            assert floquet_type == check_original_pulse(orig_pulse)
            self.floquet_type = floquet_type
            
        check_validity_of_floquet_pulse(floq_pulse)
            
        
        super().__init__(dim, floq_pulse)
        self.NF = NF
        self.floquet_type = floquet_type
        
    def H(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        return get_HF(self.H0(r), self.H1(t, r), self.Omega, self.NF, floquet_type=self.floquet_type) 
    
    def dHdR(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        return get_HF(self.dH0dR(r), self.dH1dR(t, r), self.Omega, self.NF, floquet_type=self.floquet_type)
    
    def get_floquet_space_dim(self) -> int:
        return _dim_to_dimF(self.dim, self.NF)
    
    def set_NF(self, NF: int) -> None:
        if isinstance(NF, int) and NF > 0:
            self.NF = NF
        else:
            raise ValueError(f"The number of Floquet replicas must be a positive integer, but {NF} is given.")
    
    def __call__(self, 
        t: Real, r: Union[Real, ArrayLike],
    ) -> Tuple[sp.csr_matrix, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        HF: sp.csr_matrix  = self.H(t, r)
        # evals, evecs = diagonalize_hamiltonian(HF, enforce_gauge=False) 
        evals, evecs = diagonalize_hamiltonian_history(HF, self.evec_last)
        self.evec_last = evecs
        dHdR: sp.csr_matrix = self.dHdR(t, r)
        d, F = evaluate_nonadiabatic_couplings(dHdR.toarray(), evals, evecs)
        return HF, evals, evecs, d, F 

# Methods

def _is_real_symmetric(hamiltonian: ArrayLike) -> bool:
    return np.allclose(hamiltonian, hamiltonian.T) and np.iscomplexobj(hamiltonian)

def matrix_col_dot(A: np.ndarray, B: np.ndarray):
    """ Compute the dot product of two matrices along the columns
    """
    return np.einsum('ij,ij->j', A, B)

def _enforce_gauge_from_last(evecs: ArrayLike, evecs_last: ArrayLike) -> ArrayLike:
    """Remove the gauge ambiguity by enforcing the continuity of the eigenvectors.
    Also known as the state-following, or the state tracking algorithm.

    Args:
        evecs (ArrayLike): the current eigenvectors
        evecs_last (ArrayLike): the previous eigenvectors

    Returns:
        ArrayLike: the gauge-consistent eigenvectors
        
    Reference:
        `mudslide` by Shane Parker. see the following link for original implementation:
        <https://github.com/smparker/mudslide>
    """
    signs = np.sign(matrix_col_dot(evecs, evecs_last.conjugate()))
    evecs *= signs
    return evecs

def diagonalize_hamiltonian_history(hamiltonian: ArrayLike, evec_last: Union[None, ArrayLike]=None) -> Tuple[ArrayLike, ArrayLike]:
    if isinstance(hamiltonian, sp.spmatrix):
        evals, evecs = LA.eigh(hamiltonian.toarray())
    else:
        evals, evecs = LA.eigh(hamiltonian)
    if evec_last is not None:
        evecs = _enforce_gauge_from_last(evecs, evec_last)
                
    return evals, evecs

def diagonalize_2d_real_symmetric(H: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    a = H[0, 0] 
    b = H[1, 1]
    c = H[0, 1]
        
    lambda1 = 0.5 * (a + b - np.sqrt((a - b)**2 + 4 * c**2))
    lambda2 = 0.5 * (a + b + np.sqrt((a - b)**2 + 4 * c**2))
        
    evals = np.array([lambda1, lambda2])
    theta = np.arctan2(2 * c, b - a) / 2
    evecs = np.array(
        [[ np.cos(theta), np.sin(theta)], 
         [-np.sin(theta), np.cos(theta)]]
    )
    return evals, evecs

def diabatic_to_adiabatic(
    O: ArrayLike,
    U: ArrayLike, 
    out: Union[ArrayLike, None]=None
)-> ArrayLike:
    if out is not None:
        np.dot(O, U, out=out)
        np.dot(U.conj().T, out, out=out)
    else:
        return np.dot(U.conj().T, np.dot(O, U)) 
    
def adiaobatic_to_diabatic(
    O: ArrayLike,
    U: ArrayLike, 
    out: Union[ArrayLike, None]=None
) -> ArrayLike:
    if out is not None:
        np.dot(U, O, out=out)
        np.dot(out, U.conj().T, out=out)
    else:
        return np.dot(U, np.dot(O, U.conj().T))
    
def evaluate_nonadiabatic_couplings(
    dHdR: ArrayLike,
    evals: ArrayLike,
    evecs: ArrayLike, 
    out_d: Union[ArrayLike, None]=None,
) -> Tuple[ArrayLike, ArrayLike]:
    if dHdR.ndim == 2:
        return _evaluate_nonadiabatic_couplings_scalar(dHdR, evals, evecs, out_d)
    elif dHdR.ndim == 3:
        return _evaluate_nonadiabatic_couplings_vector(dHdR, evals, evecs, out_d)
    else:
        raise ValueError(f"The number of dimensions of dHdR must be 2 or 3, but the input dHdR has {dHdR.ndim} dimensions.")

def _evaluate_nonadiabatic_couplings_scalar(
    dHdR: ArrayLike, 
    evals: ArrayLike,
    evecs: ArrayLike,
    out_d: Union[ArrayLike, None]=None,
) -> Tuple[ArrayLike, ArrayLike]:
    # assert dHdR.ndim == 2
    if out_d is not None:
        assert out_d.shape == dHdR.shape
        diabatic_to_adiabatic(dHdR, evecs, out=out_d)
    else:
        out_d = diabatic_to_adiabatic(dHdR, evecs)
    F = -np.diagonal(out_d)
    out_d = _post_process_d_scalar(out_d, evals)
    return out_d, F
    

def _evaluate_nonadiabatic_couplings_vector(
    dHdR: ArrayLike, 
    evals: ArrayLike,
    evecs: ArrayLike,
    out_d: Union[ArrayLike, None]=None,
) -> Tuple[ArrayLike, ArrayLike]:
    if out_d is not None:
        assert out_d.shape[0] == dHdR.shape[-1]
        ndim_cl = dHdR.shape[-1]
        for ii in range(ndim_cl):
            diabatic_to_adiabatic(dHdR[:, :, ii], evecs, out=out_d[ii])
    else:
        out_d = np.array([diabatic_to_adiabatic(dHdR[:, :, ii], evecs) for ii in range(dHdR.shape[-1])])
    F = -np.diagonal(out_d, axis1=1, axis2=2).T.astype(np.float64)
    out_d = _post_process_d_vector(out_d, evals)
    return out_d, F
 
@jit(nopython=True)  
def _post_process_d_scalar(out_d: ArrayLike, evals: ArrayLike) -> ArrayLike:
    ndim = out_d.shape[0]
    for ii in range(ndim):
        out_d[ii, ii] = 0.0
        for jj in range(ii+1, ndim):
            dE = evals[jj] - evals[ii]
            out_d[ii, jj] /= dE
            out_d[jj, ii] /= -dE
    return out_d

@jit(nopython=True)
def _post_process_d_vector(out_d: ArrayLike, evals: ArrayLike) -> ArrayLike:
    ndim_qm = evals.shape[0]
    ndim_cl = out_d.shape[0]
    for ii in range(ndim_cl):
        for jj in range(ndim_qm):
            out_d[ii, jj, jj] = 0.0
            for kk in range(ndim_qm):
                dE = evals[kk] - evals[jj]
                out_d[ii, jj, kk] /= dE
                out_d[ii, kk, jj] /= -dE
    return out_d
# %%
