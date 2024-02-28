# %%
import numpy as np
import scipy.linalg as LA
import scipy.sparse as sp

from numba import jit   

from abc import ABC, abstractmethod

from typing import Tuple, Union
from numpy.typing import ArrayLike

from pymddrive.pulses.pulses import Pulse, get_carrier_frequency
from pymddrive.models.floquet import get_HF, FloquetType, _dim_to_dimF


class NonadiabaticHamiltonianBase(ABC):
    def __init__(
        self,
        dim: int,
    ) -> None:
        self.dim = dim
        
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
    
class FloquetHamiltonian(TD_NonadiabaticHamiltonianBase):
    def __init__(
        self,
        dim: int,
        pulse: Pulse,
        NF: int,
        Omega: Union[float, None]=None,
        floquet_type: FloquetType=FloquetType.COSINE,
    ) -> None:
        """ Quasi-Floquet Hamiltonian for a time-dependent Hamiltonian """
        """ whose time dependence is definded by a 'Pulse' object. """
        if Omega is None:
            self.Omega = get_carrier_frequency(pulse)
        else:
            assert Omega == get_carrier_frequency(pulse)
            self.Omega = Omega
            
        super().__init__(dim, pulse)
        self.NF = NF
        self.floquet_type = floquet_type
        
        if floquet_type == FloquetType.COSINE:
            # for cosine type floquet hamiltonian, setting the carrier frequency to zero
            # will effectively negate the fast oscillating carriers within the envolope of a pulse
            self.pulse.set_Omega(0.0)
        elif floquet_type == FloquetType.SINE:
            # for sine type floquet hamiltonian, setting the carrier frequency to pi/2
            # will effectively negate the fast oscillating carriers within the envolope of a pulse
            self.pulse.set_Omega(np.pi / 2)
        else:
            raise NotImplementedError(f"The floquet type {floquet_type} is not implemented.")
        
        
    def H(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        return get_HF(self.H0(r), self.H1(t, r), self.Omega, self.NF, floquet_type=self.floquet_type) 
    
    def dHdR(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        return get_HF(self.dH0dR(r), self.dH1dR(t, r), self.Omega, self.NF, floquet_type=self.floquet_type)
    
    def get_floquet_space_dim(self) -> int:
        return _dim_to_dimF(self.dim, self.NF)
    

# Methods

def _enforce_gauge(evecs: ArrayLike) -> ArrayLike:
    if np.iscomplexobj(evecs):
        negate_phase = np.exp(-1j * np.angle(evecs.diagonal()))
        evecs *= negate_phase
    else:
        sign = np.sign(evecs.diagonal())
        evecs *= sign
    return evecs

def _is_real_symmetric(hamiltonian: ArrayLike) -> bool:
    return np.allclose(hamiltonian, hamiltonian.T) and np.iscomplexobj(hamiltonian)

def diagonalize_hamiltonian(hamiltonian: ArrayLike, enforce_gauge: bool=True) -> Tuple[ArrayLike, ArrayLike]:
    if isinstance(hamiltonian, sp.spmatrix):
        evals, evecs = LA.eigh(hamiltonian.toarray())
    else:
        evals, evecs = LA.eigh(hamiltonian)
    if enforce_gauge:
        evecs = _enforce_gauge(evecs) 
    else:
        pass
                
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
