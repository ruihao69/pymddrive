# %%
import numpy as np
from numpy.typing import NDArray

from pymddrive.low_level.states import State, StateType, QuantumStateRepresentation

from typing import Union, Tuple, Optional

def preprocess_variables(R: Optional[Union[float, NDArray[np.float64]]], P: Optional[Union[float, NDArray[np.float64]]], rho_or_psi: Optional[NDArray[np.complex128]]) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]], Optional[NDArray[np.complex128]]]:
    if (R is not None) and (P is not None):
        R, P = process_classical_variables(R, P)
    if rho_or_psi is not None:
        rho_or_psi = _preprocess_quantum_variable(rho_or_psi)
    return R, P, rho_or_psi

def _preprocess_classical_variable(classical_var: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
    np_classical_var = np.array(classical_var)
    if np_classical_var.ndim == 0:
        np_classical_var = np.array([np_classical_var])
    elif np_classical_var.ndim > 1:
        raise ValueError(f"Invalid shape for classical variable: {np_classical_var.shape}")
    return np_classical_var

def _preprocess_quantum_variable(quantum_var: NDArray[np.complex128]) -> NDArray[np.complex128]:
    if quantum_var.ndim != 2:
        raise ValueError(f"Invalid shape for quantum variable: {quantum_var.shape}")
    return quantum_var if np.iscomplexobj(quantum_var) else quantum_var.astype(np.complex128)

def process_classical_variables(*classical_vars: Union[float, NDArray[np.float64]]) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    return tuple(_preprocess_classical_variable(var) for var in classical_vars)

def get_state(mass: float, R: Optional[Union[float, NDArray[np.float64]]], P: Optional[Union[float, NDArray[np.float64]]], rho_or_psi: Optional[NDArray[np.complex128]]) -> State:
    R, P, rho_or_psi = preprocess_variables(R, P, rho_or_psi)
    if (R is not None) and (P is not None) and (rho_or_psi is not None):
        return State(R, P, mass, rho_or_psi)
    elif (R is not None) and (P is not None):
        return State(R, P, mass)
    elif (R is None) and (P is None) and (rho_or_psi is not None):
        return State(rho_or_psi)
    else:
        raise ValueError("Invalid state variables")

# generation code for the State class 

# %% The temporary testing/debuging code
def _debug_test():
    R = 3
    P = 3.5
    rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    
    statemqc = get_state(2000.0, R, P, rho) 
    
    # test __repr__ 
    print(statemqc) 
    
    # test type
    print(type(statemqc))
    
    # test flatten
    print(statemqc.flatten())
    
    # a = statemqc + statemqc 
    # # print(a)
    # print(type(a))
    
    # test get_variables 
    r, p, rho = get_variables(statemqc)
    
    print(r)
    print(p)
    print(rho)
    print(r.shape)
    print(p.shape)
    print(rho.shape)
    
    # test from_unstructured
    flat = statemqc.flatten()
    statemqc2 = statemqc.from_unstructured(flat)
    print(f"{np.allclose(statemqc.flatten(), statemqc2.flatten())=}")
    
        
# %% The __main__ code
if __name__ == "__main__": 
    _debug_test() 

# %%
