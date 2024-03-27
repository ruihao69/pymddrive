# %%
import numpy as np
from numpy.typing import NDArray

from pymddrive.low_level.states import DensityMatrixState

from typing import Union, Tuple, Optional
from enum import Enum, unique
from dataclasses import dataclass

    
AVAILABLE_STATE_TYPES = Union[DensityMatrixState]


@unique
class StateType(Enum):
    ABSTRACT = 'abstract'
    CLASSICAL = 'classical'
    QUANTUM = 'quantum'
    MQC = 'mixed quantum classical'
    
@unique 
class QuantumType(Enum):
    NONE = 'none' # No quantum state attached
    WAVEFUNCTION = 'wavefunction' # Wavefunction state
    DENSITY_MATRIX = 'density matrix' # Density matrix state
    
    
@dataclass(frozen=True)
class State:
    state_data: AVAILABLE_STATE_TYPES
    state_type: StateType
    quantum_type: QuantumType
    
    @classmethod
    def from_variables(
        cls,
        R: Optional[Union[float, NDArray[np.float64]]]=None,
        P: Optional[Union[float, NDArray[np.float64]]]=None,
        rho_or_psi: Optional[NDArray[np.complex128]]=None
    ) -> "State":
        state_type, quantum_type = infer_datatype(R, P, rho_or_psi)
        if state_type == StateType.CLASSICAL:
            R, P = process_classical_variables(R, P)
            raise NotImplementedError("Classical states are not yet implemented")
        elif state_type == StateType.QUANTUM:
            rho = _preprocess_quantum_variable(rho_or_psi)
            raise NotImplementedError("Pure quantum states are not yet implemented")
        elif state_type == StateType.MQC:
            R, P = process_classical_variables(R, P)
            if quantum_type == QuantumType.WAVEFUNCTION:
                psi = _preprocess_quantum_variable(rho_or_psi)
                raise NotImplementedError("Wavefunction states are not yet implemented")
            else:
                rho = _preprocess_quantum_variable(rho_or_psi)
                return cls(DensityMatrixState(R, P, rho), state_type, quantum_type)
        else:
            raise NotImplemented
    
    @classmethod
    def from_unstructured(
        cls,
        flat_data: NDArray[np.complex128],
        state_template: "State"
    ) -> "State":
        new_state_data = state_template.state_data.from_unstructured(flat_data)
        return cls(new_state_data, state_template.state_type, state_template.quantum_type)
    
    def _get_raw_str(self) -> Tuple[str, str, str, str]:
        low_level_name = self.state_data.__class__.__name__
        fields = self._get_data_fields()
        shapes = self._get_data_shapes(fields)
        
        s1 = f"""State class wrapped around low level c++ structs with Eigen3 Tensors""" 
        s2 = f"""low level class: {low_level_name}"""
        s3 = f"""data fields: {fields}"""
        s4 = f"""shape: {shapes}""" 
        return (s1, s2, s3, s4)
    
    @classmethod
    def axpy(cls, a: float, x: "State", y: "State") -> "State":
        res = x.state_data.axpy(a, x.state_data, y.state_data)
        return cls(res, x.state_type, x.quantum_type) 
    
    def rk4_step_inplace(self, dt: float, k1: "State", k2: "State", k3: "State", k4: "State") -> None:
        self.state_data.rk4_step_inplace(dt, k1.state_data, k2.state_data, k3.state_data, k4.state_data) 
    
    def __repr__(self) -> str:
        s1, s2, s3, s4 = self._get_raw_str()
        maxlen = max(len(s1), len(s2), len(s3), len(s4))
        sep = "-" * maxlen
        return f"""{sep}\n{s1:^{maxlen}}\n{sep}\n{s2}\n{s3}\n{s4}\n{sep}"""
     
    def flatten(self) -> NDArray[np.complex128]:
        return self.state_data.flatten()
    
    def get_variables(self) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]], Optional[NDArray[np.complex128]]]:
        if (self.state_type == StateType.CLASSICAL) or (self.state_type == StateType.MQC):
            R, P = self.state_data.R, self.state_data.P
        else:
            R, P = None, None
            
        if (self.state_type == StateType.QUANTUM) or (self.state_type == StateType.MQC):
            rho_or_psi = self.state_data.rho if self.quantum_type == QuantumType.DENSITY_MATRIX else self.state_data.psi
        else:
            rho_or_psi = None
        
        return R, P, rho_or_psi
    
    def _get_data_fields(self):
        if self.state_type == StateType.CLASSICAL:
            return ("R", "P")
        else:
            quantum_var = "rho" if self.quantum_type == QuantumType.DENSITY_MATRIX else "psi"
            if self.state_type == StateType.MQC:
                return ("R", "P", quantum_var)
            else:
                return (quantum_var, )  
            
    def _get_data_shapes(self, data_fields):
        return tuple(getattr(self.state_data, field).shape for field in data_fields)

def _get_state_type(flag_none_r: bool, flag_none_p: bool, flag_none_rho: bool):
    if not flag_none_r and not flag_none_p and not flag_none_rho:
        return StateType.MQC
    elif not flag_none_r and not flag_none_p and flag_none_rho:
        return StateType.CLASSICAL
    elif flag_none_r and flag_none_p and not flag_none_rho:
        return StateType.QUANTUM
    else:
        raise ValueError(f"Cannot determine the state type from R: {not flag_none_r}, P: {not flag_none_p}, and rho: {not flag_none_rho}.")
    
def _get_quantum_state_type(rho_or_psi: NDArray[np.complex128]) -> QuantumType:
    if rho_or_psi is None:
        return QuantumType.NONE
    elif rho_or_psi.ndim == 1:
        return QuantumType.WAVEFUNCTION
    elif rho_or_psi.ndim == 2:
        return QuantumType.DENSITY_MATRIX
    else:
        raise ValueError(f"Invalid quantum state shape: {rho_or_psi.shape}")
    
def _preprocess_classical_variable(classical_var: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
    np_classical_var = np.array(classical_var)
    if np_classical_var.ndim == 0:
        np_classical_var = np.array([np_classical_var])
    elif np_classical_var.ndim > 1:
        raise ValueError(f"Invalid shape for classical variable: {np_classical_var.shape}")
    return np_classical_var

def _preprocess_quantum_variable(quantum_var: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return quantum_var if np.iscomplexobj(quantum_var) else quantum_var.astype(np.complex128)

def process_classical_variables(*classical_vars: Union[float, NDArray[np.float64]]) -> Tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    return tuple(_preprocess_classical_variable(var) for var in classical_vars)
    
    
def infer_datatype(
    R: Optional[Union[float, NDArray[np.float64]]],
    P: Optional[Union[float, NDArray[np.float64]]],
    rho_or_psi: Optional[NDArray[np.complex128]]
) -> Union[StateType, QuantumType]:
    flag_none_r = R is None
    flag_none_p = P is None
    flag_none_rho_or_psi = rho_or_psi is None
    state_type = _get_state_type(flag_none_r, flag_none_p, flag_none_rho_or_psi)
    quantum_type = _get_quantum_state_type(rho_or_psi)
    return state_type, quantum_type
      
def zeros_like(state: State) -> State:
    return State.from_unstructured(np.zeros_like(state.flatten()), state)

# %% The temporary testing/debuging code
def _debug_test():
    R = 3
    P = 3.5
    rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    
    statemqc = State.from_variables(R, P, rho)
    
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
    r, p, rho = statemqc.get_variables()
    
    print(r)
    print(p)
    print(rho)
    print(r.shape)
    print(p.shape)
    print(rho.shape)
    
    # test from_unstructured
    flat = statemqc.flatten()
    statemqc2 = State.from_unstructured(flat, statemqc)
    print(f"{np.allclose(statemqc.flatten(), statemqc2.flatten())=}")
    
        
# %% The __main__ code
if __name__ == "__main__": 
    _debug_test() 

# %%
