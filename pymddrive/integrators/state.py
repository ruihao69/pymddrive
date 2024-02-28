# %% The package
import numpy as np
from dataclasses import dataclass

from typing import Union
from numpy.typing import ArrayLike, DTypeLike

from pymddrive.integrators.structured_data import CompositeData

@dataclass
class StateClassical:
    dim_classical: int 
    dim_quantum: None
    
    def __post_init__(self):
        self.dtype: DTypeLike = [
            ('R', np.float64, (self.dim_classical,)),
            ('P', np.float64, (self.dim_classical,)),
        ]
    
@dataclass
class StateQuantum:
    dim_classical: None
    dim_quantum: int
    
    def __post_init__(self):
        self.dtype: DTypeLike = [
            ('rho', np.complex128, (self.dim_quantum, self.dim_quantum)),
        ]
    
@dataclass
class StateMQC:
    dim_classical: int 
    dim_quantum: int
    
    def __post_init__(self):
        self.dtype: DTypeLike = [
            ('R', np.float64, (self.dim_classical,)),
            ('P', np.float64, (self.dim_classical,)),
            ('rho', np.complex128, (self.dim_quantum, self.dim_quantum)),
        ]
    
def check_state_type(
    r: Union[None, float, int, np.ndarray]=None, 
    p: Union[None, float, int, np.ndarray]=None, 
    rho: Union[None, ArrayLike]=None
) -> Union[StateClassical, StateQuantum, StateMQC]:
    def get_classical_dim(r, p):
        if isinstance(r, (float, int)) and isinstance(p, (float, int)):
            return 1
        if isinstance(r, np.ndarray) and isinstance(p, np.ndarray):
            if r.size == p.size:
                return r.size
            else:
                raise ValueError("The size of r and p must be the same.")
        raise ValueError("The types of r and p must be the same.")
        
    if r is None and p is None and rho is None:
        return None
        # raise ValueError("Cannot determine a state when the position, momentum, and density matrix are all None.") 
    
    if r is not None and p is not None and rho is None:
        return StateClassical(dim_classical=get_classical_dim(r, p), dim_quantum=None)        
    
    if r is None and p is None and rho is not None:
        return StateQuantum(dim_classical=None, dim_quantum=rho.shape[0])
    
    if r is not None and p is not None and rho is not None:
        return StateMQC(dim_classical=get_classical_dim(r, p), dim_quantum=rho.shape[0])
    
    raise ValueError(f"Cannot determine the state type. You input types are R({type(r)}), P({type(p)}), and rho({type(rho)}).")

def get_dtype(state_type: Union[StateClassical, StateQuantum, StateMQC]) -> DTypeLike:
    return state_type.dtype

def initialize_data(
    state_type: Union[StateClassical, StateQuantum, StateMQC],
    r: Union[None, float, int, np.ndarray]=None,
    p: Union[None, float, int, np.ndarray]=None,
    rho: Union[None, ArrayLike]=None, 
) -> ArrayLike:
    dtype = get_dtype(state_type)
    data = np.zeros(1, dtype=dtype)[0]
    
    if r is not None:
        data['R'][:] = r
    if p is not None:
        data['P'][:] = p
    if rho is not None:
        data['rho'][:] = rho
    
    return data

class State(CompositeData):
    def __init__(
        self, 
        r: Union[None, float, int, np.ndarray]=None,
        p: Union[None, float, int, np.ndarray]=None,
        rho: Union[None, ArrayLike]=None,
        *args, 
        **kwargs
    ) -> None:
        state_type = check_state_type(r, p, rho)
        if state_type is None:
            try:
                super().__init__(*args, **kwargs)
            except:
                raise ValueError("Neither r, p and/or rho or data is provided in the argument.")
        else:
            data = initialize_data(state_type, r, p, rho)
        
            super().__init__(data, *args, **kwargs)
        
        # self.r, self.p, self.rho = self.get_variables()
        
    def __array__(self) -> ArrayLike:
        return self.data
        
    def get_variables(self):
        r = self.data['R'] if "R" in self.data.dtype.names else None
        p = self.data['P'] if "P" in self.data.dtype.names else None
        rho = self.data['rho'] if "rho" in self.data.dtype.names else None
        return r, p, rho
    
def zeros_like(state: State) -> State:
    return State(r=None, p=None, rho=None, data=np.zeros(1, dtype=state.data.dtype)[0])

# %% The temporary testing/debuging code
def _debug_test():
    r = 3
    p = 3.5
    rho = np.array([[1, 0], [0, 0]])
    
    statemqc = State(r=r, p=p, rho=rho)
    # print(statemqc)
    
    statecl = State(r=r, p=p, rho=None)
    # print(statecl)
    # print(statecl.r.shape)
    
    print(type(statemqc))
    
    a = statemqc + statemqc 
    # print(a)
    print(type(a))
    
    statemqc += statemqc
    print(statemqc)
    print(type(statemqc))
    r, p, rho = statemqc.get_variables()
    
    print(r)
    print(p)
    print(rho)
    print(r.shape)
    print(p.shape)
    print(rho.shape)
    
        
# %% The __main__ code
if __name__ == "__main__": 
    _debug_test() 

# %%
