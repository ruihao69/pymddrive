# %% The package
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured

from enum import Enum, unique
from dataclasses import dataclass

from numbers import Number, Real
from typing import Union, Self, Type
from numpy.typing import ArrayLike, DTypeLike

@unique
class StateType(Enum):
    ABSTRACT = 'abstract'
    CLASSICAL = 'classical'
    QUANTUM = 'quantum'
    MQC = 'mixed quantum classical'
   
@dataclass 
class StateData:
    data: Union[ArrayLike, None] = None
    stype: StateType = StateType.ABSTRACT
    
    
class State(StateData):
    def __init__(self, data: ArrayLike, stype: StateType=StateType.ABSTRACT) -> None:
        super().__init__(data, stype)
        
    @classmethod    
    def from_variables(
        cls: Type["State"], 
        R: Union[None, Real, ArrayLike]=None,
        P: Union[None, Real, ArrayLike]=None,
        rho: Union[None, ArrayLike]=None
    )-> "State":
        state_type: StateType = get_state_type(R, P, rho)
        if state_type == StateType.CLASSICAL:
            n = 1 if isinstance(R, Real) else len(R)
            dtype = np.dtype([("R", np.float64, (n, )), ("P", np.float64, (n, ))])
            data = np.array((R, P), dtype=dtype)
            return cls(data=data, stype=StateType.CLASSICAL)
        elif state_type == StateType.QUANTUM:
            n_dm = rho.shape[0]
            dtype = np.dtype([("rho", np.complex128, (n_dm, n_dm, ))])
            data = np.array((rho, ), dtype=dtype)
            return cls(data=data, stype=StateType.QUANTUM)
        elif state_type == StateType.MQC:
            n_dm = rho.shape[0]
            n = 1 if isinstance(R, Real) else len(R)
            dtype = np.dtype([("R", np.float64, (n, )), ("P", np.float64, (n, )), ("rho", np.complex128, (n_dm, n_dm, ))])
            data = np.array((R, P, rho), dtype=dtype)
            return cls(data=data, stype=StateType.MQC)
        else:
            raise NotImplemented 
    @classmethod 
    def from_unstructured(
        cls: Type["State"], 
        flat_data: ArrayLike, 
        dtype: DTypeLike,
        stype: StateType,
        copy: bool=False
    ) -> "State":
        data = unstructured_to_structured(flat_data, dtype, copy=copy)
        return cls(data, stype)
        
    def get_variables(self):
        if self.stype == StateType.CLASSICAL:
            return self.data["R"], self.data["P"], None
        elif self.stype == StateType.QUANTUM:
            return None, None, self.data["rho"]
        elif self.stype == StateType.MQC:
            return self.data["R"], self.data["P"], self.data["rho"]
        else:
            raise NotImplementedError(f'Cannot get variables for state type: {self.stype}')
        
    def flatten(self, copy: bool=False) -> ArrayLike:
        return structured_to_unstructured(self.data, copy=copy)
        
    def _get_raw_str(self):
        s1 = f"""State class wrapped around numpy's structured array"""
        s2 = f"""data fields: {self.data.dtype.names}"""
        s3 = f"""types: {tuple(self.data[name].dtype for name in self.data.dtype.names)}"""
        s4 = f"""shape: {tuple(self.data[name].shape for name in self.data.dtype.names)}""" 
        return (s1, s2, s3, s4)
    
    def __repr__(self) -> str:
        s1, s2, s3, s4 = self._get_raw_str()
        maxlen = max(len(s1), len(s2), len(s3), len(s4))
        sep = "-" * maxlen
        return f"""{sep}\n{s1:^{maxlen}}\n{sep}\n{s2}\n{s3}\n{s4}\n{sep}"""
        
    def __add__(self, other: Union[Self, Number]) -> "State":
        if isinstance(other, State):
            return State(data=_add_structured_arrays(self.data, other.data), stype=self.stype)
        elif isinstance(other, Number):
            return State(data=_scalar_add_structured_array(self.data, other), stype=self.stype)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'State' and '{}'".format(type(other)))
    
    def __mul__(self, scalar: Number) -> "State":
        if isinstance(scalar, Number):
            return State(data=_scalar_multiply_structured_array(self.data, scalar), stype=self.stype)
        else:
            raise TypeError("Unsupported operand type(s) for *: 'State' and '{}'".format(type(scalar)))
    
    def __lmul__(self, scalar: Number) -> "State":
        return self.__mul__(scalar)
    
    def __rmul__(self, scalar: Number) -> "State":
        return self.__mul__(scalar)   
    
    def __neg__(self) -> "State":
        return self.__mul__(-1)
    
    def __sub__(self, other: Union[Self, Number]) -> "State":
        return self.__add__(-other)
    
    def __truediv__(self, scalar: Number) -> "State":
        if isinstance(scalar, Number):
            return State(data=_scalar_multiply_structured_array(self.data, 1.0 / scalar), stype=self.stype)
    
    def __imul__(self, scalar: Number) -> "State":
        if isinstance(scalar, Number):
            for name in self.data.dtype.names:
                self.data[name] *= scalar
        else:
            raise TypeError("Unsupported operand type(s) for *: 'State' and '{}'".format(type(scalar)))
        return self
    
    def __iadd__(self, other: Union[Self, Number]) -> "State":
        if isinstance(other, State):
            for name in self.data.dtype.names:
                self.data[name] += other.data[name]
        elif isinstance(other, Number):
            for name in self.data.dtype.names:
                self.data[name] += other
        else:
            raise TypeError("Unsupported operand type(s) for +: 'State' and '{}'".format(type(other)))
        return self
    
    def __isub__(self, other: Union[Self, Number]) -> "State":
        if isinstance(other, State):
            for name in self.data.dtype.names:
                self.data[name] -= other.data[name]
        elif isinstance(other, Number):
            for name in self.data.dtype.names:
                self.data[name] -= other
        else:
            raise TypeError("Unsupported operand type(s) for -: 'State' and '{}'".format(type(other)))
        return self
    
    def __idiv__(self, scalar: Number) -> "State":
        if isinstance(scalar, Number):
            for name in self.data.dtype.names:
                self.data[name] /= scalar
        else:
            raise TypeError("Unsupported operand type(s) for /: 'State' and '{}'".format(type(scalar)))
        return self
    
def _add_structured_arrays(a: ArrayLike, b: Union[ArrayLike, Number]) -> ArrayLike:
    return np.array(tuple(a[name] + b[name] for name in a.dtype.names), dtype=a.dtype)
    
def _scalar_add_structured_array(a: ArrayLike, scalar: Number) -> ArrayLike:
    return np.array(tuple(a[name] + scalar for name in a.dtype.names), dtype=a.dtype)
    
def _scalar_multiply_structured_array(a: ArrayLike, scalar: Number) -> ArrayLike:
    return np.array(tuple(a[name] * scalar for name in a.dtype.names), dtype=a.dtype)
    
def _get_state_type(flag_none_r: bool, flag_none_p: bool, flag_none_rho: bool):
    if not flag_none_r and not flag_none_p and not flag_none_rho:
        return StateType.MQC
    elif not flag_none_r and not flag_none_p and flag_none_rho:
        return StateType.CLASSICAL
    elif flag_none_r and flag_none_p and not flag_none_rho:
        return StateType.QUANTUM
    else:
        raise ValueError(f"Cannot determine the state type from R: {not flag_none_r}, P: {not flag_none_p}, and rho: {not flag_none_rho}.")
    
def get_state_type(
    R: Union[None, Real, ArrayLike], 
    P: Union[None, Real, ArrayLike], 
    rho: Union[None, ArrayLike]
) -> StateType:
    flag_none_r = R is None
    flag_none_p = P is None
    flag_none_rho = rho is None
    return _get_state_type(flag_none_r, flag_none_p, flag_none_rho) 

    
      
def zeros_like(state: State) -> State:
    return State(data=np.zeros_like(state.data), stype=state.stype)

# %% The temporary testing/debuging code
def _debug_test():
    R = 3
    P = 3.5
    rho = np.array([[1, 0], [0, 0]])
    
    statemqc = State.from_variables(R=R, P=P, rho=rho)
    # print(statemqc)
    
    statecl = State.from_variables(R=R, P=P, rho=None)
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
    
    s1 = State.from_variables(R=1, P=2, rho=None)
    s2 = State.from_variables(R=2, P=1, rho=None)
    
    s = s1 + s2
    
    print(f"{s.data=}")
    
    
        
# %% The __main__ code
if __name__ == "__main__": 
    _debug_test() 

# %%
