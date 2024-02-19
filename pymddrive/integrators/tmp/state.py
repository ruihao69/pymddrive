# %% The package: pymddrive.integrators.state
from dataclasses import dataclass

import numpy as np

from numpy.typing import ArrayLike, DTypeLike
from typing import Any, Union
from numbers import Number
from numpy.lib import recfunctions as rfn

from pymddrive.utils import is_hermitian_matrix


@dataclass
class State:
    dim_classical: Union[int, None]
    dim_quantum: Union[int, None]
    dtype: DTypeLike
    data: ArrayLike
    
    def __repr__(self) -> str:
        # return "State Default Class"
        return f"""{self.__class__.__name__}
    * dim_classical: {self.get_dimclassical()}
    * dim_quantum: {self.get_dimquantum()}
        """
        
    def __copy__(self):
        return self.__class__(
            dim_classical=self.dim_classical,
            dim_quantum=self.dim_quantum,
            dtype=self.dtype,
            data=self.data
        )
       
    def get_dimclassical(self):
        return self.dim_classical
    
    def get_dimquantum(self):
        return self.dim_quantum
    
    def flatten(self, copy=True):
        return rfn.structured_to_unstructured(np.asarray(self), copy=copy)

class StateClassical(State):  
    def __init__(
        self, 
        r: Union[Number, None],
        p: Union[Number, None],
        *args,
        **kwargs
    ) -> None:
        if r is None:
            super().__init__(*args, **kwargs)
        else:
            dim_quantum = None
            if isinstance(r, float | int):
                # self.dim_classical = 1
                dim_classical = 1
            elif len(r.shape) == 0:
                dim_classical = 1 
            elif len(r.shape) == 1:
                dim_classical = r.shape[0]
            else:
                raise NotImplementedError("Can only deal with 1d classical arraies (single atom/mode).") 
        
        
            dtype: DTypeLike = [
                ('R', np.float64, dim_classical),
                ('P', np.float64, dim_classical)
            ]
            data = np.zeros(1, dtype)
            data['R'] = r
            data['P'] = p
        
            super().__init__(
                dim_classical=dim_classical,
                dim_quantum=dim_quantum,
                dtype=dtype,
                data=data 
            )
            
        self.r = self.data['R']
        self.p = self.data['P']
        
    def __add__(self, other):
        return StateClassical(
            r=None, p=None,
            dim_classical=self.dim_classical,
            dim_quantum = self.dim_quantum,
            dtype=self.dtype,
            data=np.array((
                self.r + other.r, self.p + other.p
            ), dtype=self.dtype)
        )
        # return StateClassical(
        #     r = self.r + other.r,
        #     p = self.p + other.p
        # )
        
    def __sub__(self, other):
        return StateClassical(
            r=None, p=None,
            dim_classical=self.dim_classical,
            dim_quantum = self.dim_quantum,
            dtype=self.dtype,
            data=np.array((
                self.r - other.r, self.p - other.p
            ), dtype=self.dtype)
        )
        # return StateClassical(
        #     r = self.r - other.r,
        #     p = self.p - other.p
        # )
        
    def __mul__(self, scalar: float | int | complex):
        return StateClassical(
            r=None, p=None,
            dim_classical=self.dim_classical,
            dim_quantum = self.dim_quantum,
            dtype=self.dtype,
            data=np.array((
                self.r * scalar, self.p * scalar 
            ), dtype=self.dtype)
        )
        # return StateClassical(
        #     r = self.r * scalar,
        #     p = self.p * scalar
        # )
        
    def __imul__(self, scalar) -> "StateClassical":
        self.r *= scalar
        self.p *= scalar
        return self
        
        
    def __lmul__(self, other):
        return self.__mul__(other)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __iadd__(self, other):
        self.r += other.r
        self.p += other.p
        return self
               
class StateQuantum(State):
    def __init__(
        self, 
        rho: Union[ArrayLike, None],
        *args,
        **kwargs
    ) -> None:
        if rho is None:
            super().__init__(*args, **kwargs)
        else:
            dim_classical = None
            dim_quantum = rho.shape[-1]
            dtype: DTypeLike = [
                ('rho', np.complex128, (dim_quantum, dim_quantum))
            ]
            data = np.zeros(1, dtype=dtype)
            super().__init__(
                dim_classical=dim_classical,
                dim_quantum=dim_quantum,
                dtype=dtype,
                data=data
            )
            
        self.rho = self.data['rho']
    
    def __add__(self, other):
        return StateQuantum(
            rho=None,
            dim_classical=self.dim_classical,
            dim_quantum=self.dim_quantum ,
            dtype=self.dtype,
            data=np.array((self.rho + other.rho), dtype=self.dtype)
        )
                
    def __sub__(self, other):
        return StateQuantum(
            rho=None,
            dim_classical=self.dim_classical,
            dim_quantum=self.dim_quantum ,
            dtype=self.dtype,
            data=np.array((self.rho - other.rho), dtype=self.dtype)
        )
        # return StateClassical(
        #     rho=self.rho-other.rho
        # )
    
    def __mul__(self, scalar: float | int | complex):
        return StateQuantum(
            rho=None,
            dim_classical=self.dim_classical,
            dim_quantum=self.dim_quantum ,
            dtype=self.dtype,
            data=np.array((self.rho*scalar), dtype=self.dtype)
        )
        # return StateQuantum(
        #     rho=self.rho*scalar
        # )
        
    def __imul__(self, scalar) -> "StateQuantum":
        self.rho *= scalar
        return self
        
    def __lmul__(self, other):
        return self.__mul__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __iadd__(self, other):
        self.rho += other.rho
        return self 

class StateMQC(State):
    
    def __init__(
        self,
        r: Union[None, float, ArrayLike],
        p: Union[None, float, ArrayLike],
        rho: Union[None, ArrayLike],
        *args,
        **kwargs
    ) -> None:
        if r is None:
            super().__init__(*args, **kwargs)
        else:
            if isinstance(r, float | int):
                # self.dim_classical = 1
                dim_classical = 1
            elif len(r.shape) == 0:
                dim_classical = 1 
            elif len(r.shape) == 1:
                dim_classical = r.shape[0]
            else:
                raise NotImplementedError("Can only deal with 1d classical arraies (single atom/mode).") 
            
            dim_quantum = rho.shape[-1]
        
            dtype: DTypeLike = [
                ('R', np.float64, (dim_classical,)),
                ('P', np.float64, (dim_classical,)),
                ('rho', np.complex128, (dim_quantum, dim_quantum))
            ]
            data = np.zeros(1, dtype)
            data['R'] = r
            data['P'] = p
            data['rho'] = rho
        
            super().__init__(
                dim_classical=dim_classical,
                dim_quantum=dim_quantum,
                dtype=dtype,
                data=data 
            )
            
        self.r = self.data['R']
        self.p = self.data['P']
        self.rho = self.data['rho']
    
    def __add__(self, other):
        return StateMQC(
            r=None, p=None, rho=None,
            dim_classical=self.dim_classical,
            dim_quantum=self.dim_quantum,
            dtype=self.dtype,
            data=np.array((
                self.r + other.r,
                self.p + other.p,
                self.rho + other.rho
            ), dtype=self.dtype)
        )
        # return StateMQC(
        #     r=self.r + other.r,
        #     p=self.p + other.p,
        #     rho=self.rho + other.rho
        # )
        
    def __sub__(self, other):
        return StateMQC(
            r=None, p=None, rho=None,
            dim_classical=self.dim_classical,
            dim_quantum=self.dim_quantum,
            dtype=self.dtype,
            data=np.array((
                self.r - other.r,
                self.p - other.p,
                self.rho - other.rho
            ), dtype=self.dtype)
        )
        # return StateMQC(
        #     r=self.r - other.r,
        #     p=self.p - other.p,
        #     rho=self.rho - other.rho
        # )
        
    def __mul__(self, scalar: float | int | complex):
        return StateMQC(
            r=None, p=None, rho=None,
            dim_classical=self.dim_classical,
            dim_quantum=self.dim_quantum,
            dtype=self.dtype,
            data=np.array((
                self.r * scalar,
                self.p * scalar,
                self.rho * scalar 
            ), dtype=self.dtype)
        )
        # return StateMQC(
        #     r=self.r * scalar,
        #     p=self.p * scalar,
        #     rho=self.rho * scalar
        # )
        
    def __imul__(self, scalar: float | int | complex):
        self.r *= scalar
        self.p * scalar
        self.rho *= scalar
        return self
        
    def __lmul__(self, other):
        return self.__mul__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __iadd__(self, other):
        self.r += other.r
        self.p += other.p
        self.rho += other.rho
        return self
    
def zeros_like(s: State):
    ttype = type(s)
    return ttype(
        r=None, p=None, rho=None,
        dim_classical=s.dim_classical,
        dim_quantum=s.dim_quantum,
        dtype=s.dtype,
        data=np.zeros(1, dtype=s.dtype)
    )
# %% The temporary code snippet for the main function
if __name__ == "__main__":
    rho = np.array([[1, 2], [2, 1]], dtype=np.complex128)
    s = StateMQC(r=1.0, p=2.0, rho=rho)
    # s = StateClassical(r=1.0, p=2.0)
    _2s = s + s
    print(_2s.data) 
    s*=2; c=s
    print(s.data)
    print(c.data)
    # print(s.data)
    # print(s.data)
# %%
