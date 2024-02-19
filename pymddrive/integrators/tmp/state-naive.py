# %% The package: pymddrive.integrators.state
from dataclasses import dataclass

import numpy as np

from numpy.typing import ArrayLike, DTypeLike
from typing import Any, Union
from numpy.lib import recfunctions as rfn

from pymddrive.utils import is_hermitian_matrix


@dataclass
class State:
    # dim_classical: int
    # dim_quantum: int
    # dtype: DTypeLike
    # data: ArrayLike
    # init_from_self: bool = False
    
    def __repr__(self) -> str:
        return "State Default Class"
    
    def __add__(self, other):
        pass
    
    def __sub__(self, other):
        pass
        
    
    def __mul__(self, scalar: Union[float, int, complex]):
        pass
    
    def __lmul__(self, other):
        pass
    
    def __rmul__(self, other):
        pass
    
    def __iadd__(self, other):
        pass
        
    def flatten(self):
        pass
    
    def get_dimclassical(self):
        return self.dim_classical
    
    def get_dimquantum(self):
        return self.dim_quantum

class StateClassical(State):  
    def __init__(
        self, 
        r,
        p,
        dim_classical: int = None,
        dim_quantum: int = None,
        data: ArrayLike = None,
        *args,
        **kwargs
    ) -> None:
        self.r = np.array(r)
        self.p = np.array(p)
        
    def __repr__(self) -> str:
        return f"""
    StateClassical(
        type: {self.dtype}, 
        dims_cl: {self.dim_classical}, 
        dims_qm: {self.dim_quantum}
    )
    """
    
    def __add__(self, other):
        return StateClassical(
            r = self.r + other.r,
            p = self.p + other.p
        )
        
    def __sub__(self, other):
        return StateClassical(
            r = self.r - other.r,
            p = self.p - other.p
        )
        
    def __mul__(self, scalar: float | int | complex):
        return StateClassical(
            r = self.r * scalar,
            p = self.p * scalar
        )
        
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
        rho: ArrayLike,
        dim_classical: int = None,
        dim_quantum: int = None,
        *args,
        **kwargs
    ) -> None:
        self.rho = rho
        self.dim_classical = None
        self.dim_quantum = rho.shape[0]
        
    def __repr__(self) -> str:
        return f"StateQuantum(dim={self.rho.shape})"
    
    def __add__(self, other):
        return StateClassical(
            rho=self.rho+other.rho
        )
        
    def __sub__(self, other):
        return StateClassical(
            rho=self.rho-other.rho
        )
    
    def __mul__(self, scalar: float | int | complex):
        return StateQuantum(
            rho=self.rho*scalar
        )
        
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
        self.r = r
        self.p = p
        self.rho = rho
        if isinstance(r, (float, int)):
            self.dim_classical = 1
        else:
            # print(r)
            # self.dim_classical = r.shape[0]
            pass
        self.dim_quantum = rho.shape[0]
             
    def __repr__(self) -> str:
        return f"""StateMQC(
        type: {self.dtype}, 
        dims_cl: {self.dim_classical}, 
        dims_qm: {self.dim_quantum}
    )"""
    
    def __add__(self, other):
        return StateMQC(
            r=self.r + other.r,
            p=self.p + other.p,
            rho=self.rho + other.rho
        )
        
    def __sub__(self, other):
        return StateMQC(
            r=self.r - other.r,
            p=self.p - other.p,
            rho=self.rho - other.rho
        )
        
    def __mul__(self, scalar: float | int | complex):
        return StateMQC(
            r=self.r * scalar,
            p=self.p * scalar,
            rho=self.rho * scalar
        )
        
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
    if isinstance(s, StateClassical):
        return StateClassical(
            r=np.zeros_like(s.r),
            p=np.zeros_like(s.p)
        )
    if isinstance(s, StateQuantum):
        return StateQuantum(
            rho=np.zeros_like(s.rho)
        )
    if isinstance(s, StateMQC):
        return StateMQC(
            r=np.zeros_like(s.r),
            p=np.zeros_like(s.p),
            rho=np.zeros_like(s.rho)
        ) 

# %% The temporary code snippet for the main function
if __name__ == "__main__":
    rho = np.array([[1, 2], [2, 1]], dtype=np.complex128)
    s = StateMQC(r=1.0, p=2.0, rho=rho)
    # s = StateClassical(r=1.0, p=2.0)
    _2s = s + s
    # print(s.data)
    # print(s.data)
# %%
