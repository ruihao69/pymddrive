from numpy.typing import ArrayLike

from abc import ABC, abstractmethod
from typing import Union

class HamiltonianBase(ABC):
    def __init__(
        self,
        dim: int,
    ) -> None:
        self.dim: int = dim
        self.last_evecs: Union[ArrayLike, None] = None
     
    @abstractmethod
    def H(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        pass
    
    @abstractmethod
    def dHdR(self, t: float, r: Union[float, ArrayLike]) -> ArrayLike:
        pass
    
    def update_last_evecs(self, evecs: ArrayLike) -> None:
        self.last_evecs = evecs