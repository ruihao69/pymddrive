import numpy as np
from numpy.typing import NDArray

from .langevin_base import LangevinBase

from dataclasses import dataclass
from typing import Union, Optional

class NullLangevin(LangevinBase):
    def __init__(self,) -> None:
        pass
    
    # override the get_gamma method in LangevinBase
    def get_gamma(self, t: float, R: Optional[NDArray[np.float64]]=None) -> NDArray[np.float64]:
        return 0.0
    
    def get_kT(self) -> None:
        return None
    
    def get_beta(self) -> None:
        return None
    
    # def frictional_force(
    #     self,
    #     gamma: NDArray[np.float64],
    #     P: NDArray[np.float64],
    # ) -> NDArray[np.float64]:
    #     return np.zeros_like(P)
    def evaluate_langevin(
        self,
        t: float,
        R: NDArray[np.float64],
        P: NDArray[np.float64],
        dt: float,
    ) -> NDArray[np.float64]:
        """override the evaluate_langevin method in LangevinBase

        Args:
            t (float): _description_
            R (NDArray[np.float64]): _description_
            P (NDArray[np.float64]): _description_
            dt (float): _description_

        Returns:
            NDArray[np.float64]: _description_
        """
        return np.zeros_like(P)