# %%
import attr
from attrs import define, field
import numpy as np

from pymddrive.my_types import GenericOperator, GenericVectorOperator, RealVector, ActiveSurface, ComplexOperator, ComplexVector

from typing import Optional

@define
class Cache:
    ##########
    # hamiltonian related cache
    ##########
    H: GenericOperator = field(on_setattr=attr.setters.frozen)
    evals: RealVector = field(on_setattr=attr.setters.frozen)
    evecs: GenericOperator = field(on_setattr=attr.setters.frozen)
    dHdR: GenericVectorOperator = field(on_setattr=attr.setters.frozen)
    nac: GenericVectorOperator  = field(on_setattr=attr.setters.frozen)

    ##########
    # ehrenfest related cache
    ##########
    meanF: RealVector = field(on_setattr=attr.setters.frozen)    

    ##########
    # surface hopping related cache
    ##########
    active_surface: ActiveSurface = field(on_setattr=attr.setters.frozen) 
     
    ##########
    # Langevin related cache
    ##########
    # treat the lanvegin force as first-order
    # that is, only update the friction and random force once per MD step
    # meaning each RK step will use the same friction and random force
    F_langevin: RealVector = field(on_setattr=attr.setters.frozen) 
    
    def update_cache(
        self, 
        H: Optional[GenericOperator] = None,
        evals: Optional[RealVector] = None,
        evecs: Optional[GenericOperator] = None,
        dHdR: Optional[GenericVectorOperator] = None,
        nac: Optional[GenericVectorOperator] = None,
        meanF: Optional[RealVector] = None,
        active_surface: Optional[ActiveSurface] = None,
        F_langevin: Optional[RealVector] = None,
    ) -> None:
        if H is not None:
            self.H[:] = H
        if evals is not None:
            self.evals[:] = evals
        if evecs is not None:
            self.evecs[:] = evecs
        if dHdR is not None:
            self.dHdR[:] = dHdR
        if nac is not None:
            self.nac[:] = nac
        if meanF is not None:
            self.meanF[:] = meanF
        if active_surface is not None:
            self.active_surface[:] = active_surface
        if F_langevin is not None:
            self.F_langevin[:] = F_langevin
            
        
    
    @classmethod    
    def from_dimensions(cls, dim_elec: int, dim_nucl: int) -> 'Cache':
        return cls(
            H = np.zeros((dim_elec, dim_elec), dtype=np.complex128),
            evals = np.zeros(dim_elec, dtype=np.float64),
            evecs = np.zeros((dim_elec, dim_elec), dtype=np.complex128),
            dHdR = np.zeros((dim_elec, dim_elec, dim_nucl), dtype=np.complex128),
            nac = np.zeros((dim_elec, dim_elec, dim_nucl), dtype=np.complex128),
            meanF = np.zeros(dim_nucl, dtype=np.float64),
            active_surface = np.zeros(1, dtype=np.int64),
            F_langevin = np.zeros(dim_nucl, dtype=np.float64),
        )
     
    

     
# %%
if __name__ == '__main__':
    from tests.test_utils import get_random_O, get_random_vO, compute_dc
    
    is_complex = False
    n_elec = 2
    n_nucl = 3
    
    H = get_random_O(n_elec, is_complex)
    dHdR = get_random_vO(n_elec, n_nucl, is_complex)
    nac, F, evals, evecs = compute_dc(H, dHdR)
    
    cache = Cache.from_dimensions(n_elec, n_nucl)
    cache.update_cache(H=H, evals=evals, evecs=evecs, dHdR=dHdR, nac=nac)
    
    
    
    
    print(cache)
# %%
