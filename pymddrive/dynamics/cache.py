# %%
from attrs import define

from pymddrive.my_types import GenericOperator, GenericVectorOperator, RealVector, ActiveSurface, ComplexOperator, ComplexVector

from typing import Optional

@define(frozen=True)
class Cache:
     ##########
     # hamiltonian related cache
     ##########
     H: GenericOperator               # hamiltonian
     evals: RealVector                # eigenvalues
     evecs: GenericOperator           # eigenvectors
     dHdR: GenericVectorOperator      # derivative of hamiltonian w.r.t. nuclear coordinates
     nac: GenericVectorOperator       # non-adiabatic coupling

     ##########
     # ehrenfest related cache
     ##########
     meanF: Optional[RealVector] = None

     ##########
     # surface hopping related cache
     ##########
     active_surface: Optional[ActiveSurface] = None
     
     ##########
     # Langevin related cache
     ##########
     # treat the lanvegin force as first-order
     # that is, only update the friction and random force once per MD step
     # meaning each RK step will use the same friction and random force
     F_langevin: Optional[RealVector] = None

     
# %%
if __name__ == '__main__':
    from tests.test_utils import get_random_O, get_random_vO, compute_dc
    
    is_complex = False
    n_elec = 2
    n_nucl = 3
    
    H = get_random_O(n_elec, is_complex)
    dHdR = get_random_vO(n_elec, n_nucl, is_complex)
    nac, F, evals, evecs = compute_dc(H, dHdR)
    
    cache = Cache(
        H = H,
        evals = evals,
        evecs = evecs,
        dHdR = dHdR,
        nac = nac, 
    )
    
    print(cache)
# %%
