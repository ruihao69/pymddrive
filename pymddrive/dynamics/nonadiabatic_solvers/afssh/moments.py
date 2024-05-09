import numpy as np
import attr
from attrs import define, field

from pymddrive.my_types import ComplexVectorOperator

@define 
class Moments:
    delta_R: ComplexVectorOperator = field(on_setattr=attr.setters.frozen)
    delta_P: ComplexVectorOperator = field(on_setattr=attr.setters.frozen)
    
    
    @classmethod
    def initialize(cls, n_classical: int, n_quantum: int) -> "Moments":
        shape = (n_quantum, n_quantum, n_classical)
        
        delta_R = np.zeros(shape=shape, dtype=np.complex128)
        delta_P = np.zeros(shape=shape, dtype=np.complex128)
