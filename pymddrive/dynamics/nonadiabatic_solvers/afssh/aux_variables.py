import attr
from attrs import define, field

from pymddrive.my_types import RealDiagonalVectorOperator

@define
class AuxVariables:
    """ Simplified version of A-FSSH auxiliary variables, see """
    """ J. Chem. Theory Comput. 2016, 12, 5256-5268. DOI: 10.1021/acs.jctc.6b00673"""
    delta_R: RealDiagonalVectorOperator = field(on_setattr=attr.setters.frozen)
    delta_P: RealDiagonalVectorOperator = field(on_setattr=attr.setters.frozen)
    delta_F_prev_tilde: RealDiagonalVectorOperator = field(on_setattr=attr.setters.frozen)