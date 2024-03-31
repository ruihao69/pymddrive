from enum import Enum, unique, auto
 
@unique
class QuantumRepresentation(Enum):
    WAVEFUNCTION = auto()
    DENSITY_MATRIX = auto()