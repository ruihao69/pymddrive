from enum import Enum, unique
 
@unique
class QuantumRepresentation(Enum):
    Wavefunction = 'WaveFunction'
    DensityMatrix = 'DensityMatrix'