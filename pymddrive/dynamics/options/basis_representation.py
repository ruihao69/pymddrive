from enum import Enum, unique, auto

# use Enum class to define the dynamics options
@unique
class BasisRepresentation(Enum):
    DIABATIC = auto()
    ADIABATIC = auto()