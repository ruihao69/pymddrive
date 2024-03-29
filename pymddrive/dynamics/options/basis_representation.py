from enum import Enum, unique

# use Enum class to define the dynamics options
@unique
class BasisRepresentation(Enum):
    Diabatic = 'Diabatic'
    Adiabatic = 'Adiabatic'