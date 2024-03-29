from enum import Enum, unique 
 
@unique
class NonadiabaticDynamicsMethods(Enum):
    EHRENFEST = 'Ehrenfest'
    FSSH = 'FSSH'