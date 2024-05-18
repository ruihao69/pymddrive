from enum import Enum, unique 
 
@unique
class NonadiabaticDynamicsMethods(Enum):
    EHRENFEST = 'Ehrenfest'
    FSSH = 'FSSH'
    AFSSH = 'A-FSSH'
    COMPLEX_FSSH = 'Complex FSSH (Miao2019JCP)'