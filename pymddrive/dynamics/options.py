from enum import Enum, unique

# use Enum class to define the dynamics options
@unique   
class BasisRepresentation(Enum):
    Diabatic = 'Diabatic'
    Adiabatic = 'Adiabatic'
    
@unique
class QunatumRepresentation(Enum):
    Wavefunction = 'Wavefunction'
    DensityMatrix = 'Density Matrix'
    
@unique
class NonadiabaticDynamicsMethods(Enum):
    EHRENFEST = 'Ehrenfest'
    FSSH = 'FSSH'
   
@unique 
class NumericalIntegrators(Enum):
    ############################################################################
    #                                                                          #
    # Fully coupled. (rho/psi) and (R, P) are treated on identical time grid.  #
    #                                                                          #
    ############################################################################
 
    ZVODE = 'zvode from scipy.integrate (default)'
    RK4 = 'Homemade runge-kutta 4th order'  
    ############################################################################
    #                                                                          #
    #     Slightly decoupled. (rho/psi) use a finer time grid than (R, P)      #
    #                                                                          #
    ############################################################################
    VVRK4 = 'velocity-verlet coupled to rk4'
    VVRK4_GPAW = 'velocity-verlet coupled to rk4 with GPAW'