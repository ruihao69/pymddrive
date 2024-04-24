# %%
import numpy as np
import scipy.constants as const

def mu_in_eA_to_Cm(mu: float) -> float:
    return mu * const.e * const.angstrom

def mu_in_eA_to_au(mu: float) -> float:
    return mu * const.angstrom / const.physical_constants['Bohr radius'][0] 

def mu_in_Debye_to_au(mu: float) -> float:
    DEBYE_IN_SI = 3.33564e-30
    return mu * DEBYE_IN_SI / const.physical_constants['Bohr radius'][0] / const.e

def intensity_from_field(E: float) -> float:
    return E**2 / 2 * const.c * const.epsilon_0

def field_from_intensity(I: float) -> float:
    return np.sqrt(2 * I / const.c / const.epsilon_0)

def field_in_SI_to_au(E: float) -> float:
    return E / const.physical_constants['atomic unit of electric field'][0]

def laser_indensity_in_TWcm2_to_electric_field(I_TWcm2: float) -> float:
    # First convert the intensity to SI units
    # ie. TW/cm^2 to W/m^2
    T_Wm2 = I_TWcm2 * 1e12 * 1e4
    
    # Then convert the intensity to electric field (in SI)
    E_in_SI = field_from_intensity(T_Wm2)
    
    # Finally convert the electric field to atomic units
    return field_in_SI_to_au(E_in_SI)
    

def main():
    mu_in_Debye = 0.1
    mu_in_au = mu_in_Debye_to_au(mu_in_Debye)
    print(f"{mu_in_au=}")
    
    I_in_TW_cm2 = 300.
    
    E_in_au = laser_indensity_in_TWcm2_to_electric_field(I_in_TW_cm2)
    print(f"{E_in_au=}")
    
    

# %%
if __name__ == "__main__":
    main()
# %%
