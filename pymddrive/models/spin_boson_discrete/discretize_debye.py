# %%
import numpy as np

from pymddrive.my_types import RealVector

from typing import Tuple

def discretize_Debye_bath(
    lambd: float, 
    Omega: float, 
    A: int
) -> Tuple[RealVector, RealVector]:
    """
    J(\omega) = 2 \lambda  \frac{\omega\Omega}{\omega^2 + \Omega^2}
    """
    
    alpha_list = np.arange(1, A+1)
    omega_alpha = Omega * np.tan((alpha_list - 0.5) / (2 * A) * np.pi)
    g_alpha = omega_alpha * np.sqrt(2 * lambd / A) 
    return omega_alpha, g_alpha

def discretize_Debye_bath_liu(
    lambd: float,
    Omega: float,   
    A: int  
) -> Tuple[RealVector, RealVector]:
    """
    J(\omega) = 2 \lambda  \frac{\omega\Omega}{\omega^2 + \Omega^2}
    """
    alpha_list = np.arange(1, A+1)
    omega_alpha = Omega * np.tan(0.5*np.pi*(1 - alpha_list / (A + 1)))
    g_alpha = omega_alpha * np.sqrt(2 * lambd / (A + 1)) 
    argsort = np.argsort(omega_alpha)
    return omega_alpha[argsort], g_alpha[argsort]


def testing():
    Omega = 0.1
    lambd = 1.0
    A:int = 100
    
    omega_alpha, g_alpha = discretize_Debye_bath(lambd, Omega, A)
    # omega_alpha, g_alpha = discretize_Debye_bath_liu(lambd, Omega, A)
    
    def plot_debye_bath():
        
        import matplotlib.pyplot as plt
        Jw = lambda w: 2 * lambd * Omega * w / (w**2 + Omega**2)
        
        w = np.linspace(0.0, 10.0, 1000)
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        ax.plot(w, Jw(w), label="Continuum")
        g_alpha_mid = (g_alpha[1:] + g_alpha[:-1]) / 2 
        omega_alpha_mid = (omega_alpha[1:] + omega_alpha[:-1]) / 2
        jw_descrete = g_alpha_mid**2 / omega_alpha_mid * np.pi / 2  / (omega_alpha[1:] - omega_alpha[:-1])  
        #ax.plot(omega_alpha, g_alpha**2/omega_alpha*np.pi/2, 'o', label="Discrete")
        ax.plot(omega_alpha_mid, jw_descrete, 'o', label="Discrete")
        plt.show()
        
    plot_debye_bath()
    
    def test_reorganization_energy():
        reorg = 0.5*np.sum(g_alpha**2 / omega_alpha**2)
        g_alpha_dimless = g_alpha / np.sqrt(omega_alpha)
        reorg_alt = np.dot(g_alpha_dimless**2, omega_alpha) / 4
        print(f"The reorganization energy is: {reorg}")
        print(f"The reorganization energy is: {reorg_alt}")
        print(f"The expected reorganization energy is: {lambd}")
        
    test_reorganization_energy()
        
    
# %%
if __name__ == "__main__":
    testing()
    
    
    
# %%
