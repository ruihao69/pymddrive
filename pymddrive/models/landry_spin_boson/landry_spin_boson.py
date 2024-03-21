# %% Use tully one to test the nonadiabatic hamiltonian abc class
import numpy as np
from numpy.typing import ArrayLike

from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.models.landry_spin_boson.math_utils import get_sigma_z, get_sigma_x, get_sigma_y

class LandrySpinBoson(HamiltonianBase):
    def __init__(
        self, 
        Omega_nuclear: float = 0.021375, # harmonic frequency of the nuclear mode
        M: float = 1.0, # mass of the nuclear mode
        V: float = 0.00475, # coupling strength
        Er: float = 0.00475, # reorganization energy
        epsilon0: float = 0.00475, # on-site energy
        gamma: float = 0.04275, # friction coefficient
        kT: float = 0.00095,  # thermal energy
    ) -> None:
        # Initialize the HamiltonianBase
        super().__init__(dim=2)
        
        # Save the parameters as attributes
        self.lambd: float = np.sqrt(Er * M * Omega_nuclear**2 / 2) # electronic - vibrational mode coupling
        self.M: float = M
        self.Omega_nuclear: float = Omega_nuclear
        self.V: float = V
        self.Er: float = Er
        self.epsilon0: float = epsilon0
        self.gamma: float = gamma
        self.kT: float = kT
        
    @staticmethod
    def U(
        M: float,
        Omega: float,
        R: float, 
    ) -> ArrayLike:
        """The nuclear potential.

        Args:
            M (float): The mass of the nuclear mode
            Omega (float): The harmonic frequency of the nuclear mode
            R (float): The nuclear coordinate

        Returns:
            ArrayLike: The nuclear potential part of the Hamiltonian
        """
        return 0.5 * M * Omega**2 * R**2 * np.eye(2)
    
    @staticmethod
    def dUdR(
        M: float,
        Omega: float,
        R: float, 
    ) -> ArrayLike:
        """The nuclear potential.

        Args:
            M (float): The mass of the nuclear mode
            Omega (float): The harmonic frequency of the nuclear mode
            R (float): The nuclear coordinate

        Returns:
            ArrayLike: The nuclear potential part of the Hamiltonian
        """
        return M * Omega**2 * R * np.eye(2)
    
    @staticmethod
    def Vz(
        lambd: float,
        epsilon0: float,
        R: float,
    ) -> ArrayLike:
        return (lambd * R + 0.5 * epsilon0) * get_sigma_z() 
    
    @staticmethod
    def dVzdR(
        lambd: float,
    ) -> ArrayLike:
        return lambd * get_sigma_z()
    
    @staticmethod
    def Vx(
        V: float,
    ) -> ArrayLike:
        return V * get_sigma_x()
    
    def frictional_force(
        self,
        P: float, 
    ) -> float:
        return -self.gamma * P
    
    def __repr__(self) -> str:
        return f"LandrySpinBoson(Omega_nuclear={self.Omega_nuclear}, M={self.M}, V={self.V}, Er={self.Er}, epsilon0={self.epsilon0}, gamma={self.gamma}, kT={self.kT})"
    
    def H(
        self,
        t: float,
        R: float,
    ) -> ArrayLike:
        U = self.U(self.M, self.Omega_nuclear, R)
        Vz = self.Vz(self.lambd, self.epsilon0, R)
        return U + Vz + self.Vx(self.V)
    
    def dHdR(
        self,
        t: float,
        R: float,
    ) -> ArrayLike:
        dUdR = self.dUdR(self.M, self.Omega_nuclear, R)
        dVzdR = self.dVzdR(self.lambd)
        return dUdR + dVzdR 
    
    @staticmethod
    def spin_boson_offset(
        M :float,
        Omega: float,   
        lambd: float
    ) -> float:
        return lambd / M / Omega**2
    
    def get_donor_R(self) -> float:
        return -self.spin_boson_offset(self.M, self.Omega_nuclear, self.lambd)
    
    def get_acceptor_R(self) -> float:
        return self.spin_boson_offset(self.M, self.Omega_nuclear, self.lambd)
    
# %% 
def _test_landry_spin_boson():
    import scipy.linalg as LA
    import matplotlib.pyplot as plt
    # Create an instance of the LandrySpinBoson model
    model = LandrySpinBoson()
    nsamples = 1000
    L = 15
    x0 = -2.5
    R = np.linspace(-L+x0, x0+L, 1000)
    H = np.zeros((2, 2, nsamples))
    dHdR = np.zeros_like(H)
    E = np.zeros((2, nsamples)) 
    d12 = np.zeros(nsamples)
    for ii, rr in enumerate(R):
        H[:, :, ii] = model.H(0, rr)
        dHdR[:, :, ii] = model.dHdR(0, rr)
        evals, evecs = LA.eigh(H[:, :, ii])
        E[:, ii] = evals
        V = evecs.conjugate().T @ dHdR[:, :, ii] @ evecs
        d12[ii] = V[0, 1]
        
        
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(R, E[0, :], label="Ground state")
    ax.plot(R, E[1, :], label="Excited state")
    ax.plot(R, H[0, 0, :], label="Donor: V11")
    ax.plot(R, H[1, 1, :], label="Acceptor: V22")
    ax.axvline(model.get_donor_R(), c='k', ls='--', label='Donor Eq.')
    ax.axvline(model.get_acceptor_R(), c='k', ls='-.', label='Acceptor Eq.')
    ax.set_xlabel("R")
    ax.set_ylabel("Energy")
    ax.legend()
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(R, d12)
    ax.set_xlabel("R")
    ax.set_ylabel("Nonadiabatic coupling")
    
# %% 
if __name__ == "__main__":
    _test_landry_spin_boson() 
        
# %%
