# %% Use tully one to test the nonadiabatic hamiltonian abc class
import attr
import numpy as np
from attrs import define, field

from pymddrive.my_types import RealVector, RealOperator, GenericOperator, GenericVectorOperator, RealVectorOperator
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.models.landry_spin_boson.math_utils import get_sigma_z, get_sigma_x, get_sigma_y

@define
class LandrySpinBoson(HamiltonianBase):
    Omega_nuclear: float = field(default=0.021375, on_setattr=attr.setters.frozen)
    M: float = field(default=1.0, on_setattr=attr.setters.frozen)
    V: float = field(default=0.00475, on_setattr=attr.setters.frozen)
    Er: float = field(default=0.00475, on_setattr=attr.setters.frozen)
    epsilon0: float = field(default=0.00475, on_setattr=attr.setters.frozen)
    gamma: float = field(default=0.04275, on_setattr=attr.setters.frozen)
    kT: float = field(default=0.00095, on_setattr=attr.setters.frozen)
    lambd: float = field(init=False)
    dim: int = field(default=2, init=False)
    
    def __attrs_post_init__(self):
        self.lambd = np.sqrt(self.Er * self.M * self.Omega_nuclear**2 / 2)
        
    @staticmethod
    def U(
        M: float,
        Omega: float,
        R: RealVector,
    ) -> RealOperator:
        """The nuclear potential.

        Args:
            M (float): The mass of the nuclear mode
            Omega (float): The harmonic frequency of the nuclear mode
            R (float): The nuclear coordinate

        Returns:
            ArrayLike: The nuclear potential part of the Hamiltonian
        """
        return  np.sum(0.5 * M * Omega**2 * R**2) * np.eye(2)
    
    @staticmethod
    def dUdR(
        M: float,
        Omega: float,
        R: RealVector,
    ) -> RealVectorOperator: 
        """The nuclear potential.

        Args:
            M (float): The mass of the nuclear mode
            Omega (float): The harmonic frequency of the nuclear mode
            R (float): The nuclear coordinate

        Returns:
            ArrayLike: The nuclear potential part of the Hamiltonian
        """
        return np.sum(M * Omega**2 * R) * np.eye(2)[:, :, np.newaxis]
    
    @staticmethod
    def Vz(
        lambd: float,
        epsilon0: float,
        R: RealVector,
    ) -> RealOperator:
        return np.sum(lambd * R + 0.5 * epsilon0) * get_sigma_z()
    
    @staticmethod
    def dVzdR(
        lambd: float,
    ) -> GenericVectorOperator:
        return lambd * get_sigma_z()[:, :, np.newaxis]
    
    @staticmethod
    def Vx(
        V: float,
    ) -> GenericOperator:
        return V * get_sigma_x()
    
    def __repr__(self) -> str:
        return f"LandrySpinBoson(Omega_nuclear={self.Omega_nuclear}, M={self.M}, V={self.V}, Er={self.Er}, epsilon0={self.epsilon0}, gamma={self.gamma}, kT={self.kT})"
    
    def H(
        self,
        t: float,
        R: RealVector,
    ) -> GenericOperator:
        U = self.U(self.M, self.Omega_nuclear, R)
        Vz = self.Vz(self.lambd, self.epsilon0, R)
        return U + Vz + self.Vx(self.V)
    
    def dHdR(
        self,
        t: float,
        R: RealVector,
    ) -> GenericOperator:
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
    
    def get_friction(self, ) -> float:
        return self.gamma
    
    def get_kT(self, ) -> float:
        return self.kT
    
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
        H[:, :, ii] = model.H(0, np.array([rr]))
        dHdR[:, :, ii] = model.dHdR(0, np.array([rr]))[:, :, 0]
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
