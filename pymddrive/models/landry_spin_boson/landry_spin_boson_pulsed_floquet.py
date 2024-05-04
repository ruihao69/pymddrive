# %%
import attr
from attrs import define, field
import numpy as np

from pymddrive.my_types import RealVector, RealOperator, GenericOperator, GenericVectorOperator, RealVectorOperator
from pymddrive.models.nonadiabatic_hamiltonian import QuasiFloquetHamiltonianBase
from pymddrive.models.landry_spin_boson.math_utils import get_sigma_z, get_sigma_x, get_sigma_y, mu_Et, dmu_dR_Et
from pymddrive.pulses import PulseBase as Pulse 
from pymddrive.models.landry_spin_boson.landry_spin_boson import LandrySpinBoson

@define
class LandrySpinBosonPulsedFloquet(QuasiFloquetHamiltonianBase):
    """Dataclass for the Landry Spin Boson model with a pulse.

    Args:
        TD_HamiltonianBase: Base class for time-dependent Hamiltonians

    Fields:
        Omega_nuclear (float): The nuclear frequency
        M (float): The mass of the nuclear mode (in atomic units)
        V (float): The electronic coupling (in atomic units)
        Er (float): The reorganization energy (in atomic units)
        epsilon0 (float): Two-level system energy splitting 
        gamma (float): The friction coefficient for the nuclear mode
        kT (float): The thermal energy (in atomic units, 300 K)
        lambd (float): The reorganization energy (in atomic units)
        dim (int): The dimension of the system
        mu (float): The electric dipole moment (in atomic units)
        pulse (Pulse): The pulse that drives the system
    
    """
    Omega_nuclear: float = field(default=0.021375, on_setattr=attr.setters.frozen)
    M: float = field(default=1.0, on_setattr=attr.setters.frozen)
    V: float = field(default=0.00475, on_setattr=attr.setters.frozen)
    Er: float = field(default=0.00475, on_setattr=attr.setters.frozen)
    epsilon0: float = field(default=0.00475, on_setattr=attr.setters.frozen)
    gamma: float = field(default=0.04275, on_setattr=attr.setters.frozen)
    kT: float = field(default=0.00095, on_setattr=attr.setters.frozen)
    lambd: float = field(default=float('nan'), on_setattr=attr.setters.frozen)
    dim: int = field(default=2, init=False)
    mu: float = field(default=0.04, on_setattr=attr.setters.frozen)
    
    def __init__(
        self,
        ultrafast_pulse: Pulse,
        envelope_pulse: Pulse,
        NF: int,
        Omega_nuclear: float = 0.021375,
        M: float = 1.0,
        V: float = 0.00475,
        Er: float = 0.00475,
        epsilon0: float = 0.00475,
        gamma: float = 0.04275,
        kT: float = 0.00095,
        mu: float = 0.04,
    ) -> None:
        super().__init__(dim=2, ultrafast_pulse=ultrafast_pulse, envelope_pulse=envelope_pulse, NF=NF)
        object.__setattr__(self, "Omega_nuclear", Omega_nuclear)
        object.__setattr__(self, "M", M)
        object.__setattr__(self, "V", V)
        object.__setattr__(self, "Er", Er)
        object.__setattr__(self, "epsilon0", epsilon0)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "kT", kT)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "lambd", np.sqrt(Er * M * Omega_nuclear**2 / 2))
        
               
    def __repr__(self) -> str:
        return f"LandrySpinBoson(Omega_nuclear={self.Omega_nuclear}, M={self.M}, V={self.V}, Er={self.Er}, epsilon0={self.epsilon0}, gamma={self.gamma}, kT={self.kT})"
    
    def permanent_dipole(self, R: RealVector) -> RealOperator:
        """The permanent dipole operator. 
        Note now the dipole does not depend on the nuclear coordinate yet.
        
        Todo:
            Implement the nuclear coordinate dependence. e.g. 
                mu(R) = mu0 + mu1 * R

        Args:
            R (RealVector): The nuclear coordinate

        Returns:
            RealOperator: The permanent dipole operator
        """
        return self.mu * np.array([[0, 0], [0, 1]])
    
    def permanent_dipole_gradient(self, R: RealVector) -> RealVectorOperator:
        """The gradient of the permanent dipole operator.

        Args:
            R (RealVector): The nuclear coordinate

        Returns:
            RealVectorOperator: Zero matrix (since the dipole does not depend on the nuclear coordinate yet)
        """
        return np.zeros((self.dim, self.dim, R.shape[-1]))
    
    def transition_dipole(self, R: RealVector) -> RealOperator:
        """The transition dipole operator.

        Args:
            R (RealVector): The nuclear coordinate

        Returns:
            RealOperator: The transition dipole operator
        """
        return self.mu * np.array([[0, 1], [1, 0]])
    
    def transition_dipole_gradient(self, R: RealVector) -> RealVectorOperator:
        """The gradient of the transition dipole operator.

        Args:
            R (RealVector): The nuclear coordinate

        Returns:
            RealVectorOperator: Zero matrix (since the dipole does not depend on the nuclear coordinate yet)
        """
        return np.zeros((self.dim, self.dim, R.shape[-1]))
    
    def H0(self, R: RealVector) -> GenericOperator:
        # nuclear potential (on the diagonal)
        U = LandrySpinBoson.U(self.M, self.Omega_nuclear, R)
        
        # Two-level quantum system
        # --- sigmaz term
        Vz = LandrySpinBoson.Vz(self.lambd, self.epsilon0, R)
        
        # --- sigmax term
        Vx = LandrySpinBoson.Vx(self.V)
        
        return U + Vz + Vx
    
    def H1(self, t: float, R: RealVector) -> GenericOperator:
        # light-matter interaction term
        # mu = self.permanent_dipole(R)
        mu = self.transition_dipole(R)
        Et = self.envelope_pulse(t)
        return mu_Et(mu, Et)
    
    def dH0dR(
        self,
        R: RealVector,
    ) -> GenericOperator:
        dUdR = LandrySpinBoson.dUdR(self.M, self.Omega_nuclear, R)
        dVzdR = LandrySpinBoson.dVzdR(self.lambd)
        return dUdR + dVzdR 
    
    def dH1dR(
        self,
        t: float,
        R: RealVector,
    ) -> GenericVectorOperator:
        # return dmu_dR_Et(self.permanent_dipole_gradient(R), self.envelope_pulse(t))
        return dmu_dR_Et(self.transition_dipole_gradient(R), self.envelope_pulse(t))
    
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
    
    def get_dim_nuclear(self, ) -> int:
        return 1
    