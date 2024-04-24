# %%
import numpy as np
import scipy.sparse as sps

from pymddrive.my_types import RealVector
from pymddrive.models.landry_spin_boson import get_landry_spin_boson
from pymddrive.models.nonadiabatic_hamiltonian import HamiltonianBase
from pymddrive.integrators.state import get_state
from pymddrive.dynamics.options import NonadiabaticDynamicsMethods, BasisRepresentation, NumericalIntegrators
from pymddrive.dynamics.get_dynamics import get_dynamics
from pymddrive.dynamics.run import run_ensemble
from landry_utils import sample_boltzmann
from spin_boson_postprocess import post_process

import os
from typing import Optional



def stop_condition(t, s):
    return t > 5000

def run_landry_spin_boson(
    R0: RealVector,
    P0: RealVector,
    init_state: int,
    hamiltonian: HamiltonianBase,
    mass: float=1.0,
    dt: float=0.1,
    NF: Optional[int]=None,
    data_dir: str='./',
    filename: str="fssh.nc",
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.FSSH,
    basis_rep: BasisRepresentation = BasisRepresentation.ADIABATIC,
    integrator: NumericalIntegrators = NumericalIntegrators.RK4, 
) -> None:
    assert R0.shape == P0.shape
    n_ensemble = R0.shape[0]
    
    def get_initial_rho(NF: Optional[int]=None):
        rho0 = np.zeros((2, 2), dtype=np.complex128)
        rho0[init_state, init_state] = 1
        if NF is None:
            return rho0
        else:
            zeros_like = np.zeros_like(rho0)
            data = [zeros_like] * NF + [rho0] + [zeros_like] * NF
            return sps.block_diag(data).toarray()
            
    
    # initial states 
    t0 = 0.0
    rho0 = [get_initial_rho(NF=NF) for _ in range(n_ensemble)]
    if basis_rep == BasisRepresentation.ADIABATIC:
        for ii in range(n_ensemble):
            H = hamiltonian.H(t0, R0[ii])
            evals, evecs = np.linalg.eigh(H)
            rho0_ii = evecs.T.conjugate() @ rho0[ii] @ evecs
            rho0[ii] = rho0_ii
            
    s0_list = [get_state(mass=mass, R=R0[ii], P=P0[ii], rho_or_psi=rho0[ii]) for ii in range(n_ensemble)]
    
    # get the dynamics ensemble
    dynamics_list = []
    for ii in range(n_ensemble):
        dynamics = get_dynamics(
            t0=t0,
            s0=s0_list[ii],
            dt=dt,
            hamiltonian=hamiltonian,
            method=solver,
            dynamics_basis=basis_rep
        )
        dynamics_list.append(dynamics)
        
    # prepare the data directory
    filename = os.path.join(data_dir, filename)
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        
    run_ensemble(
        dynamics_list=dynamics_list,
        break_condition=stop_condition,
        filename=filename,
        numerical_integrator=integrator,
        save_every=10,
        mode='append'
    )
    
    post_process(data_dir)

def get_filename(solver: NonadiabaticDynamicsMethods) -> str:
    return f"{solver.name.lower()}.nc"
    

def main(
    project_prefix: str,
    ntrajs: int,
    E0: Optional[float]=None,
    Omega: Optional[float]=None,
    phi: Optional[float]=None,
    N: Optional[int]=None,
    pulse_type: str='no_pulse',
    init_state: int = 0,
    basis_rep: BasisRepresentation = BasisRepresentation.DIABATIC,
    solver: NonadiabaticDynamicsMethods = NonadiabaticDynamicsMethods.EHRENFEST,
    integrator: NumericalIntegrators = NumericalIntegrators.RK4,
    NF: Optional[int]=None
):    
    # get the Landry Spin Boson model with a sine-squared pulse
    hamiltonian = get_landry_spin_boson(E0=E0, Omega=Omega, N=N, phi=phi, pulse_type=pulse_type, NF=NF)
    filename = get_filename(solver)
    
    # sample the initial conditions
    R_eq = hamiltonian.get_donor_R() if init_state == 0 else hamiltonian.get_acceptor_R()
    R0, P0 = sample_boltzmann(n_samples=ntrajs, kT=hamiltonian.get_kT(), Omega=hamiltonian.Omega_nuclear, mass=1.0, R_eq=R_eq)
    
    # run the dynamics
    run_landry_spin_boson(
        R0=R0,
        P0=P0,
        init_state=init_state,
        hamiltonian=hamiltonian,
        data_dir=project_prefix,
        filename=filename,
        solver=solver,
        basis_rep=basis_rep,
        integrator=integrator,
        NF=NF
    )
        
        
    
    
# %%
if __name__ == "__main__":
    ntrajs = 1
    
    def estimate_NF(A: float, Omega: float, tol: float=1e-6) -> int:
        from scipy.special import jv
        # n = math.ceil(A / Omega)
        n = A / Omega
        NF = 1
        while True:
            val = abs(jv(NF, n))
            if val < tol:
                break
            NF += 1
        return NF
    
    # parameters for the laser pulse
    dipole = 0.09   # dipole moment in atomic units
    E0 = 0.0925     # 300 TW/cm^2 laser intensity to E-field amplitude in atomic units
    Omega = 0.05696 # 800 nm laser wavelength to carrier frequency in atomic units
    A = dipole * E0
    phi = 0.0       # laser carrier phase
    N = 8           # Sine square envelope pulse with 8 cycles
    pulse_type = 'sine_squared_pulse'
    
    # project_prefix = f"data_ehrenfest_diabatic_E0_{E0}-Omega_{Omega}-N_{N}-phi_{phi}"
    # main(project_prefix=project_prefix, ntrajs=ntrajs, E0=E0, Omega=Omega, phi=phi, N=N, pulse_type=pulse_type, solver=NonadiabaticDynamicsMethods.EHRENFEST, basis_rep=BasisRepresentation.DIABATIC)
    
    project_prefix = f"data_floquet_fssh_E0_{E0}-Omega_{Omega}-N_{N}-phi_{phi}"
    NF = estimate_NF(A, Omega)  
    main(project_prefix=project_prefix, ntrajs=ntrajs, E0=E0, Omega=Omega, phi=phi, N=N, pulse_type=pulse_type, solver=NonadiabaticDynamicsMethods.FSSH, basis_rep=BasisRepresentation.ADIABATIC, NF=NF)
    
    # project_prefix = f"data_ehrenfest_diabatic"
    # pulse_type = 'no_pulse'
    # main(project_prefix=project_prefix, ntrajs=ntrajs, pulse_type=pulse_type, solver=NonadiabaticDynamicsMethods.EHRENFEST, basis_rep=BasisRepresentation.DIABATIC)
# %%
