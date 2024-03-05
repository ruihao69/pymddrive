# %% the package
import numpy as np  

from pymddrive.pulses import PulseBase as Pulse
from pymddrive.pulses import MorletReal, Gaussian, UnitPulse
from pymddrive.models.tullyone.tullyone import TullyOne
from pymddrive.models.tullyone.tullyone_td_type1 import TullyOneTD_type1
from pymddrive.models.tullyone.tullyone_td_type2 import TullyOneTD_type2
from pymddrive.models.tullyone.tullyone_floquet_type1 import TullyOneFloquet_type1
from pymddrive.models.tullyone.tullyone_floquet_type2 import TullyOneFloquet_type2

from typing import Union
from numbers import Real
from enum import Enum, unique

@unique 
class TullyOnePulseTypes(Enum): 
    NO_PULSE = "NoPulse"
    ZEROPULSE = "ZeroPulse" # for debugging
    UNITPULSE = "UnitPulse" # for debugging
    PULSE_TYPE1 = "PulseType1"
    PULSE_TYPE2 = "PulseType2"
    PULSE_TYPE3 = "PulseType3"

@unique 
class TD_Methods(Enum):
    BRUTE_FORCE = "BruteForce"
    FLOQUET = "Floquet"
    
def get_tullyone(
    A: Real = 0.01, B: Real = 1.6, C: Real = 0.005, D: Real = 1.0, # Tully parameters
    t0: Union[Real, None] = None, Omega: Union[Real, None] = None, 
    tau: Union[Real, None] = None, # pulse parameters
    pulse_type: TullyOnePulseTypes = TullyOnePulseTypes.NO_PULSE,
    td_method: TD_Methods = TD_Methods.BRUTE_FORCE,
    NF: Union[int, None] = None
):
    if pulse_type == TullyOnePulseTypes.NO_PULSE:
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOne(A=A, B=B, C=C, D=D)
        else:
            raise ValueError(f"You are trying to get a floquet model for a time independent Hamiltonian.")
    else:
        if (t0 is None) or (Omega is None) or (tau is None):
            raise ValueError(f"You need to provide the pulse parameters t0, Omega, and tau for Time-dependent problems.")
        
    if td_method == TD_Methods.FLOQUET and NF is None:
        raise ValueError(f"You need to provide the number of Floquet replicas NF for Floquet models.")
    
    if pulse_type == TullyOnePulseTypes.PULSE_TYPE1:
        orig_pulse = MorletReal(A=C, t0=t0, tau=tau, Omega=Omega, phi=0)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=0, D=0, pulse=orig_pulse)
        elif td_method == TD_Methods.FLOQUET:
            floq_pulse = Gaussian.from_quasi_floquet_morlet_real(orig_pulse)
            return TullyOneFloquet_type1(A=A, B=B, C=0, D=0, orig_pulse=orig_pulse, floq_pulse=floq_pulse, NF=NF)
        else:
            raise ValueError(f"Invalid TD method: {td_method}")
        
    elif pulse_type == TullyOnePulseTypes.PULSE_TYPE2:
        orig_pulse = MorletReal(A=1.0, t0=t0, tau=tau, Omega=Omega, phi=0)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type2(A=A, B=B, C=C, D=D, pulse=orig_pulse)
        elif td_method == TD_Methods.FLOQUET:
            floq_pulse = Gaussian.from_quasi_floquet_morlet_real(orig_pulse)
            return TullyOneFloquet_type2(A=A, B=B, C=C, D=D, orig_pulse=orig_pulse, floq_pulse=floq_pulse, NF=NF)
        else:
            raise ValueError(f"Invalid TD method: {td_method}")
        
    elif pulse_type == TullyOnePulseTypes.PULSE_TYPE3:
        orig_pulse = MorletReal(A=C/2, t0=t0, tau=tau, Omega=Omega, phi=0)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=C/2, D=D, pulse=orig_pulse)
        elif td_method == TD_Methods.FLOQUET:
            floq_pulse = Gaussian.from_quasi_floquet_morlet_real(orig_pulse)
            return TullyOneFloquet_type1(A=A, B=B, C=C/2, D=D, orig_pulse=orig_pulse, floq_pulse=floq_pulse, NF=NF)
        else:
            raise ValueError(f"Invalid TD method: {td_method}")
    elif pulse_type == TullyOnePulseTypes.ZEROPULSE:
        orig_pulse = Pulse()
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=C, D=D, pulse=orig_pulse)
        else:
            raise ValueError(f"For the UnitPulse, you can only use the BruteForce method. But you are trying to use {td_method}.")
    elif pulse_type == TullyOnePulseTypes.UNITPULSE:
        orig_pulse = UnitPulse(A=C)
        if td_method == TD_Methods.BRUTE_FORCE:
            return TullyOneTD_type1(A=A, B=B, C=0, D=0, pulse=orig_pulse)
        else:
            raise ValueError(f"For the UnitPulse, you can only use the BruteForce method. But you are trying to use {td_method}.")
    else:
        raise ValueError(f"Invalid pulse type: {pulse_type}")
    
# %% testing/debugging code

def _evaluate_tullyone_hamiltonian(t, r, model):
    from pymddrive.models.nonadiabatic_hamiltonian import evaluate_hamiltonian, evaluate_nonadiabatic_couplings
    dim_elc = model.dim
    dim_cls = r.size
    E_out = np.zeros((dim_cls, dim_elc))
    F_out = np.zeros((dim_cls, dim_elc))
    for ii, rr in enumerate(r):
        _, dHdR, evals, evecs = evaluate_hamiltonian(t, rr, hamiltonian=model)
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        E_out[ii, :] = evals
        F_out[ii, :] = F
    return E_out, F_out

def _evaluate_tullyone_floquet_hamiltonian(t, r, model):
    from pymddrive.models.nonadiabatic_hamiltonian import evaluate_hamiltonian, evaluate_nonadiabatic_couplings
    dim_elc = model.dim
    NF = model.NF
    dim_F = dim_elc*(NF*2+1)
    E_out = np.zeros((len(r), dim_F))
    F_out = np.zeros((len(r), dim_F))
    for ii, rr in enumerate(r):
        _, dHdR, evals, evecs = evaluate_hamiltonian(t, rr, hamiltonian=model)
        d, F = evaluate_nonadiabatic_couplings(dHdR, evals, evecs)
        E_out[ii, :] = evals
        F_out[ii, :] = F
    return E_out, F_out

def _plot_tullyone_hamiltonian(r, E, F):
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use('science')
    
    fig = plt.figure(figsize=(3*2, 2), dpi=300)
    gs = fig.add_gridspec(1, 2)
    axs = gs.subplots().flatten()
    
    # plot the eigen energies 
    ax = axs[0]
    for ii in range(E.shape[1]):
        ax.plot(r, E[:, ii], label=f"E{ii}")
    ax.legend() 
    ax.set_xlabel("R")
    ax.set_ylabel("Eigen Energies")
    
    # plot the adiabatic forces
    ax = axs[1]
    for ii in range(F.shape[1]):
        ax.plot(r, F[:, ii], label=f"F{ii}")
    ax.legend()
    ax.set_xlabel("R")
    ax.set_ylabel("Adiabatic Forces")
    
    fig.tight_layout()
    plt.show()
    

def _test_tullyone():
    hamiltonian = TullyOne()
    r = np.linspace(-10, 10, 1000)
    E, F = _evaluate_tullyone_hamiltonian(0, r, hamiltonian)
    _plot_tullyone_hamiltonian(r, E, F)
    
def _test_tullyone_pulsed(pulse_type: TullyOnePulseTypes):
    Omega = 0.03
    tau = 100
    t0 = 0
    hamiltonian = get_tullyone(
        t0=t0, Omega=Omega, tau=tau,
        pulse_type=pulse_type,
        td_method=TD_Methods.BRUTE_FORCE
    )
    r = np.linspace(-10, 10, 1000)
    t = 0
    E, F = _evaluate_tullyone_hamiltonian(t, r, hamiltonian)
    _plot_tullyone_hamiltonian(r, E, F)
    
def _test_tullyone_floquet(pulse_type: TullyOnePulseTypes):
    Omega = 0.01
    tau = 100
    t0 = 0
    NF = 1
    hamiltonian = get_tullyone(
        t0=t0, Omega=Omega, tau=tau,
        pulse_type=pulse_type,
        td_method=TD_Methods.FLOQUET,
        NF=NF
    )
    r = np.linspace(-10, 10, 1000)
    t = 0
    E, F = _evaluate_tullyone_floquet_hamiltonian(t, r, hamiltonian)
    _plot_tullyone_hamiltonian(r, E, F)
        
    
# %% 
if __name__ == "__main__":
    # test the time-independent TullyOne model
    print("=====================================================")
    print("Testing the time-independent TullyOne model")
    _test_tullyone()
    print("=====================================================")
    # test the time-dependent TullyOne model with different pulse types
    print("=====================================================")
    print("Testing the TullyOne with PulseType1")
    _test_tullyone_pulsed(TullyOnePulseTypes.PULSE_TYPE1)
    print("=====================================================")
    print("=====================================================")
    print("Testing the TullyOne with PulseType2")
    _test_tullyone_pulsed(TullyOnePulseTypes.PULSE_TYPE2)
    print("=====================================================")
    print("=====================================================")
    print("Testing the TullyOne with PulseType3")
    _test_tullyone_pulsed(TullyOnePulseTypes.PULSE_TYPE3)
    print("=====================================================")
    
    # test the time-dependent Floquet TullyOne model with different pulse types 
    print("=====================================================")
    print("Testing the Floquet TullyOne with PulseType1")
    _test_tullyone_floquet(TullyOnePulseTypes.PULSE_TYPE1)
    print("=====================================================")
    print("=====================================================")
    print("Testing the Floquet TullyOne with PulseType2")
    _test_tullyone_floquet(TullyOnePulseTypes.PULSE_TYPE2)
    print("=====================================================")
    print("=====================================================")
    print("Testing the Floquet TullyOne with PulseType3")
    _test_tullyone_floquet(TullyOnePulseTypes.PULSE_TYPE3)
    print("=====================================================")
    
# %%
