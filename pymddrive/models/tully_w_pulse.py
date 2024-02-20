# %% The package
import numpy as np
import scipy.linalg as LA
from pymddrive.models.scatter import NonadiabaticHamiltonian
from pymddrive.models.tully import TullyOne

from numpy.typing import ArrayLike
from typing import Any, Union

from pymddrive.models.scatter import NonadiabaticHamiltonian
from pymddrive.pulses.pulses import Pulse

""" Pulse #1: position independent, real, Morlet pulse. """
class TullyOnePulseOne(TullyOne):
    def __init__(
        self,
        A: float = 0.01,
        B: float = 1.6,
        pulse: Pulse = Pulse(),
        
    ) -> None:
        self.A = A 
        self.B = B
        self.pulse = pulse
        
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOnePulseOne(A={self.A}, B={self.B}, pulse={self.pulse})"
    
    @staticmethod
    def V12(
        t: float, 
        r: Union[float, ArrayLike],
        pulse: Pulse,
    ):
        return pulse(t) * np.ones_like(r)
    
    def get_H(
        self, 
        t: float,
        r: Union[float, ArrayLike],
    ):
        V11 = TullyOne.V11(r, self.A, self.B)
        V12 = TullyOnePulseOne.V12(t, r, self.pulse)
        return TullyOne._H(r, V11, V12)
            
    def get_dHdr(self, t: float, r: Union[float, ArrayLike]):
        _ = t
        dV11 = self.A * self.B * np.exp(-np.abs(r) * self.B)
        return TullyOne._dHdr(dV11, np.zeros_like(dV11))
                    
""" Pulse #2: position dependent, real, Morlet pulse. """
class TullyOnePulseTwo(TullyOne):
    def __init__(
        self,
        A: float = 0.01,
        B: float = 1.6,
        C: float = 0.005,
        D: float = 1.0,
        pulse: Pulse = Pulse(),
        
    ) -> None:
        self.A = A 
        self.B = B
        self.C = C
        self.D = D
        self.pulse = pulse
        
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOnePulseTwo(A={self.A}, B={self.B}, C={self.C}, D={self.D}, pulse={self.pulse})"
    
    @staticmethod
    def V12(
        t: float, 
        r: Union[float, ArrayLike],
        C: float,
        D: float,
        pulse: Pulse,
    ):
        return pulse(t) * TullyOne.V12(r, C, D)
    
    def get_H(self, t: float, r: Union[float, ArrayLike]):
        V11 = TullyOne.V11(r, self.A, self.B)
        V12 = TullyOnePulseTwo.V12(t, r, self.C, self.D, self.pulse)
        return TullyOne._H(r, V11, V12)
    
    def get_dHdr(self, t: float, r: Union[float, ArrayLike]):
        dV11 = self.A * self.B * np.exp(-np.abs(r) * self.B)
        dV12 = -2 * self.C * self.D * r * np.exp(-self.D * r**2) * self.pulse(t)
        return TullyOne._dHdr(dV11, dV12)
        
""" Pulse #3: Morlet pulse superposition over a time independent coupling. """
class TullyOnePulseThree(TullyOne):
    def __init__(
        self,
        A: float = 0.01,
        B: float = 1.6,
        C: float = 0.005,
        D: float = 1.0,
        pulse: Pulse = Pulse(),
        
    ) -> None:
        self.A = A 
        self.B = B
        self.C = C
        self.D = D
        self.pulse = pulse
        
    def __repr__(self) -> str:
        return f"Nonadiabatic Hamiltonian TullyOnePulseTwo(A={self.A}, B={self.B}, C={self.C}, D={self.D}, pulse={self.pulse})"
    
    @staticmethod
    def V12(
        t: float, 
        r: float,
        C: float,
        D: float,
        pulse: Pulse,
    ):
        return pulse(t) + TullyOne.V12(r, C, D)
    
    def get_H(
        self, 
        t: float,
        r: Union[float, ArrayLike],
    ):
        V11 = TullyOne.V11(r, self.A, self.B)
        V12 = TullyOnePulseThree.V12(t, r, self.C, self.D, self.pulse)
        return TullyOne._H(r, V11, V12)
            
    def get_dHdr(self, t: float, r: Union[float, ArrayLike]):
        dV11 = self.A * self.B * np.exp(-np.abs(r) * self.B)
        dV12 = -2 * self.C * self.D * r * np.exp(-self.D * r**2) * self.pulse(t)
        return TullyOne._dHdr(dV11, dV12)
            
# %% Temprarory test code

if __name__ == "__main__":
    from pymddrive.pulses.morlet import MorletReal
    C = 0.005
    p = MorletReal(A=C, t0=4, tau=1, Omega=10, phi=0)
    t1p1 = TullyOnePulseOne(A=0.01, B=1.6, pulse=p)
    
    npossamples = 1000 
    nsnapshot = 10
    x = np.linspace(-5, 5, npossamples)
    t = np.linspace(0, 7.5, nsnapshot)
    
    Eg = np.zeros((nsnapshot, npossamples))
    Ee = np.zeros((nsnapshot, npossamples))
    Fg = np.zeros((nsnapshot, npossamples))
    Fe = np.zeros((nsnapshot, npossamples))
    d12 = np.zeros((nsnapshot, npossamples))
    
    for ii, tt in enumerate(t):
        for jj, xx in enumerate(x):
            H, evals, evecs, d, F = t1p1(xx, tt)
            Eg[ii, jj] = evals[0]
            Ee[ii, jj] = evals[1]
            Fg[ii, jj] = F[0]
            Fe[ii, jj] = F[1]    
            d12[ii, jj] = d[0, 1]
            
    import matplotlib.pyplot as plt
    
    # water fall plot: 1. Eigen energies
    # fig1 = plt.figure(dpi=200)
    
    # print(np.array([Eg, Ee]).shape)
    
    def water_fall_plot(x, t, dat, xlabel=None, title=None, scale1=1, scale2=1, xlim=None, inverse_fill=False, *args, **kwargs, ):
        fig = plt.figure(dpi=300, figsize=(3, 5))
        ax = fig.add_subplot(111)
        for side in ['right', 'top', 'left']:
            ax.spines[side].set_visible(False)

        portiony = (t.max()-t.min())/scale1
        if xlim is None:
            portionx = (x.max()-x.min())/scale2
        else:
            portionx = (xlim[1]-xlim[0])/scale2
        for i in np.flip(range(nsnapshot)):
            yi = dat[:, i, :] + i*portiony
            # print(yi.shape)
            c_list = ['b', 'r']
            if inverse_fill:
                _dE = np.abs(yi[0][0] - yi[1][0])
                dE = np.array([_dE, -_dE]) / 2
            else:
                dE = np.array([0, 0])
                
            for ll, y in enumerate(yi):
                # ax.plot(x+portionx*i, y, c=c_list[ll])
                ax.fill_between(x+portionx*i, y, i*portiony-dE[ll], color=c_list[ll], alpha=0.5)
            # ax.plot(x, yi[0], c='b')
            # ax.plot(x, yi[1], c='r')
            ax.text(x.min()+portionx*i, i*portiony, f"time: {t[i]:.2f}", fontsize=5, color='k', ha='right')

        # ax.legend()
        ax.set_yticks([])
        ax.yaxis.set_ticks_position('none')
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        
    water_fall_plot(x, t, np.array([Eg, Ee]), xlabel="x (a.u.)", title="Pulse #1: energies", scale1=2000, scale2=10, inverse_fill=True)
    water_fall_plot(x, t, np.array([Fg, Fe]), xlabel="x (a.u.)", title="Pulse #1: forces", scale1=2000, scale2=10)
    water_fall_plot(x, t, np.array([d12]), xlabel="x (a.u.)", title="Pulse #1: d12", scale1=5, scale2=10, xlim=(-4, 4))
    
    p = MorletReal(A=1.0, t0=4, tau=1, Omega=10, phi=0)
    t1p2 = TullyOnePulseTwo(A=0.01, B=1.6, pulse=p)
    
    npossamples = 1000 
    nsnapshot = 10
    x = np.linspace(-5, 5, npossamples)
    t = np.linspace(0, 7.5, nsnapshot)
    
    Eg = np.zeros((nsnapshot, npossamples))
    Ee = np.zeros((nsnapshot, npossamples))
    Fg = np.zeros((nsnapshot, npossamples))
    Fe = np.zeros((nsnapshot, npossamples))
    d12 = np.zeros((nsnapshot, npossamples))
    
    for ii, tt in enumerate(t):
        for jj, xx in enumerate(x):
            H, evals, evecs, d, F = t1p2(xx, tt)
            Eg[ii, jj] = evals[0]
            Ee[ii, jj] = evals[1]
            Fg[ii, jj] = F[0]
            Fe[ii, jj] = F[1]
            d12[ii, jj] = d[0, 1]
            
    water_fall_plot(x, t, np.array([Eg, Ee]), xlabel="x (a.u.)", title="Pulse #2: energies", scale1=2000, scale2=10, inverse_fill=True)
    water_fall_plot(x, t, np.array([Fg, Fe]), xlabel="x (a.u.)", title="Pulse #2: forces", scale1=2000, scale2=10)
    water_fall_plot(x, t, np.array([d12]), xlabel="x (a.u.)", title="Pulse #2: d12", scale1=5, scale2=10, xlim=(-4, 4))
            
    p = MorletReal(A=C, t0=4, tau=1, Omega=10, phi=0)
    t1p3 = TullyOnePulseThree(A=0.01, B=1.6, pulse=p)
    
    npossamples = 1000 
    nsnapshot = 10
    x = np.linspace(-5, 5, npossamples)
    t = np.linspace(0, 7.5, nsnapshot)
    
    Eg = np.zeros((nsnapshot, npossamples))
    Ee = np.zeros((nsnapshot, npossamples))
    Fg = np.zeros((nsnapshot, npossamples))
    Fe = np.zeros((nsnapshot, npossamples))
    d12 = np.zeros((nsnapshot, npossamples))
    
    for ii, tt in enumerate(t):
        for jj, xx in enumerate(x):
            H, evals, evecs, d, F = t1p3(xx, tt)
            Eg[ii, jj] = evals[0]
            Ee[ii, jj] = evals[1]
            Fg[ii, jj] = F[0]
            Fe[ii, jj] = F[1]
            d12[ii, jj] = d[0, 1]
            
    water_fall_plot(x, t, np.array([Eg, Ee]), xlabel="x (a.u.)", title="Pulse #3: energies", scale1=2000, scale2=10, inverse_fill=True)
    water_fall_plot(x, t, np.array([Fg, Fe]), xlabel="x (a.u.)", title="Pulse #3: forces", scale1=2000, scale2=10)
    water_fall_plot(x, t, np.array([d12]), xlabel="x (a.u.)", title="Pulse #3: d12", scale1=5, scale2=10, xlim=(-4, 4))
    
# %% test fro 
