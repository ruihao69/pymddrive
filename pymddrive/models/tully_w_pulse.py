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
    
    def __call__(
        self, 
        x: Union[float, ArrayLike],
        t: float,
        use_numerical_eigen: bool = False, 
        *args: Any,
        **kwargs: Any,
    ):
        if isinstance(x, float):
            H = self.get_H(x, t)
            evals, evecs = self.diagonalize(H, use_numerical_eigen)
            dHdx = self.get_dHdx(x, t)
            d, F = NonadiabaticHamiltonian.get_nonadiabatic_couplings(dHdx, evecs, evals)
            d12 = d[0, 1]
            
            return H, evals, evecs, d12, F
            
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("The batch mode is not implemented at this time.")
        else:
            raise ValueError("The input x should be either float or numpy.ndarray.")
            
    @staticmethod
    def V12(
        t: float, 
        pulse: Pulse,
    ):
        return pulse(t) 
    
    def get_H(
        self, 
        x: Union[float, ArrayLike],
        t: float
    ):
        if isinstance(x, float):
            V11 = TullyOne.V11(x, self.A, self.B)
            V22 = -V11
            V12 = V21 = TullyOnePulseOne.V12(t, self.pulse)
            return np.array(
                [[V11, V12], 
                 [V21, V22]]
            )
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("The batch mode is not implemented at this time.")
            
    def get_dHdx(self, x: float, *args: Any, **kwargs: Any):
        dV11 = self.A * self.B * np.exp(-np.abs(x) * self.B)
        return np.array(
            [[dV11, 0.0], 
             [0.0, -dV11]]
        )
        
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
    
    def __call__(
        self, 
        x: Union[float, ArrayLike],
        t: float,
        use_numerical_eigen: bool = False, 
        *args: Any,
        **kwargs: Any,
    ):
        if isinstance(x, float):
            H = self.get_H(x, t)
            evals, evecs = self.diagonalize(H, use_numerical_eigen)
            dHdx = self.get_dHdx(x, t)
            d, F = NonadiabaticHamiltonian.get_nonadiabatic_couplings(dHdx, evecs, evals)
            d12 = d[0, 1]
            
            return H, evals, evecs, d12, F
            
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("The batch mode is not implemented at this time.")
        else:
            raise ValueError("The input x should be either float or numpy.ndarray.")
    
    @staticmethod
    def V12(
        x: float,
        t: float, 
        C: float,
        D: float,
        pulse: Pulse,
    ):
        return pulse(t) * TullyOne.V12(x, C, D)
    
    def get_H(
        self, 
        x: Union[float, ArrayLike],
        t: float
    ):
        if isinstance(x, float):
            # V11 = TullyOnePulseOne.V11(x, self.A, self.B)
            V11 = TullyOne.V11(x, self.A, self.B)
            V22 = -V11
            V12 = V21 = TullyOnePulseTwo.V12(x, t, self.C, self.D, self.pulse) 
            return np.array(
                [[V11, V12], 
                 [V21, V22]]
            )
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("The batch mode is not implemented at this time.")
            
    def get_dHdx(self, x: float, t: float, *args: Any, **kwargs: Any):
        dV11 = self.A * self.B * np.exp(-np.abs(x) * self.B)
        dV12 = -2 * self.C * self.D * x * np.exp(-self.D * x**2) * self.pulse(t)
        return np.array(
            [[dV11, dV12], 
             [dV12, -dV11]]
        )
        
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
    
    def __call__(
        self, 
        x: Union[float, ArrayLike],
        t: float,
        use_numerical_eigen: bool = False, 
        *args: Any,
        **kwargs: Any,
    ):
        if isinstance(x, float):
            H = self.get_H(x, t)
            evals, evecs = self.diagonalize(H, use_numerical_eigen)
            dHdx = self.get_dHdx(x, t)
            d, F = NonadiabaticHamiltonian.get_nonadiabatic_couplings(dHdx, evecs, evals)
            d12 = d[0, 1]
            
            return H, evals, evecs, d12, F
            
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("The batch mode is not implemented at this time.")
        else:
            raise ValueError("The input x should be either float or numpy.ndarray.")
    
    @staticmethod
    def V12(
        x: float,
        t: float, 
        C: float,
        D: float,
        pulse: Pulse,
    ):
        return pulse(t) + TullyOne.V12(x, C, D)
    
    def get_H(
        self, 
        x: Union[float, ArrayLike],
        t: float
    ):
        if isinstance(x, float):
            # V11 = TullyOnePulseOne.V11(x, self.A, self.B)
            V11 = TullyOne.V11(x, self.A, self.B)
            V22 = -V11
            V12 = V21 = TullyOnePulseThree.V12(x, t, self.C, self.D, self.pulse) 
            return np.array(
                [[V11, V12], 
                 [V21, V22]]
            )
        elif isinstance(x, np.ndarray):
            raise NotImplementedError("The batch mode is not implemented at this time.")
            
    def get_dHdx(self, x: float, t: float, *args: Any, **kwargs: Any):
        dV11 = self.A * self.B * np.exp(-np.abs(x) * self.B)
        dV12 = -2 * self.C * self.D * x * np.exp(-self.D * x**2) * self.pulse(t)
        return np.array(
            [[dV11, dV12], 
             [dV12, -dV11]]
        ) 
            
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
            H, evals, evecs, d12[ii, jj], F = t1p1(xx, tt)
            Eg[ii, jj] = evals[0]
            Ee[ii, jj] = evals[1]
            Fg[ii, jj] = F[0]
            Fe[ii, jj] = F[1]    
            
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
            H, evals, evecs, d12[ii, jj], F = t1p2(xx, tt)
            Eg[ii, jj] = evals[0]
            Ee[ii, jj] = evals[1]
            Fg[ii, jj] = F[0]
            Fe[ii, jj] = F[1]
            
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
            H, evals, evecs, d12[ii, jj], F = t1p3(xx, tt)
            Eg[ii, jj] = evals[0]
            Ee[ii, jj] = evals[1]
            Fg[ii, jj] = F[0]
            Fe[ii, jj] = F[1]
            
    water_fall_plot(x, t, np.array([Eg, Ee]), xlabel="x (a.u.)", title="Pulse #3: energies", scale1=2000, scale2=10, inverse_fill=True)
    water_fall_plot(x, t, np.array([Fg, Fe]), xlabel="x (a.u.)", title="Pulse #3: forces", scale1=2000, scale2=10)
    water_fall_plot(x, t, np.array([d12]), xlabel="x (a.u.)", title="Pulse #3: d12", scale1=5, scale2=10, xlim=(-4, 4))
    
# %% test fro 
