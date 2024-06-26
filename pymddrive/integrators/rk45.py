# %% The package: pymddrive.integrators.rk4
import warnings
warnings.warn("The RK45(Tsit5) integrator has not been fully tested yet.")

import numpy as np

from typing import Callable, Union, Any
from numpy.typing import ArrayLike

# from pymddrive.integrators.rungekutta import 
from pymddrive.low_level.states import State
from pymddrive.integrators.state import get_state
from pymddrive.integrators.rungekutta import (
    tsit5_tableau, 
    evaluate_initial_dt,
    runge_kutta_step, 
    get_optimal_step_size
)

# Type short hand
type_time = float

class RungeKutta45:
    def __init__(
        self,
        derivative: Callable[[type_time, State], State],
        t0: type_time,
        y0: State,
        rtol: float,
        atol: float,
        first_step=None,
        safety: float=0.9, 
        ifactor: float=10.0, 
        dfactor: float=0.2,
        *args: Any,
        **kwargs: Any
    ) -> None:
        raise NotImplementedError("The RK45(Tsit5) integrator has not been fully tested yet.")
        self.derivative = derivative
        self.t0 = t0
        self.y0 = y0
        self.rtol = rtol
        self.atol = atol
        self.first_step = first_step
        self.safety = safety
        self.ifactor = ifactor
        self.dfactor = dfactor
        
        if self.first_step is None:
            first_step = evaluate_initial_dt(
                derivative, t0, y0, 4, rtol, atol
            )
        self.h = first_step
        
    def _adaptive_step(self, t0: type_time, y0: State):
        # The runge kutta steps
        y1, f1, y1_err, k = runge_kutta_step(
            self.derivative, t0, y0, self.h, tsit5_tableau
        )
        
        # error ratio
        _y0 = y0.flatten() 
        _y1 = y1.flatten()
        _y1_err = y1_err.flatten()
        
        error_tol = self.atol + self.rtol * np.maximum(np.abs(_y0), np.abs(_y1))
        mean_sq_error_ratio = np.mean((_y1_err / error_tol)**2)
        
        accept_step = mean_sq_error_ratio <= 1.0
        
        # update the rk45 state 
        t_next = t0 + self.h if accept_step else t0
        y_next = y1 if accept_step else y0
        self.h = get_optimal_step_size(
            self.h, mean_sq_error_ratio, self.safety, self.ifactor, self.dfactor
        )
        return t_next, y_next
        

# %% The temporary test code
def _debug_test():
    from pymddrive.integrators.state import get_state
    assert np.all(tsit5_tableau.a[0] == 0)
    assert tsit5_tableau.a[1][0] == 0.161
    assert tsit5_tableau.c[2] - tsit5_tableau.a[2, 1] == tsit5_tableau.a[2][0]
    assert tsit5_tableau.c[3] - tsit5_tableau.a[3, 1] - tsit5_tableau.a[3, 2] == tsit5_tableau.a[3][0]
    assert tsit5_tableau.c[4] - tsit5_tableau.a[4, 1] - tsit5_tableau.a[4, 2] - tsit5_tableau.a[4, 3] == tsit5_tableau.a[4][0]
    assert tsit5_tableau.c[5] - tsit5_tableau.a[5, 1] - tsit5_tableau.a[5, 2] - tsit5_tableau.a[5, 3] - tsit5_tableau.a[5, 4] == tsit5_tableau.a[5][0]
    
    def derivative(t: float, s: State):
        out = s.zeros_like()
        r, p, _ = s.get_variables()
        out.set_R(p * 0.01 / s.get_mass())  
        out.set_P(-r * 0.01)
        return out
    n_particle = 100
    R = np.random.normal(0, 1, n_particle)
    P = np.random.normal(0, 1, n_particle)
    mass = 1.0
    s = get_state(mass, R, P, None)
    
    N = 100000
    time = np.zeros(N)
    out_R = np.zeros((N, n_particle))
    out_P = np.zeros((N, n_particle))
    
    def benchmark(N: int):
        t = 0.0 
        n_particle = 100
        R = np.random.normal(0, 1, n_particle)
        P = np.random.normal(0, 1, n_particle)
        mass = 1.0
        s = get_state(mass, R, P, None)
        rk45 = RungeKutta45(
            derivative=derivative,
            t0=t,
            y0=s,
            atol=1e-5,
            rtol=1e-5,
            first_step=None
        )
        import timeit
        start = timeit.default_timer() 
        i = 0
        for _ in range(N):
            t, s = rk45._adaptive_step(t, s)
            out_R[i], out_P[i], _ = s.get_variables()
            time[i] = t
            i += 1
            
        time_elapsed = timeit.default_timer() - start
        print(f"The speed of the integrator: {N/time_elapsed} iteractions per second")
    
    benchmark(N)
         
    from matplotlib import pyplot as plt
    
    plt.plot(time, out_R[:, 0], label="r")
    plt.plot(time, out_P[:, 0], label="p")
    plt.title("Time series")
    plt.legend()
    
    plt.show()
    plt.figure(dpi=300)
    plt.plot(out_R[:, 0], out_P[:, 0])
    plt.plot(out_R[:, 1], out_P[:, 1])
    plt.plot(out_R[:, -1], out_P[:, -1])
    plt.xlabel("r")
    plt.ylabel("p")
    plt.title("Phase space")
    plt.show()
    
    E = np.nansum(out_R**2 + out_P**2, axis=-1)
    plt.plot(time, E-E[0])
    plt.yscale('symlog') 
    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.title("Energy conservation")

# %% The __main__ code    
if __name__ == "__main__":
    _debug_test() 


# %%
