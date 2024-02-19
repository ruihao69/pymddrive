# %% The package: pymddrive.integrators.rk4
import dataclasses

import numpy as np

from typing import Callable, Union, Any
from numpy.typing import ArrayLike

# from pymddrive.integrators.rungekutta import 
from pymddrive.integrators.state import State, zeros_like
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
if __name__ == "__main__":
    assert np.all(tsit5_tableau.a[0] == 0)
    assert tsit5_tableau.a[1][0] == 0.161
    assert tsit5_tableau.c[2] - tsit5_tableau.a[2, 1] == tsit5_tableau.a[2][0]
    assert tsit5_tableau.c[3] - tsit5_tableau.a[3, 1] - tsit5_tableau.a[3, 2] == tsit5_tableau.a[3][0]
    assert tsit5_tableau.c[4] - tsit5_tableau.a[4, 1] - tsit5_tableau.a[4, 2] - tsit5_tableau.a[4, 3] == tsit5_tableau.a[4][0]
    assert tsit5_tableau.c[5] - tsit5_tableau.a[5, 1] - tsit5_tableau.a[5, 2] - tsit5_tableau.a[5, 3] - tsit5_tableau.a[5, 4] == tsit5_tableau.a[5][0]
    
    def derivative(t: float, s: State):
        out = zeros_like(s)
        r, p, _ = s.get_variables()
        kr, kp, _ = out.get_variables()
        # print(r.shape)
        kr[:] = p
        kp[:] = -r
                
        return 0.01*out
        #return out
    t = 0.0 
    rho = np.array([[0.5, 0], [0, 0.5]])
    n_particle = 100
    s = State(r=np.random.normal(0, 1, n_particle), p=np.random.normal(0, 1, n_particle), rho=None)
    # t = 0.0 
    # rho = np.array([[0.5, 0], [0, 0.5]])
    # s = State(r=-10.0, p=2.5, rho=None)
    # print(s.dtype)
    # print(s.data.shape)
    # print(s.data['R'].shape)
    # print(s.data['P'].shape)
    # print(s.data['rho'].shape)
    
    
    N = 100000
    time = np.zeros(N)
    out = np.zeros(N, dtype=s.data.dtype)
    # r_out = np.zeros(N)
    # p_out = np.zeros(N)
    # rho_out = np.zeros((N, 2, 2))
    
    def benchmark(N: int):
        t = 0.0 
        tf = N * 1.0
        rho = np.array([[0.5, 0], [0, 0.5]])
        n_particle = 100
        s = State(r=np.random.normal(0, 1, n_particle), p=np.random.normal(0, 1, n_particle), rho=None)
        rk45 = RungeKutta45(
            derivative=derivative,
            t0=t,
            y0=s,
            atol=1e-7,
            rtol=1e-5,
            first_step=None
        )
        import timeit
        start = timeit.default_timer() 
        # for i in range(N):
        i = 0
        # while t < tf:
        for _ in range(N):
            # t, s = rk4(t, s, derivative)
            t, s = rk45._adaptive_step(t, s)
            # print(s.data['R'].shape)
            # out[i] = s.r
            # t[i] = s.p
            # rho_out[i] = s.rho[:]
            out[i] = s.data
            # out[i] = s.state
            time[i] = t
            i += 1
            print("dt: ", rk45.h)
            
        time_elapsed = timeit.default_timer() - start
        print(f"The speed of the integrator: {N/time_elapsed} iteractions per second")
    
    benchmark(N)
         
    from matplotlib import pyplot as plt
    
    r = out['R']
    p = out['P']
    # print(r.shape)
    
    plt.plot(time, r[:, 0], label="r")
    plt.plot(time, p[:, 0], label="p")
    plt.legend()
    
    plt.show()
    plt.figure(dpi=300)
    plt.plot(r[:, 0], p[:, 0])
    plt.plot(r[:, 1], p[:, 1])
    plt.plot(r[:, -1], p[:, -1])
    
    print(r[:, 0])
    print(r[:, 0].min(), r[:, 0].max())
    print(time)
    E = r**2 + p**2
    print(E.shape)

# %%
E = np.nansum(r**2 + p**2, axis=-1)
print(E)
print(E.shape)
plt.plot(time, E-E[0])
plt.yscale('symlog')


# %%
