# %% The package: pymddrive.integrators.rk4
import numpy as np
from typing import Any, Callable, Tuple, Union
from dataclasses import field

# from pymddrive.integrators.rungekutta import 
from pymddrive.integrators.state import State, zeros_like

# Type short hand
type_time = float

def rkgill4(
    t: type_time,
    y: State,
    derivative: Callable[[type_time, State], State],
    deriv_options: Union[dict, None]= None,
    dt: type_time = 1.0,
    nstep: int = 1,
) -> Tuple[type_time, State]:
    deriv_options = deriv_options or {}
    for _ in range(nstep):
        k1 = dt * derivative(t, y, **deriv_options)
        k2 = dt * derivative(t + 0.5 * dt, y + 0.5 * k1, **deriv_options)
        k3 = dt * derivative(t + 0.5 * dt, y + (0.5 * (1.0 - np.sqrt(2.0))) * k1 + (0.5 * (1.0 + np.sqrt(2.0))) * k2, **deriv_options)
        k4 = dt * derivative(t + 1.0 * dt, y + (-0.5 * np.sqrt(2.0)) * k2 + (1.0 + 0.5 * np.sqrt(2.0)) * k3, **deriv_options)

        t += dt
        y += 1.0 / 6.0 * (k1 + (2.0 - np.sqrt(2.0)) * k2 + (2.0 + np.sqrt(2.0)) * k3 + k4)

    return (t, y)

def rk4(
    t: type_time,
    y: State,
    derivative: Callable[[type_time, State], State],
    deriv_options: Union[dict, None]= None,
    dt: type_time = 1.0,
    nstep: int = 1,
) -> Tuple[type_time, State]:
    deriv_options = deriv_options or {}
    for _ in range(nstep):
        k1 = dt * derivative(t, y, **deriv_options)
        k2 = dt * derivative(t + 0.5 * dt, y + 0.5 * k1, **deriv_options)
        k3 = dt * derivative(t + 0.5 * dt, y + 0.5 * k2, **deriv_options)
        k4 = dt * derivative(t + 1.0 * dt, y + k3, **deriv_options)

        t += dt
        y += 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return (t, y)

# %% The temporary testing/debuging code
def _debug_test():
    def derivative(t: float, s: State):
        out = zeros_like(s)
        r, p, _ = s.get_variables()
        kr, kp, _ = out.get_variables()
        # print(type(out))
        # print("shape: ", r.shape)
        kr[:] = p
        kp[:] = -r
                
        return 0.01*out
        #return out
    rho = np.array([[0.5, 0], [0, 0.5]])
    n_particle = 100
    # s = State(r=np.random.normal(0, 1, n_particle), p=np.random.normal(0, 1, n_particle), rho=None)
    R = np.random.normal(0, 1, n_particle)
    P = np.random.normal(0, 1, n_particle)
    rho = None
    s = State.from_variables(R=R, P=P, rho=rho)
    
    N = 100000
    time = np.zeros(N)
    out = np.zeros(N, dtype=s.data.dtype)
    
    
    def benchmark(N: int):
        t = 0.0 
        n_particle = 100
        R = np.random.normal(0, 1, n_particle)
        P = np.random.normal(0, 1, n_particle)
        s = State.from_variables(R=R, P=P, rho=None)
        import timeit
        start = timeit.default_timer() 
        for i in range(N):
            t, s = rk4(t, s, derivative, dt=0.05)
            out[i] = s.data
            time[i] = t
        time_elapsed = timeit.default_timer() - start
        print(f"The speed of the integrator: {N/time_elapsed} iteractions per second")
    
    benchmark(N)
         
    from matplotlib import pyplot as plt
    
    r = out['R']
    p = out['P']
    
    plt.plot(time, r[:, 0], label="r")
    plt.plot(time, p[:, 0], label="p")
    plt.title("Time series")
    plt.legend()
    
    plt.show()
    plt.figure(dpi=300)
    plt.plot(r[:, 0], p[:, 0])
    plt.plot(r[:, 1], p[:, 1])
    plt.plot(r[:, -1], p[:, -1])
    plt.xlabel("r")
    plt.ylabel("p")
    plt.title("Phase space")
    plt.show()
    
    E = r**2 + p**2
    print(E.shape)

    E = np.nansum(r**2 + p**2, axis=-1)
    plt.plot(time, E-E[0])
    plt.yscale('symlog') 
    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.title("Energy conservation")

# %% The test code
if __name__ == "__main__":
    _debug_test()
    
# %%
