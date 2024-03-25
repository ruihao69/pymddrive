# %% The package: pymddrive.integrators.rk4
import numpy as np
from numpy.typing import NDArray

from pymddrive.low_level.states import State

from typing import Callable, Tuple, Union

# Type short hand
type_time = float


def rkgill4(
    t: type_time,
    y: NDArray[np.any],
    derivative: Callable[[type_time, NDArray[np.any]], NDArray[np.any]],
    deriv_options: Union[dict, None]= None,
    dt: type_time = 1.0,
    nstep: int = 1,
) -> Tuple[type_time, NDArray[np.any]]:
    raise NotImplementedError("The RKGill4 integrator is not implemented yet.")

def rk4(
    t: type_time,
    y: NDArray[np.any],
    derivative: Callable[[type_time, NDArray[np.any]], NDArray[np.any]],
    deriv_options: Union[dict, None]= None,
    dt: type_time = 1.0,
    nstep: int = 1,
) -> Tuple[type_time, NDArray[np.any]]:
    deriv_options = deriv_options or {}
    for _ in range(nstep):
        k1 = derivative(t, y, **deriv_options)
        k2 = derivative(t + 0.5 * dt, y + 0.5 * dt * k1, **deriv_options)
        k3 = derivative(t + 0.5 * dt, y + 0.5 * dt * k2, **deriv_options)
        k4 = derivative(t + 1.0 * dt, y + dt * k3, **deriv_options)
        # k1 = derivative(t, y)
        # k2 = derivative(t + 0.5 * dt, State.axpy(0.5 * dt, k1, y))
        # k3 = derivative(t + 0.5 * dt, State.axpy(0.5 * dt, k2, y))
        # k4 = derivative(t + 1.0 * dt, State.axpy(dt, k3, y))
        y += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt
        
    return (t, y)

# %% The temporary testing/debuging code
def _debug_test():
    n_particle = 100
    from pymddrive.integrators.state import get_state
    def derivative(t: float, y: NDArray[np.any]):
        dy_dt = np.zeros_like(y)
        R, P = y[:n_particle], y[n_particle:]
        dy_dt[:n_particle] = P / 1.0 * 0.01
        dy_dt[n_particle:] = -R / 1.0 * 0.01
        return dy_dt
    
    mass = 1.0
    R = np.random.normal(0, 1, n_particle)
    P = np.random.normal(0, 1, n_particle)
    y = np.concatenate([R, P])
    
    N = 100000
    time = np.zeros(N)
    
    def benchmark(N: int):
        t = 0.0 
        n_particle = 100
        R = np.random.normal(0, 1, n_particle)
        P = np.random.normal(0, 1, n_particle)
        mass = 1.0
        y = np.concatenate([R, P])
        import timeit
        start = timeit.default_timer() 
        R_out = np.zeros((N, n_particle))
        P_out = np.zeros((N, n_particle))
        
        for i in range(N):
            t, y = rk4(t, y, derivative, dt=0.05)
            R_out[i, :], P_out[i, :] = y[:n_particle], y[n_particle:]
            time[i] = t
        time_elapsed = timeit.default_timer() - start
        print(f"The speed of the integrator: {N/time_elapsed} iteractions per second")
        return time, R_out, P_out
    
    time, R_out, P_out = benchmark(N)
         
    from matplotlib import pyplot as plt
    
    print(f"{R.shape=}") 
     
    plt.plot(time, R_out[:, 0], label="r")
    plt.plot(time, P_out[:, 0], label="p")
    plt.title("Time series")
    plt.legend()
    
    plt.show()
    plt.figure(dpi=300)
    plt.plot(R_out[:, 0], P_out[:, 0])
    plt.plot(R_out[:, 1], P_out[:, 1])
    plt.plot(R_out[:, -1], P_out[:, -1])
    plt.xlabel("r")
    plt.ylabel("p")
    plt.title("Phase space")
    plt.show()
    
    E = R_out**2 + P_out**2
    print(E.shape)

    E = np.nansum(R_out**2 + P_out**2, axis=-1)
    plt.plot(time, E-E[0])
    plt.yscale('symlog') 
    plt.xlabel("time")
    plt.ylabel("Energy")
    plt.title("Energy conservation")
    

# %% The test code
if __name__ == "__main__":
    _debug_test()
    
# %%
