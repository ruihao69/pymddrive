# %% The package: pymddrive.integrators.rk4
import numpy as np

from pymddrive.low_level.states import State

from typing import Callable, Tuple, Union

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
    raise NotImplementedError("The RKGill4 integrator is not implemented yet.")

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
        k1 = derivative(t, y, **deriv_options)
        k2 = derivative(t + 0.5 * dt, State.axpy(0.5 * dt, k1, y), **deriv_options)
        k3 = derivative(t + 0.5 * dt, State.axpy(0.5 * dt, k2, y), **deriv_options)
        k4 = derivative(t + 1.0 * dt, State.axpy(1.0 * dt, k3, y), **deriv_options)
        # k1 = derivative(t, y)
        # k2 = derivative(t + 0.5 * dt, State.axpy(0.5 * dt, k1, y))
        # k3 = derivative(t + 0.5 * dt, State.axpy(0.5 * dt, k2, y))
        # k4 = derivative(t + 1.0 * dt, State.axpy(dt, k3, y))
        y.rk4_step_inplace(dt, k1, k2, k3, k4)
        t += dt
        
    return (t, y)

# %% The temporary testing/debuging code
def _debug_test():
    from pymddrive.integrators.state import get_state
    def derivative(t: float, s: State):
        out = s.zeros_like()
        # r, p, _ = get_variables(s)
        r, p, _ = s.get_variables()
        # out.set_R(p * 0.01 / s.get_mass())
        # out.set_P(-r * 0.01)
        # out.set_R(s.get_P() * 0.01 / s.get_mass())
        # out.set_P(-s.get_R() * 0.01)
        out.set_R(p * 0.01 / s.get_mass())
        out.set_P(-r * 0.01)
        return out
    n_particle = 100
    mass = 1.0
    R = np.random.normal(0, 1, n_particle)
    P = np.random.normal(0, 1, n_particle)
    rho_dummy = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)
    s = get_state(mass, R, P, rho_dummy)
    
    N = 100000
    time = np.zeros(N)
    
    def benchmark(N: int):
        t = 0.0 
        n_particle = 100
        R = np.random.normal(0, 1, n_particle)
        P = np.random.normal(0, 1, n_particle)
        rho_dummy = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)
        mass = 1.0
        s = get_state(mass, R, P, rho_dummy)
        import timeit
        start = timeit.default_timer() 
        R_out = np.zeros((N, n_particle))
        P_out = np.zeros((N, n_particle))
        
        for i in range(N):
            t, s = rk4(t, s, derivative, dt=0.05)
            # R_out[i, :], P_out[i, :], _ = s.get_R(), s.get_P(), s.get_rho()
            R_out[i, :], P_out[i, :], _ = s.get_variables()
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
    
def _pure_cpp_test():
    from pymddrive.low_level.states import State
    n_particle = 100
    R = np.random.normal(0, 1, n_particle)
    P = np.random.normal(0, 1, n_particle)
    rho_dummy = np.array([[0.5, 0], [0, 0.5]], dtype=np.complex128)
    mass = 1.0
    s = State(R, P, mass, rho_dummy)
    
    def derivative(t: float, s: State):
        out = s.zeros_like()
        out.set_R(s.get_P() * 0.01 / s.get_mass())
        out.set_P(-s.get_R() * 0.01)
        return out
    
    def rk4(t: float, s: State, dt: float):
        k1 = derivative(t, s)
        k2 = derivative(t + 0.5 * dt, State.axpy(0.5 * dt, k1, s))
        k3 = derivative(t + 0.5 * dt, State.axpy(0.5 * dt, k2, s))
        k4 = derivative(t + 1.0 * dt, State.axpy(dt, k3, s))
        s.rk4_step_inplace(dt, k1, k2, k3, k4)
        t += dt
        return t, s
    
    N = 100000 
    R_out = np.zeros((N, n_particle))
    P_out = np.zeros((N, n_particle))
    import time 
    
    start = time.time()
    t = 0.0
    for i in range(N):
        t, s = rk4(t, s, 0.05)
        R_out[i, :], P_out[i, :] = s.get_R(), s.get_P() 
    time_elapsed = time.time() - start
    print(f"The speed of the integrator: {N/time_elapsed} iteractions per second")


# %% The test code
if __name__ == "__main__":
    _debug_test()
    _pure_cpp_test()
    
# %%
