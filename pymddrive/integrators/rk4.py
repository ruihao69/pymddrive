# %% The package: pymddrive.integrators.rk4
import numpy as np
from typing import Any, Callable, Tuple

# from pymddrive.integrators.rungekutta import 
from pymddrive.integrators.state import State, zeros_like

# Type short hand
type_time = float

def rk4(
    t: type_time,
    y: State,
    derivative: Callable[[type_time, State], State],
    dt: type_time = 1.0,
    nstep: int = 1,
) -> Tuple[type_time, State]:
    k1 = dt * derivative(t, y)
    k2 = dt * derivative(t + 0.5 * dt, y + 0.5 * k1)
    k3 = dt * derivative(t + 0.5 * dt, y + 0.5 * k2)
    k4 = dt * derivative(t + 1.0 * dt, y + k3)
        
    t += dt
    y += 1.0 / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return (t, y)

# %% The test code
if __name__ == "__main__":
    # from pymddrive.integrators.state import StateMQC, zeros_like
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
        rho = np.array([[0.5, 0], [0, 0.5]])
        n_particle = 100
        s = State(r=np.random.normal(0, 1, n_particle), p=np.random.normal(0, 1, n_particle), rho=None)
        import timeit
        start = timeit.default_timer() 
        for i in range(N):
            t, s = rk4(t, s, derivative, dt=0.05)
            # out[i] = s.r
            # t[i] = s.p
            # rho_out[i] = s.rho[:]
            out[i] = s.data
            # out[i] = s.state
            time[i] = t
        time_elapsed = timeit.default_timer() - start
        print(f"The speed of the integrator: {N/time_elapsed} iteractions per second")
    
    benchmark(N)
         
    from matplotlib import pyplot as plt
    
    r = out['R']
    p = out['P']
    print(r.shape)
    
    plt.plot(time, r[:, 0], label="r")
    plt.plot(time, p[:, 0], label="p")
    plt.legend()
    
    plt.show()
    plt.figure(dpi=300)
    plt.plot(r[:, 0], p[:, 0])
    plt.plot(r[:, 1], p[:, 1])
    plt.plot(r[:, -1], p[:, -1])
    plt.show()
    
# %%
E = np.nansum(r**2 + p**2, axis=-1)
print(E)
print(E.shape)
plt.plot(time, E-E[0])
plt.yscale('symlog')

# %%
