# %%
import numpy as np
from pymddrive.low_level.states import DensityMatrixState as State
from pymddrive.low_level.states import axpy, rk4_step

def get_random_state(nr, ne):
    R = np.random.rand(nr)
    P = np.random.rand(nr)
    rho = np.random.rand(ne, ne) + 1j * np.random.rand(ne, ne)
    rho = rho + rho.T.conj()
    return State(R, P, rho)

def test_reset(nu: int, ne: int):
    state = get_random_state(nu, ne)
    print("Before resetting")
    print(f"{state.R=}")
    print(f"{state.P=}")
    print(f"{state.rho=}")
    
    R = np.random.rand(10)
    
    # state.R[:] = np.random.rand() 
    state.R[:] = np.random.rand(*state.R.shape)
    state.rho[:] = np.random.rand(*state.rho.shape) + 1j * np.random.rand(*state.rho.shape)
    
    rho_offsize = np.random.rand(ne*2, ne*2) + 1j * np.random.rand(ne*2, ne*2)
    rho_offsize = rho_offsize + rho_offsize.T.conj()
    
    print("After resetting")
    print(f"{state.R=}")
    print(f"{state.P=}")
    print(f"{state.rho=}")
    
    print("Flatten view of state")
    print(f"{state.flatten()=}")
    
    state1 = get_random_state(1, 2)
    state2 = get_random_state(1, 2)
    
    print(f"{state1.flatten()=}")
    print(f"{state2.flatten()=}")
    
    state3 = state1 + state2
    print(f"{state3.flatten()=}")
    
    assert np.allclose(state3.R, state1.R + state2.R), "Error in addition"
    
def test_from_unstructured():
    a = get_random_state(1, 2)
    _b = np.zeros_like(a.flatten())
    _b[:] = np.random.rand(*_b.shape)
    b = a.from_unstructured(_b)
    assert np.allclose(b.flatten(), _b), "Error in from_unstructured"

from dataclasses import dataclass

@dataclass
class StatePy:
    R: np.ndarray
    P: np.ndarray
    rho: np.ndarray

    def flatten(self):
        return np.concatenate([self.R, self.P, self.rho.flatten()])

    def __add__(self, other):
        return StatePy(self.R + other.R, self.P + other.P, self.rho + other.rho)
    
    def __mul__(self, scalar):
        return StatePy(self.R * scalar, self.P * scalar, self.rho * scalar)
    
    def axpy(self, alpha, other):
        R = self.R + alpha * other.R
        P = self.P + alpha * other.P
        rho = self.rho + alpha * other.rho
        return StatePy(R, P, rho)
    
def rk4_step_inplace_py(dt: float, state: StatePy, k1: StatePy, k2: StatePy, k3: StatePy, k4: StatePy):
    state.R += dt/6 * (k1.R + 2*k2.R + 2*k3.R + k4.R)
    state.P += dt/6 * (k1.P + 2*k2.P + 2*k3.P + k4.P)
    state.rho += dt/6 * (k1.rho + 2*k2.rho + 2*k3.rho + k4.rho)
    
def rk4_step_py(dt: float, state: State, k1: State, k2: State, k3: State, k4: State):
    state_out = State(state.R, state.P, state.rho)
    rk4_step_inplace_py(dt, state_out, k1, k2, k3, k4)
    return state_out
    
    
def benchmark_axpy(ntests: int=10000):
    import time
    nu, ne = 1, 2
    ne_F = 2 * ne + 1
    state1 = get_random_state(nu, ne_F) 
    state2 = get_random_state(nu, ne_F)
    
    state_py1 = StatePy(state1.R, state1.P, state1.rho)
    state_py2 = StatePy(state2.R, state2.P, state2.rho)
    
    start = time.perf_counter()
    for _ in range(ntests):
        # state3 = state1 + state2
        state3 = axpy(0.2, state1, state2)
        # state3 = state1 * 0.2 + state2
    end = time.perf_counter()
    print(f"Elapsed time (C++): {end-start:.2e}")
    
    start_py = time.perf_counter()
    for _ in range(ntests):
        # state3 = state_py1 + state_py2
        state3 = state_py1.axpy(0.2, state_py2)
        # state3 = state_py1 * 0.2 + state_py2
    end_py = time.perf_counter()
    print(f"Elapsed time (Python): {end_py-start_py:.2e}")
    
def benchmark_rk4_step_inplace(ntests: int=10000):
    import time
    nu, ne = 1, 2
    ne_F = 2 * ne + 1
    state = get_random_state(nu, ne_F) 
    k1 = get_random_state(nu, ne_F)
    k2 = get_random_state(nu, ne_F)
    k3 = get_random_state(nu, ne_F)
    k4 = get_random_state(nu, ne_F)
    
    start = time.perf_counter()
    for _ in range(ntests):
        rk4_step(0.1, state, k1, k2, k3, k4)
    end = time.perf_counter()
    print(f"Elapsed time (C++): {end-start:.2e}")
    
    state_py = StatePy(state.R, state.P, state.rho)
    k1_py = StatePy(k1.R, k1.P, k1.rho)
    k2_py = StatePy(k2.R, k2.P, k2.rho)
    k3_py = StatePy(k3.R, k3.P, k3.rho)
    k4_py = StatePy(k4.R, k4.P, k4.rho)
    
    start_py = time.perf_counter()
    for _ in range(ntests):
        rk4_step_inplace_py(0.1, state_py, k1_py, k2_py, k3_py, k4_py)
    end_py = time.perf_counter()
    print(f"Elapsed time (Python): {end_py-start_py:.2e}")
    
def benchmark_rk4_step(ntests: int=100000):
    import time
    nu, ne = 1, 2
    ne_F = 2 * ne + 1
    state = get_random_state(nu, ne_F) 
    k1 = get_random_state(nu, ne_F)
    k2 = get_random_state(nu, ne_F)
    k3 = get_random_state(nu, ne_F)
    k4 = get_random_state(nu, ne_F)
    
    start = time.perf_counter()
    for _ in range(ntests):
        state = rk4_step(0.1, state, k1, k2, k3, k4)
    end = time.perf_counter()
    print(f"Elapsed time (C++): {end-start:.2e}")
    
    state_py = StatePy(state.R, state.P, state.rho)
    k1_py = StatePy(k1.R, k1.P, k1.rho)
    k2_py = StatePy(k2.R, k2.P, k2.rho)
    k3_py = StatePy(k3.R, k3.P, k3.rho)
    k4_py = StatePy(k4.R, k4.P, k4.rho)
    
    start_py = time.perf_counter()
    for _ in range(ntests):
        state_py = rk4_step_py(0.1, state_py, k1_py, k2_py, k3_py, k4_py)
    end_py = time.perf_counter()
    print(f"Elapsed time (Python): {end_py-start_py:.2e}")
    
def get_variables(state):
    return state.R, state.P, state.rho
    
def test_get_variables():
    state = get_random_state(1, 2)
    # R, P, rho = state.get_varibles()
    R, P, rho = get_variables(state)
    print(f"{np.allclose(R, state.R)=}")
    print(f"{np.allclose(P, state.P)=}")
    print(f"{np.allclose(rho, state.rho)=}")
    
    # test that I can mutate the variables inplace
    print(f"before mutate: {R=}, {state.R=}")
    R[:] = np.random.rand(*R.shape)
    print(f"after mutate: {R=}, {state.R=}")
    
       
if __name__ == "__main__":
    print("=== Testing reset ===")
    test_reset(1, 2)
    print("=====================")
    
    print("=== Testing from_unstructured ===")
    test_from_unstructured() 
    print("=====================")
    
    print("=== Testing benchmark axpy ===")
    benchmark_axpy()
    print("=====================")
    
    print("=== Testing benchmark rk4_step_inplace ===")
    benchmark_rk4_step_inplace()
    print("=====================")
    
    print("=== Testing benchmark rk4_step ===")
    benchmark_rk4_step()
    print("=====================")
    
    print("=== Testing get_variables ===")
    test_get_variables()
    print("=====================")
    
    



# %%
