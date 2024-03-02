# %% The package: pymddrive.integrators.rungekutta
""" This module conatains general utility functions for the Runge-Kutta type integrators. """

import numpy as np

from collections import namedtuple
from typing import Callable, Union, Tuple

from pymddrive.integrators.state import State   

RungeKuttaCoefficients = namedtuple("RungeKuttaCoefficients", ["a", "b", "c", "bb"])

# tables
""" Tsit5 Butcher tableau: 10.1016/j.camwa.2011.06.002"""
tsit5_tableau = RungeKuttaCoefficients(
    a = np.array([
        [0, 0, 0, 0, 0, 0, 0],
        [0.161, 0, 0, 0, 0, 0, 0],
        [-0.008480655492356992, 0.3354806554923570, 0, 0, 0, 0, 0],
        [2.897153057105495, -6.359448489975075, 4.362295432869581, 0, 0, 0, 0],
        [5.32586482843926, -11.74888356406283,7.495539342889836, -0.09249506636175525, 0, 0, 0],
        [5.804917342158284, -12.92096931784711, 8.159367898576159, -0.07158497328140100, 0.02826905039406838, 0, 0],
        [0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0]
    ]),
    b = np.array(
        [0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0]
    ), 
    c = np.array(
        [0, 0.161, 0.327, 0.9, 0.9800255409045097, 1, 1]
    ),
    bb = np.array(
        [0.001780011052226, 0.000816434459657, -0.007880878010262, 0.144711007173263, -0.582357165452555, 0.458082105929187, 1.0/66.0]
    )
)

def _evaluate_y1(
    y0: State, 
    k: list,
    beta_i: np.ndarray,
    i: int,
    h: float
) -> State:
    yi = y0 + h * k[0] * beta_i[0]
    for j in range(1, i):
        yi += h * k[j] * beta_i[j]
    return yi

def _evaluate_y1_err(
    k: list,
    bb: np.ndarray,
    h: float
) -> State:
    y1_err = k[0] * bb[0] * h
    for i in range(1, len(k)):
        y1_err += k[i] * bb[i] * h
    return y1_err

def runge_kutta_step(
    derivative: Callable[[float, State], State],
    t0: float,
    y0: State,
    h: float, 
    tableau: RungeKuttaCoefficients, 
    *args,
    **kwargs
) -> Tuple[State, State, State, list]:
    """ Runge-Kutta stepper given a coefficient table

    Args:
        derivative (Callable[[float, State], State]): the derivative function (encodes dynamics)
        t0 (float): the initial time
        y0 (State): the intial state
        h (float): step size
        tableau (RungeKuttaCoefficients): the Runge-Kutta coefficient table

    Returns:
        Tuple[State, State, State, list]: the next state, the derivative at the next state, the error, and the intermediate steps
        
    Reference:
        - github.com/autonomousvision/occupancy_flow.git
    """
    k = None 
    
    for i, (alpha_i, beta_i) in enumerate(zip(tableau.c, tableau.a)): 
        if k is None:
            ti = t0
            yi = y0
            k = [derivative(ti, yi)]  
        else:
            ti = t0 + alpha_i * h
            yi = _evaluate_y1(y0, k, beta_i, i, h)
            k.append(derivative(ti, yi))
    
    y1 = yi
    f1 = k[-1]
    y1_err = _evaluate_y1_err(k, tableau.bb, h)
    
    
    return (y1, f1, y1_err, k)

def state_to_vectors(s: State)->Tuple[np.ndarray]:
    R, P, rho = s.get_variables() 
    vec_cl = np.concatenate((R.reshape(-1), P.reshape(-1)))
    vec_qm = rho.reshape(-1)   
    return vec_cl, vec_qm

def vectors_to_states(s: State, vec_cl: np.ndarray, vec_qm: np.ndarray) -> State:
    _R, _P, _rho = s.get_variables()
    
    rho = vec_qm.reshape(_rho.shape)
    R, P = vec_cl.reshape(2, vec_cl.shape[0]//2)
    R = R.reshape(_R.shape)
    P = P.reshape(_P.shape)
    return State.from_variables(R, P, rho)

def rms_norm(array):
    return np.sqrt(np.mean(np.square(array)))

def evaluate_initial_dt(
    derivative: Callable[[float, State], State],
    t0: float,
    y0: State,
    order: int,
    rtol: float,
    atol: float,
    f0: Union[None, State]=None
):
    
    """Evaluate the initial step size for the Runge-Kutta type integrators.

    Args:
        derivative (Callable[[float, State], State]): the derivative function (encodes dynamics)
        t0 (float): the initial time
        y0 (State): the intial state
        order (int): the order of the Runge-Kutta type integrator
        rtol (float): the relative tolerance
        atol (float): the absolute tolerance
        f0 (Union[None, State], optional): the derivative at (t0, y0). Defaults to None.

    Returns:
        h: the proposed intial step size
        
    Reference:
        - github.com/autonomousvision/occupancy_flow.git
        - E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
          Equations I: Nonstiff Problems", Sec. II.4.
    """
    if f0 is None:
        f0 = derivative(t0, y0)
    _y0 = y0.flatten(copy=True)
    _f0 = f0.flatten(copy=True)
    
    scale = atol + np.abs(_y0) * rtol
    d0 = np.linalg.norm(_y0 / scale)
    d1 = np.linalg.norm(_f0 / scale)
    
    if np.max(d0) < 1e-5 or np.max(d1) < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * max(d0, d1)
        
    _y1 = _y0 + h0 * _f0 
    y1 = State.from_unstructured(flat_data=_y1, dtype=y0.data.dtype, stype=y0.stype, copy=True)
    f1 = derivative(t0 + h0, y1)
    _f1 = f1.flatten(copy=True)
    
    d2 = np.linalg.norm((_f1 - _f0) / scale) / h0
    
    if np.max(d1) <= 1e-15 and np.max(d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / (order + 1))
        
    return min(100 * h0, h1)


def get_optimal_step_size(
    last_h: float,
    mean_error_ratio: float,
    safety: float=0.9, 
    ifactor: float=10.0, 
    dfactor: float=0.2, 
    order: float=5
) -> float:
    if mean_error_ratio == 0:
        return last_h * ifactor
    if mean_error_ratio < 1:
        dfactor = 1.0
        
    error_ratio = np.sqrt(mean_error_ratio)
    exponent = 1.0 / order 
    factor = max(1/ifactor, min(error_ratio**exponent/safety, 1.0/dfactor))
    return last_h / factor

    

# %% temporary
if __name__ == "__main__":
    print(tsit5_tableau.a)

# %%
