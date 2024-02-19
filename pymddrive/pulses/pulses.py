# %% The package
from pymddrive.utils import zeros

class Pulse:
    def __call__(
        self,
        time
    ):
        return zeros(time)
    
class MultiPulse(Pulse):
    def __init__(
        self,
        *pulses
    ):
        self.pulse_list = pulses
        # self.pulses = pulses
        
    def __call__(
        self,
        time
    ):
        return sum([p(time) for p in self.pulse_list])
        

# %% the temperary test code
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("Hello.")
    p = Pulse()
    t = np.linspace(0, 10, 100)
    sig = p(t)
    plt.plot(t, sig)