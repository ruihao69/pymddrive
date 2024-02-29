# %% 
"""
This module defines the Pulse and MultiPulse classes for handling pulse signals.
"""
import numpy as np
from typing import Union
from numbers import Real
from collections import OrderedDict

class Pulse:
    def __init__(
        self,
        Omega: Union[float, None] = None,
        cache_length: int = 30
    ):
        """
        Initialize a Pulse object.

        Args:
            cache_length (int): The maximum length of the cache.
        """
        self.Omega = Omega
        self._cache = OrderedDict()
        self._cache_length = cache_length
        
    def __call__(self, time: float):
        """
        Call the Pulse object with a time value.

        Args:
            time (float): The time value.

        Returns:
            float: The calculated value at the given time.
        """
        if time in self._cache:
            # print(f"From <class Pulse>: Retrieving value from the cache for time {time}")
            return self._cache[time]
        else:
            # print(f"From <class Pulse>: Calculating the value for time {time}")
            return self._post_call(time)
    
    def _post_call(self, time: float):
        """
        Perform post-call operations.

        Args:
            time (float): The time value.

        Returns:
            float: The calculated value at the given time.
        """
        self._cache[time] = self._pulse_func(time)
        if len(self._cache) > self._cache_length:
            self._cache.popitem(last=False)
        return self._cache[time]
    
    def _pulse_func(self, time: float):
        """
        Calculate the pulse value at the given time.

        Args:
            time (float): The time value.

        Returns:
            float: The calculated pulse value.
        """
        return 0.0
    
    def set_Omega(self, Omega: float):
        """
        Set the carrier frequency of the pulse.

        Args:
            Omega (float): The carrier frequency.
        """
        if isinstance(Omega, Real):
            self.Omega = Omega
        else:
            raise ValueError(f"After the pulse has been initialized, you can only set the carrier frequency with a real number, not {Omega}")

class UnitPulse(Pulse):
    def __init__(self, A: float=1.0, cache_length: int = 1000):
        super().__init__(None, cache_length)
        self.A = A
        
    def _pulse_func(self, t: float) -> float:
        return self.A

class CosinePulse(Pulse):
    def __init__(
        self,
        A: float = 1,        # the amplitude of the cosine pulse
        Omega: float = 1,    # the carrier frequency of the pulse
        cache_length: int = 40
    ):
        super().__init__(Omega, cache_length)
        self.A = A
    
    def _pulse_func(self, time: float):
        return self.A * np.cos(self.Omega * time)
    
    
class SinePulse(Pulse):
    def __init__(
        self,
        A: float = 1,        # the amplitude of the sine pulse
        Omega: float = 1,    # the carrier frequency of the pulse
        cache_length: int = 40
    ):
        super().__init__(Omega, cache_length)
        self.A = A
    
    def _pulse_func(self, time: float):
        return self.A * np.sin(self.Omega * time)
    
class MultiPulse(Pulse):
    def __init__(
        self,
        *pulses: Pulse,
        cache_length: int = 40
    ):
        """
        Initialize a MultiPulse object.

        Args:
            *pulses (Pulse): Variable number of Pulse objects.
            cache_length (int): The maximum length of the cache.
        """
        self.pulses = pulses
        self._cache = OrderedDict()
        self._cache_length = cache_length
        
    def __call__(self, time: float):
        """
        Call the MultiPulse object with a time value.

        Args:
            time (float): The time value.

        Returns:
            float: The calculated value at the given time.
        """
        if time in self._cache:
            # print(f"From <class MultiPulse>: Retrieving value from the cache for time {time}")
            return self._cache[time]
        else:
            # print(f"From <class MultiPulse>: Calculating the value for time {time}")
            return self._post_call(time)
    
    def _post_call(self, time: float):
        """
        Perform post-call operations.

        Args:
            time (float): The time value.

        Returns:
            float: The calculated value at the given time.
        """
        self._cache[time] = self._pulse_func(time)
        if len(self._cache) > self._cache_length:
            self._cache.popitem(last=False)
        return self._cache[time]
    
    def _pulse_func(self, time: float):
        """
        Calculate the pulse value at the given time.

        Args:
            time (float): The time value.

        Returns:
            float: The calculated pulse value.
        """
        return sum(p(time) for p in self.pulses)
    
def get_carrier_frequency(pulse: Pulse) -> float:
    return pulse.Omega        

# %% the temporary testing/debugging code
def _debug_test():
    import numpy as np
    import matplotlib.pyplot as plt
    
    # test the caching mechanism of the Pulse class
    pulse1 = Pulse()
    pulse2 = pulse1  # This is another reference to the same instance

    # Call the instance with a time value
    print(pulse1(5))
    print(pulse2(5))  # This should return the cached value

    # Print the cache to verify
    t = np.random.rand(1000) * 10
    for tt in t:
        pulse1(tt) # This should fill the cache
        
    pulse1(t[0])
        
    multi_pulse = MultiPulse(pulse1, pulse2)
    for tt in t:
        multi_pulse(tt)

# %% the __main__ code
if __name__ == "__main__":
    _debug_test() 
        
    
# %%
