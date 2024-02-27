# %% 
"""
This module defines the Pulse and MultiPulse classes for handling pulse signals.
"""

from collections import OrderedDict
from pymddrive.utils import zeros
from numbers import Real

class Pulse:
    def __init__(
        self,
        Omega: float = 1,
        cache_length: int = 1000
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
        self.Omega = Omega
    
class MultiPulse(Pulse):
    def __init__(
        self,
        *pulses: Pulse,
        cache_length: int = 1000
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

# %% the temporary test code
if __name__ == "__main__":
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
        
    
# %%
