# %% 
"""
This module defines the Pulse and MultiPulse classes for handling pulse signals.
"""
import attr
from attrs import define, field

from typing import Optional, Union
from numbers import Real
from collections import OrderedDict
from abc import ABC, abstractmethod

# TypeOmega: TypeAlias = Union[int, float, None]

@define
class PulseBase(ABC):
    Omega: float = field(default=float('nan'), on_setattr=attr.setters.frozen)
    _cache: OrderedDict = field(factory=OrderedDict, init=False)
    _cache_length: int = field(default=30, init=False)
        
    def __call__(self, time: float):
        if time in self._cache:
            return self._cache[time]
        else:
            return self._post_call(time)
        
    def _post_call(self, time: float):
        self._cache[time] = self._pulse_func(time)
        if len(self._cache) > self._cache_length:
            self._cache.popitem(last=False)
        return self._cache[time]
    
    @abstractmethod
    def _pulse_func(self, time: float):
        pass

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
        
    def gradient(self, t: float) -> float:
        """
        Args:
            t (float): the time

        Returns:
            float: the gradient for the pulse
        """
        if hasattr(self, '_gradient_func'):
            return self._gradient_func(t)
        else:
            return PulseBase.finite_deference_gradient(self, t) 
        
    @staticmethod
    def finite_deference_gradient(pulse, t: float) -> float:
        """Ad-hoc finite difference implementation of the gradient for the pulse.

        Args:
            t (float): the time

        Returns:
            float: the gradient for the pulse
        """
        delta = 1e-6
        return (pulse(t + delta) - pulse(t - delta)) / (2 * delta)
    
    
    @abstractmethod
    def cannonical_amplitude(self, t: float) -> Union[complex, float]:
        """The 'cannonical' amplitude of the pulse.
        
        In dipole approximation, the electric field of the pulse is given by
            E(t) = 0.5 * [\epsilon(t)e^{i\Omega t} + \epsilon^*(t)e^{-i\Omega t}] 
                 = Re[\epsilon(t)] cos(\Omega t) - Im[\epsilon(t)] sin(\Omega t)
        This function returns the complex amplitude \epsilon(t) of the pulse.

        Args:
            t (float): the time

        Returns:
            Union[complex, float]: the complex amplitude of the pulse
        """
        raise NotImplementedError("The cannonical_amplitude method must be implemented in the subclass.")
       
        
# %%
