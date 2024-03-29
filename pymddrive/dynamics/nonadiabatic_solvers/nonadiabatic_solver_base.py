from abc import ABC, abstractmethod

from pymddrive.my_types import RealVector, ComplexVector, ComplexOperator

from typing import Union

class NonadiabaticSolverBase(ABC):
    @abstractmethod
    def callback(self, *args, **kwargs):    
        pass
    
    @abstractmethod
    def derivative(self, *args, **kwargs):
        pass
    
    @abstractmethod
    @classmethod
    def initialize(cls, *args, **kwargs):
        pass