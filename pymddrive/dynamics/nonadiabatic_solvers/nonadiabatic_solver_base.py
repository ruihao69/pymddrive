from abc import ABC, abstractmethod

class NonadiabaticSolverBase(ABC):
    @abstractmethod
    def derivative(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def callback(self, *args, **kwargs):    
        pass
    
    @abstractmethod
    @classmethod
    def initialize(cls, *args, **kwargs):
        pass 
    