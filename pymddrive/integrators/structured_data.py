# %% The package
import numpy as np
from numpy.lib import recfunctions as rfn

from numbers import Number, Real, Complex

from typing import Union, Any
from numpy.typing import ArrayLike

class LazyEvaluation:
    def __init__(
        self, 
        op: str, 
        operand, 
        _class
    ) -> None:
        self.op = op
        self.operand = operand
        self._class = _class

    def evaluate(
        self, 
        data
    ):
        if self.op == "add":
            return self._class(data=np.array(
                tuple(data[name] + self.operand.data[name] for name in data.dtype.names)
            , dtype=data.dtype))
            
        if self.op == "add_s":
            return self._class(data=np.array(
                tuple(data[name] + self.operand for name in data.dtype.names)
            , dtype=data.dtype))
            
        elif self.op == "mul":
            return self._class(data=np.array(
                tuple(data[name] * self.operand for name in data.dtype.names)
            , dtype=data.dtype))
            
class CompositeData:
    def __init__(
        self, 
        data: ArrayLike, 
        *args: Any, 
        **kwargs: Any
    ) -> None:
        self.data = data
        self.lazy_op = None
        
    def __repr__(self) -> str:
        return f"""StructrueData with 
    - fields: {self.data.dtype.names}, 
    - types: {tuple(self.data[name].dtype for name in self.data.dtype.names)},
    - shape: {tuple(self.data[name].shape for name in self.data.dtype.names)}"""

    def __add__(self, other) -> "CompositeData":
        if isinstance(other, CompositeData):
            self.lazy_op = LazyEvaluation("add", other, _class=self.__class__)
        elif isinstance(other, Number):
            self.lazy_op = LazyEvaluation("add_s", other, _class=self.__class__)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'CompositeData' and '{}'".format(type(other)))
        return self.evaluate()

    def __radd__(self, other) -> "CompositeData":
        return self.__add__(other)

    def __mul__(self, scalar) -> "CompositeData":
        if isinstance(scalar, Number):
            self.lazy_op = LazyEvaluation("mul", scalar, _class=self.__class__)
        else:
            raise TypeError("Unsupported operand type(s) for *: 'CompositeData' and '{}'".format(type(scalar)))
        return self.evaluate()
    
    def __lmul__(self, scalar) -> "CompositeData":
        return self.__mul__(scalar)

    def __rmul__(self, scalar) -> "CompositeData":
        return self.__mul__(scalar)
    
    def __neg__(self) -> "CompositeData":
        return self * -1
    
    def __sub__(self, other) -> "CompositeData":
        return self + (-other)
    
    def __div__(self, scalar) -> "CompositeData":
        return self * (1.0 / scalar)
        
    def __imul__(self, scalar) -> "CompositeData":
        for name in self.data.dtype.names:
            self.data[name] *= scalar
        return self
    
    def __isub__(self, other) -> "CompositeData":
        for name in self.data.dtype.names:
            self.data[name] -= other.data[name]
        return self
    
    def __idiv__(self, scalar) -> "CompositeData":
        for name in self.data.dtype.names:
            self.data[name] /= scalar
        return self
    
    def __iadd__(self, other) -> "CompositeData":
        for name in self.data.dtype.names:
            self.data[name] += other.data[name]
        return self

    def evaluate(self):
        if self.lazy_op is None:
            return self
        else:
            result = self.lazy_op.evaluate(self.data)
            self.lazy_op = None  # Reset lazy operation
            return result
        
    def get_ndim(self):
        return self.data.ndim
        
    def flatten(self, copy=True):
        return rfn.structured_to_unstructured(self.data, copy=copy)
    
# %% The debugging/testing code
def _debug_test():
    cdtype = [("A", np.float64, (3,)), ("B", np.complex128, (2, 2))]

    data1 = np.array([(1.0, np.array([[1+2j, 3+4j], [5+6j, 7+8j]]))], dtype=cdtype)
    data2 = np.array([(2.0, np.array([[2+4j, 6+8j], [10+12j, 14+16j]]))], dtype=cdtype)

    composite1 = CompositeData(data1)
    composite2 = CompositeData(data2)

    # New example usage involving scalars and CompositeData instances
    result_add = composite1 + composite2
    print("Composite1: ", composite1.data["A"])
    print("Composite2: ", composite2.data["A"])
    print("Addition result:", result_add.data["A"])
    
    result_mul = composite1 * 2.0
    print("Multiplication result:", result_mul.data["A"])
    
    result_mul_add = composite1 * 2.0 + composite2
    print("Multiplication and addition result:", result_mul_add.data["A"])
    
    # In-place operations 
    composite3 = CompositeData(data1)
    print("before in-place addition: ", composite3.data["A"]) 
    composite3 += composite1 + composite2 * 2.0
    print("after in-place addition: ", composite3.data["A"])
    
    # not implemented
    # composite3 - composite1
    print(composite3.data['A'].shape)
    iter_sum = composite3 + 0.1 * sum(0.1 * cc for cc in [composite1, composite2]) 
    print(f"{iter_sum=}")
    print(f"{iter_sum.data=}")

# %% the __main__ code
if __name__ == "__main__": 
    _debug_test()
     
    
# %%
