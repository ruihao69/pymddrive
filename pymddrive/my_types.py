from nptyping import NDArray, Int64, Shape, Complex128, Float64
from typing import Union, TypeAlias
from typing import TypeVar

A = TypeVar('A')
B = TypeVar('B')
M = TypeVar('M')
N = TypeVar('N')

# Shape types
ShapeOperator = Shape['A, A']
ShapeVector = Shape['A']
ShapeVectorOperator = Shape['A, A, B']

# Custom numpy object types

RealOperator = NDArray[Shape['A, A'], Float64]
ComplexOperator = NDArray[Shape['A, A'], Complex128]
GenericOperator = Union[RealOperator, ComplexOperator]

RealVector = NDArray[Shape['A'], Float64]
ComplexVector = NDArray[Shape['A'], Complex128]
GenericVector = Union[RealVector, ComplexVector]

RealVectorOperator = NDArray[Shape['A, A, B'], Float64]
ComplexVectorOperator = NDArray[Shape['A, A, B'], Complex128]
GenericVectorOperator = Union[RealVectorOperator, ComplexVectorOperator]

RealDiagonalVectorOperator = NDArray[Shape['A, B'], Float64]
ComplexDiagonalVectorOperator = NDArray[Shape['A, B'], Complex128]
GenericDiagonalVectorOperator = Union[RealDiagonalVectorOperator, ComplexDiagonalVectorOperator]

# for surface hopping
ActiveSurface = NDArray[Shape['1'], Int64]

# for floquet
BlockFloquetOperator = NDArray[Shape['M, M, N, N'], Complex128]

# for typing generic scalar types
AnyNumber: TypeAlias = Union[int, float, complex]
RealNumber: TypeAlias = Union[int, float]