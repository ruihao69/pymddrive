from pymddrive.my_types import GenericOperator, GenericVectorOperator
from pymddrive.low_level.floquet import get_HF_cos, get_dHF_dR_cos, get_HF_sin, get_dHF_dR_sin
from pymddrive.models.floquet.floquet_types import FloquetType

HF_function_table = {
    FloquetType.COSINE: get_HF_cos,
    FloquetType.SINE: get_HF_sin,
}

dHF_dR_function_table = {
    FloquetType.COSINE: get_dHF_dR_cos,
    FloquetType.SINE: get_dHF_dR_sin,
}

def get_HF(floquet_type: FloquetType, H0: GenericOperator, H1: GenericOperator, Omega: float, NF: int):
    try:
        return HF_function_table[floquet_type](H0, H1, Omega, NF)
    except KeyError:
        raise NotImplementedError(f"The Floquet type {floquet_type} is not implemented yet.")
    
def get_dHF_dR(floquet_type: FloquetType, dH0dR: GenericVectorOperator, dH1dR: GenericVectorOperator, NF: int):
    try:
        return dHF_dR_function_table[floquet_type](dH0dR, dH1dR, NF)
    except KeyError:
        raise NotImplementedError(f"The Floquet type {floquet_type} is not implemented yet.")

