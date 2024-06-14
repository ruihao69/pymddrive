# %%
import numpy as np
from numba import njit

from pymddrive.my_types import RealVector, ComplexVector, ComplexOperator, GenericOperator, GenericVectorOperator, GenericVector

from typing import Union
# from multiprocessing import Manager

# create a manager to make global tables accessible to all processes
# manager = Manager()

# Expected values implementation
def expected_value_diagonal_operator_wavefunction(operator: GenericVector, wavefunction: ComplexVector) -> float:
    return np.sum(wavefunction.conj() * operator * wavefunction).real

def expected_value_diagonal_operator_density_matrix(operator: GenericVector, density_matrix: ComplexOperator) -> float:
    return np.sum(np.diag(density_matrix) * operator).real

def expected_value_operator_wavefunction(operator: GenericOperator, wavefunction: ComplexVector) -> float:
    return np.dot(wavefunction.conj(), operator.dot(wavefunction)).real

def expected_value_operator_density_matrix(operator: GenericOperator, density_matrix: ComplexOperator) -> float:
    return np.trace(np.dot(density_matrix, operator)).real

@njit
def expected_value_vector_operator_wavefunction(v_operator: GenericVectorOperator, wavefunction: ComplexVector) -> RealVector:
    # note numpy functions like dots are faster on contiguous arrays
    # hence we take the hustle to copy the operator to a temporary contiguous array
    _op = np.zeros((v_operator.shape[0], v_operator.shape[1]), dtype=np.complex128)
    result = np.zeros((v_operator.shape[2],), dtype=np.float64)
    for inuc in range(v_operator.shape[2]):
        _op[:] = np.ascontiguousarray(v_operator[..., inuc])
        result[inuc] = np.dot(wavefunction.conj(), np.dot(_op, wavefunction)).real
    return result

@njit
def expected_value_vector_operator_density_matrix(v_operator: GenericVectorOperator, density_matrix: ComplexOperator) -> RealVector:
    # note numpy functions like dots are faster on contiguous arrays
    # hence we take the hustle to copy the operator to a temporary contiguous array
    _op = np.zeros((v_operator.shape[0], v_operator.shape[1]), dtype=np.complex128)
    result = np.zeros((v_operator.shape[2],), dtype=np.float64)
    for inuc in range(v_operator.shape[2]):
        _op[:] = np.ascontiguousarray(v_operator[..., inuc])
        result[inuc] = np.trace(np.dot(density_matrix, _op)).real
    return result

# generic expected value function

# define a table of expected value functions
EXPECTED_VALUE_FUNCTION_TABLE = {
    # (operator_dim, quantum_state_dim): expected_value_function
    (1, 1): expected_value_diagonal_operator_wavefunction,
    (1, 2): expected_value_diagonal_operator_density_matrix,
    (2, 1): expected_value_operator_wavefunction,
    (2, 2): expected_value_operator_density_matrix,
    (3, 1): expected_value_vector_operator_wavefunction,
    (3, 2): expected_value_vector_operator_density_matrix,
}


def expected_value(
    operator: Union[GenericVector, GenericOperator, GenericVectorOperator],
    quantum_state: Union[ComplexVector, ComplexOperator],
):
    operator_dim = operator.ndim
    quantum_state_dim = quantum_state.ndim
    # get the expected value function from the table
    try:
        return EXPECTED_VALUE_FUNCTION_TABLE[(operator_dim, quantum_state_dim)](operator, quantum_state)
    except KeyError:
        raise ValueError(f"Unsupported operator and quantum state dimensions: {operator_dim}, {quantum_state_dim}")

    
# The equations of motion for the electronic degrees of freedom  
# i.e., the Schrödinger equation for the wavefunction or the von Neumann equation for the density matrix
def commutator(A: GenericOperator, B: GenericOperator) -> GenericOperator:
    return np.dot(A, B) - np.dot(B, A)

@njit
def commutator_diagA_B(A: RealVector, B: GenericOperator) -> GenericOperator:
    out = np.zeros_like(B)
    for ii in range(B.shape[0]):
        for jj in range(B.shape[1]):
            out[ii, jj] = 0.0 if ii == jj else (A[ii] - A[jj]) * B[ii, jj]
    return out

def schrodinger_diabatic(psi: ComplexVector, H: GenericOperator) -> ComplexVector:
    return -1j * np.dot(H, psi)

def von_neumann_diabatic(rho: ComplexOperator, H: GenericOperator) -> ComplexOperator:
    return -1j * commutator(H, rho)

def shrodinger_adiabatic(psi: ComplexVector, evals: RealVector, v_dot_d: GenericOperator) -> ComplexVector:
    return -1.j * np.multiply(evals, psi) - np.dot(v_dot_d, psi)

def von_neumann_adiabatic(rho: ComplexOperator, evals: RealVector, v_dot_d: GenericOperator) -> ComplexOperator:
    return -1.j * commutator_diagA_B(evals, rho) - commutator(v_dot_d, rho)
    # return -1.j * commutator(np.diagflat(evals), rho) - commutator(v_dot_d, rho)

DIABATIC_EQUATIONS = {
    1: schrodinger_diabatic,
    2: von_neumann_diabatic,
}

ADIABATIC_EQUATIONS = {
    1: shrodinger_adiabatic,
    2: von_neumann_adiabatic,
}

def diabatic_equations_of_motion(
    quantum_state: Union[ComplexVector, ComplexOperator],
    hamiltonian: GenericOperator,
) -> Union[ComplexVector, ComplexOperator]:
    try: 
        return DIABATIC_EQUATIONS[quantum_state.ndim](quantum_state, hamiltonian)
    except KeyError:
        raise ValueError(f"Unsupported quantum state dimension: {quantum_state.ndim}")
    
def adiabatic_equations_of_motion(
    quantum_state: Union[ComplexVector, ComplexOperator],
    evals: RealVector,
    v_dot_d: GenericOperator,
) -> Union[ComplexVector, ComplexOperator]:
    try: 
        return ADIABATIC_EQUATIONS[quantum_state.ndim](quantum_state, evals, v_dot_d)
    except KeyError:
        raise ValueError(f"Unsupported quantum state dimension: {quantum_state.ndim}")
    
# compuate v_dot_d
def compute_v_dot_d(v: RealVector, dc: GenericVectorOperator) -> GenericOperator:
    """compute the dot product of the nuclear velocity and the non-adiabatic coupling vector field

    Args:
        v (RealVector): velocity
        dc (GenericVectorOperator): non-adiabatic coupling vector field (derivative coupling), shape (n_electronic_states, n_electronic_states, n_nuclei)

    Returns:
        GenericOperator: \sum_i P^\alpha / M^\alpha \times dc^\alpha 
    """
    return np.tensordot(v, dc, axes=([0], [2]))

@njit
def floquet_expval_rho_2(
    O_F: GenericOperator,
    rho_F: ComplexOperator,
    Omega: float,
    t: float,
    NF: int, 
    dim: int
) -> float:
    OF_rhoF = np.dot(O_F, rho_F)
    expval_cplx: complex = 0.0
    n = 0
    for ii in range(dim):
        for m in range(-NF, NF+1):
            IN = ii + (m + NF) * dim
            I0 = ii + (n + NF) * dim
            expval_cplx += np.exp(1j*m*Omega*t) * OF_rhoF[IN, I0]
    return np.real(expval_cplx)

@njit
def floquet_expval_rho_3(
    O_F: GenericVectorOperator,
    rho_F: ComplexOperator,
    Omega: float,
    t: float,
    NF: int, 
    dim: int
) -> RealVector:
    expval = np.zeros((O_F.shape[2],), dtype=np.float64)
    O_tmp = np.zeros((O_F.shape[0], O_F.shape[1]), dtype=np.complex128)
    for inuc in range(O_F.shape[2]):
        O_tmp[:] = np.ascontiguousarray(O_F[..., inuc])
        OF_rhoF = np.dot(O_tmp, rho_F)
        expval_cplx: complex = 0.0
        n = 0
        for ii in range(dim):
            for m in range(-NF, NF+1):
                IN = ii + (m + NF) * dim
                I0 = ii + (n + NF) * dim
                expval_cplx += np.exp(1j*m*Omega*t) * OF_rhoF[IN, I0]
        expval[inuc] = np.real(expval_cplx)
    return expval

def floquet_expval_rho(
    O_F: Union[GenericOperator, GenericVectorOperator],
    rho_F: ComplexOperator,
    Omega: float,
    t: float,
    NF: int, 
    dim: int
):
    if O_F.ndim == 2:
        return floquet_expval_rho_2(O_F, rho_F, Omega, t, NF, dim)
    elif O_F.ndim == 3:
        return floquet_expval_rho_3(O_F, rho_F, Omega, t, NF, dim)
    else:
        raise ValueError(f"Unsupported operator dimension: {O_F.ndim}")
    
    

   

# %%
# testing code
def get_random_O(dim: int, is_complex=True) -> GenericOperator:
    op = np.random.rand(dim, dim)
    if is_complex:
        op = op + 1j*np.random.rand(dim, dim)
    op = op + op.T.conj()
    return op

def get_random_vO(dim: int, n_nuclei: int, is_complex=True) -> GenericVectorOperator:
    return np.array([get_random_O(dim, is_complex) for _ in range(n_nuclei)]).transpose(1, 2, 0)

def get_random_rho(dim: int) -> ComplexVector:
    rho = np.zeros((dim, dim), dtype=np.complex128)
    rho[:] += np.random.rand(dim, dim) + 1j*np.random.rand(dim, dim)
    rho = rho + rho.T.conj()
    rho = rho / np.trace(rho) 
    return rho

def get_random_psi(dim: int) -> ComplexVector:
    psi = np.zeros(dim, dtype=np.complex128)
    psi[:] += np.random.rand(dim) + 1j*np.random.rand(dim)
    psi = psi / np.linalg.norm(psi)
    return psi

def main():
    ne, nu = 2, 3
    
    # rho = get_random_rho(ne)
    psi = get_random_psi(ne)
    rho = np.outer(psi, psi.conj())
    d_op = np.random.rand(ne) + 1.j*np.random.rand(ne)
    op = get_random_O(ne)
    v_op = get_random_vO(ne, nu)
    
    # test expected_value
    print(expected_value(d_op, psi))
    print(expected_value(d_op, rho))
    print(expected_value(op, psi))
    print(expected_value(op, rho))
    print(expected_value(v_op, psi))
    print(expected_value(v_op, rho))
    
    E = np.random.rand(ne)
    comm1 = commutator_diagA_B(E, op)
    comm2 = commutator(np.diagflat(E), op)
    print(f"{np.allclose(comm1, comm2)=}")
    
    v = np.random.normal(0, 1, size=nu)
    v_dot_d1 = compute_v_dot_d(v, v_op)
    v_dot_d2 = np.zeros((ne, ne), dtype=np.complex128)
    for kk in range(nu):
        v_dot_d2[:] += v[kk] * v_op[..., kk]
        
    print(f"{np.allclose(v_dot_d1, v_dot_d2)=}")
    
    
    
    
if __name__ == '__main__':
    main()


# %%
