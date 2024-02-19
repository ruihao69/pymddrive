# %%
import numpy as np  
import scipy.linalg as LA

from numpy.typing import ArrayLike
from typing import Union

def benchmark_at_multiplication(ntest: int = 1):
    import timeit
    U1 = np.random.rand(2, 2)
    U2 = np.random.rand(2, 2) 
    U = U1 + 1.j * U2
    U = 0.5*(U + U.conj().T)
        
    H = np.random.rand(2, 2)
    H = 0.5*(H + H.T)
        
    print("Benchmark for matrix multiplication.") 
    for i in range(ntest):
        t = timeit.timeit(lambda: U.conj().T @ H @ U, number=100000)
        print(f"Time: {t}")
        
    print("Benchmark for np.dot.") 
    for i in range(ntest):
        t = timeit.timeit(lambda: np.dot(U.conj().T, np.dot(H,  U)), number=100000)
        print(f"Time: {t}")

    print("Benchmark for np.matmul.") 
    for i in range(ntest):
        t = timeit.timeit(lambda: np.matmul(U.conj().T, np.matmul(H,  U)), number=100000)
        print(f"Time: {t}")
   
    print("Benchmark for einsum.")
    # out = np.zeros((2, 2), dtype=U.dtype)
    for i in range(ntest):
        t = timeit.timeit(lambda: np.einsum('ij,jk,kl->il', U.conj().T, H, U), number=100000)
        print(f"Time: {t}")
            
def benchmark_at_multiplication_heap(ntest: int = 1):
    import timeit
    d = 20
    U1 = np.random.rand(d, d)
    U2 = np.random.rand(d, d) 
    U = U1 + 1.j * U2
    U = 0.5*(U + U.conj().T)
        
    H = np.random.rand(d, d)
    H = 0.5*(H + H.T)
        
    print("Benchmark for matrix multiplication.") 
    for i in range(ntest):
        t = timeit.timeit(lambda: U.conj().T @ H @ U, number=1000)
        print(f"Time: {t}")
        
    print("Benchmark for np.dot.") 
    out = np.zeros((d, d), dtype=U.dtype)
    for i in range(ntest):
        def foo(U, H):
            np.dot(H, U, out=out)
            np.dot(U.conj().T, out, out=out) 
        # t = timeit.timeit(lambda: np.dot(U.conj().T, np.dot(H,  U)), number=1000)
        t = timeit.timeit(lambda: foo(U, H), number=1000)
        print(f"Time: {t}")

    print("Benchmark for np.matmul.") 
    for i in range(ntest):
        t = timeit.timeit(lambda: np.matmul(U.conj().T, np.matmul(H,  U)), number=1000)
        print(f"Time: {t}")
   
    print("Benchmark for einsum.")
    # out = np.zeros((d, d), dtype=U.dtype)
    for i in range(ntest):
        t = timeit.timeit(lambda: np.einsum('ij,jk,kl->il', U.conj().T, H, U), number=1000)
        print(f"Time: {t}")
        
def diagonalize(H):
    return LA.eigh(H)

def diagonalize_lapack(H):
    if H.dtype == np.float64:
        return LA.lapack.dsyev(H)
    elif H.dtype == np.complex128:
        return LA.lapack.zheev(H)
    else:
        raise TypeError(f"The input type {H.dtype} is not supported at this time.")
    
def diagonalize(
    H: ArrayLike,
    use_lapack: Union[bool, None] = None,
):
    if use_lapack is None:
        if H.shape[0] > 24:
            use_lapack = False
        else:
            use_lapack = True
            
    if use_lapack:
        if H.dtype == np.float64:
            return LA.lapack.dsyev(H)[:2]
        elif H.dtype == np.complex128:
            return LA.lapack.zheev(H)[:2]
    else: 
        return LA.eigh(H)
    

def benchmark_diagonalization(ntest: int=1000):
    import timeit
    
    d = 24
    H1 = np.random.rand(d, d)
    H2 = np.random.rand(d, d)
    H = H1 + 1.j * H2
    H = 0.5*(H + H.T)
    
    print("Benchmark for LA.eigh.")
    time = timeit.timeit(lambda: diagonalize(H, False), number=ntest)
    print(f"Time: {time}")
    
    print("Benchmark for LA.lapack.")
    time = timeit.timeit(lambda: diagonalize(H, True), number=ntest)
    print(f"Time: {time}") 
    
    print("Benchmark for diagonalize.")
    time = timeit.timeit(lambda: diagonalize(H), number=ntest)
    print(f"Time: {time}")
    
    
# %%
if __name__ == "__main__":            
    # benchmark_at_multiplication()
    # benchmark_at_multiplication_heap()
    
    """ Conclusion1: """
    # With in the numpy framework, performance wise, the np.dot function performs the best when compared to the @ operator, np.matmul, and np.einsum.
    
    """ Conclusion2: """
    # For whether preallocating the output array is beneficial, the answer depends on the dimension. 
    # For small matrices, the difference is insignificant, and preallocating can undermine the performance of the code.
    # However, for large matrices, in the test case > 20, the benefit of preallocating the output array becomes significant.
    
    benchmark_diagonalization(10000)
    
    """ Conclusion3: """
    # For stack allocated small matricies, direct call to lapack is faster than the numpy wrapper.
    # However, for heap allocated large matricies, the numpy wrapper is faster than the direct call to lapack.
# %%
