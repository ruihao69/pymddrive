import numpy as np
from numpy.typing import ArrayLike
from netCDF4 import Dataset 

import os
import logging
from typing import Union
from multiprocessing import cpu_count

# Special arries
def zeros(t: Union[float, complex, ArrayLike]) -> Union[float, complex, ArrayLike]:
    if isinstance(t, float) or isinstance(t, complex):
        return 0
    elif isinstance(t, np.ndarray):
        return np.zeros(t.shape, dtype=t.dtype)
    else:
        raise TypeError(f"The input type is {type(t)}, which is not supported at this time.")
    
# Special matrices
def is_symmetric_matrix(H: ArrayLike):
    if not np.allclose(H, H.T):
        return False
    
    return True

def is_hermitian_matrix(H: ArrayLike):
    if not np.allclose(H, H.T.conj()):
        return False
    
    return True

def is_real_symmetric_matrix(H: ArrayLike):
    return np.all(np.isreal(H)) and is_symmetric_matrix(H)

# parallel processing
def get_ncpus():
    logger = logging.getLogger(__name__)
    
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        logger.info("We are on a SLURM system!, using SLURM_CPUS_PER_TASK to determine ncpus.")
        ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        logger.info("We are not on a SLURM system, using cpu_count to determine ncpus.")
        ncpus = cpu_count()

    return ncpus

def is_empty_ncdf(nc_filename: str) -> bool: 
    nc = Dataset(nc_filename, 'r')
    if len(nc.variables) == 0:
        return True
    else:
        vars_ = nc.variables
        if all([len(vars_[var]) == 0 for var in vars_]):
            return True
        else:
            return False
    