import os

import numpy as np

from joblib import Parallel, delayed, cpu_count
from typing import Callable

import itertools    

def get_ncpus():
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        print("We are on a SLURM system!, using SLURM_CPUS_PER_TASK to determine ncpus.")
        ncpus = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        print("We are not on a SLURM system, using cpu_count to determine ncpus.")
        ncpus = cpu_count()

    return ncpus

def kwargs_to_tuples_of_kwargs(kwargs):
    len_list = []
    for key, val in kwargs.items():
        len_list.append(len(val))
    assert np.allclose(len_list, len_list[0]) 
    return tuple(dict(zip(kwargs, items)) for items in zip(*kwargs.values()))

class ParaRunScatter:
    def __init__(self, n_jobs=None, **kwargs):
        self.n_jobs = n_jobs if isinstance(n_jobs, int) else get_ncpus()    
        self.kwargs_tuple = kwargs_to_tuples_of_kwargs(kwargs)
        # self.r0_list = r0_list
        # self.p0_list = p0_list
        
    def run(
        self, 
        func_one_scatter: Callable,
        func_accumulate: Callable,
        sim_signature: str
    )-> None:
        if not os.path.exists(sim_signature):
            os.makedirs(sim_signature)
        
        return self._parallel_run(func_one_scatter)
        
    def _parallel_run(self, func):
        return Parallel(n_jobs=self.n_jobs, return_as='generator', verbose=5)(
            delayed(func)(**kwargs) for kwargs in self.kwargs_tuple
        )