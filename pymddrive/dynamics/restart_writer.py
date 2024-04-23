# %%
import numpy as np
from netCDF4 import Dataset

import tempfile
import shutil

class RestartWriter:
    def __init__(self, dim_elec: int, dim_nucl: int) -> None:
        # create a randomized filename
        self.fn = tempfile.NamedTemporaryFile().name
        self.nc = Dataset(self.fn, 'w')
        
        # complex128 type 
        self.complex128 = np.dtype([("real",np.float64),("imag",np.float64)])
        self.complex128_t = self.nc.createCompoundType(self.complex128, "complex128")
        
        # create dimensions
        self.frame = self.nc.createDimension('frame', None)
        self.spatial = self.nc.createDimension('spatial', dim_nucl)
        self.electronic = self.nc.createDimension('electronic', dim_elec)
        
        # create data variables
        self.time = self.nc.createVariable('time', 'f8', ('frame',))
        self.position = self.nc.createVariable('R', 'f8', ('frame', 'spatial'))
        self.momentum = self.nc.createVariable('P', 'f8', ('frame', 'spatial'))
        self.density_matrix = self.nc.createVariable('rho', self.complex128_t, ('frame', 'electronic', 'electronic'))
        
    def write_frame(self, t, R, P, rho):
        # append data at iframe
        iframe = self.frame.size
        rho_composite = np.zeros((rho.shape[0], rho.shape[1]), dtype=self.complex128)
        rho_composite['real'] = np.real(rho)
        rho_composite['imag'] = np.imag(rho)
        self.time[iframe] = t
        self.position[iframe] = R
        self.momentum[iframe] = P
        self.density_matrix[iframe] = rho_composite
        
    def close(self):
        self.nc.close()
        
    def save(self, target_fn: str):
        self.close()
        shutil.copy(self.fn, target_fn)
        
        
        
        
        