# %%
from netCDF4 import Dataset

import tempfile
import shutil

class PropertiesWriter:
    def __init__(self, dim_elec: int, dim_nucl: int, **attrs) -> None:
        # create a randomized filename
        self.fn = tempfile.NamedTemporaryFile().name
        # create a netCDF file
        self.nc = Dataset(self.fn, 'a', format='NETCDF4')
        
        # create groups 
        self.data = self.nc.createGroup('data')
        
        # create dimensions
        self.frame = self.nc.createDimension('frame', None)
        self.spatial = self.nc.createDimension('spatial', dim_nucl)
        self.electronic = self.nc.createDimension('electronic', dim_elec)
        
        # create data variables
        self.time = self.nc.createVariable('time', 'f8', ('frame',))
        self.position = self.nc.createVariable('R', 'f8', ('frame', 'spatial'))
        self.momentum = self.nc.createVariable('P', 'f8', ('frame', 'spatial'))
        self.adiabatic_populations = self.nc.createVariable('adiabatic_populations', 'f8', ('frame', 'electronic'))
        self.diabatic_populations = self.nc.createVariable('diabatic_populations', 'f8', ('frame', 'electronic'))
        self.kinetic_energy = self.nc.createVariable('KE', 'f8', ('frame',))    
        self.potential_energy = self.nc.createVariable('PE', 'f8', ('frame',))
        
        # create attributes
        for key, value in attrs.items():
            self.nc.setncattr(key, value)
    
    def write_frame(self, t, R, P, adiabatic_populations, diabatic_populations, KE, PE):
        # append data at iframe
        iframe = self.frame.size
        self.time[iframe] = t
        self.position[iframe] = R
        self.momentum[iframe] = P
        self.adiabatic_populations[iframe] = adiabatic_populations
        self.diabatic_populations[iframe] = diabatic_populations
        self.kinetic_energy[iframe] = KE
        self.potential_energy[iframe] = PE
        
    def close(self):
        self.nc.close()
        
    def save(self, target_fn: str):
        self.close()
        shutil.copy(self.fn, target_fn)
        
def test(): 
    import numpy as np
    
    dim_nuclear = 1
    dim_electronic = 2
    
    t = 0.0 
    
    writer = PropertiesWriter(dim_electronic, dim_nuclear, description='test')
    
    for _ in range(1000):
        R = np.random.normal(size=(dim_nuclear, ))
        P = np.random.normal(size=(dim_nuclear, ))
        populations = np.array([np.sin(t + np.random.rand() * 0.3), np.cos(t + np.random.rand() * 0.1)])
        KE = np.random.rand()
        PE = np.random.rand()
        
        # append data at iframe
        writer.write_frame(t, R, P, populations, KE, PE)
        
        t += 1 * 0.01
    writer.save('test.nc')
    
    # read the file
    nc_read = Dataset('test.nc', 'r')
    time = np.array(nc_read.variables['time'])
    populations = np.array(nc_read.variables['populations'])
    
    import matplotlib.pyplot as plt
    
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    
    ax.plot(time, populations[:, 0], label='pop0')
    ax.plot(time, populations[:, 1], label='pop1')
    ax.legend()
    ax.set_xlabel('time')
    ax.set_ylabel('populations')
    
    plt.show()
    print(nc_read.description)
    
    

# %%
if __name__ == "__main__":
    test()
        
# %%
