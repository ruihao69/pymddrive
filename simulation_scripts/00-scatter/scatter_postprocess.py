import numpy as np
from scipy.io import netcdf_file

import os
import glob
from typing import List, Tuple
from enum import Enum

class ScatterOutputType(Enum):
    FSSH = 1
    EHRENFEST = 2
    

def load_single_pymddrive_nc(nc: netcdf_file) -> Tuple[np.ndarray]:
    t = np.array(nc.variables['time'].data)
    R = np.array(nc.variables['R'].data) 
    P = np.array(nc.variables['P'].data) 
    adiabatic_populations = np.array(nc.variables['adiabatic_populations'].data) 
    diabatic_populations = np.array(nc.variables['diabatic_populations'].data) 
    KE = np.array(nc.variables['KE'].data)  
    PE = np.array(nc.variables['PE'].data) 
    return (t, R, P, adiabatic_populations, diabatic_populations, KE, PE)

def check_filenames(files: List[str]) -> Tuple[bool, ScatterOutputType]:
    """The files should be of the form '*.*.nc'. Where the first field is the description, and the second field is the ensemble index.

    Args:
        files (List[str]): List of file names

    Returns:
        bool: True if all the files are only differing in the second field.
    """
    first_field = set()
    for file in files:
        fields = os.path.basename(file).split(".")
        if len(fields) != 3:
            raise ValueError(f"File name {file} does not have the correct format: '*.*.nc'.")
        if fields[2] != "nc":
            raise ValueError(f"File name {file} does not have the correct format: '*.*.nc'.")
        first_field.add(fields[0])
    if len(first_field) > 1:
        raise ValueError(f"Multiple first fields found: {first_field}.")
    else:
        scatter_type = ScatterOutputType.FSSH if "fssh" in first_field.pop() else ScatterOutputType.EHRENFEST
        return True, scatter_type
        

def load_trajectories(project_dir: str) -> Tuple[List[netcdf_file], ScatterOutputType]:
    traj_files = glob.glob(os.path.join(project_dir, "*.nc")) 
    fileflag, scatter_type = check_filenames(traj_files)
    if not fileflag:
        raise ValueError(f"Files do not have the correct format.")
    return [netcdf_file(file, 'r', version=1) for file in traj_files], scatter_type
    

def map_dynamics_to_scatter_result_fssh(R_last: float, P_last: float, adiabatic_populations_0_last: float):
    """ [RL, TL, RU, TU] """
    if (R_last<0) and (P_last<0):
        if adiabatic_populations_0_last>0.5:
            return np.array([1, 0, 0, 0])
        else:
            return np.array([0, 0, 1, 0])
    elif (R_last>0) and (P_last>0):
        if adiabatic_populations_0_last>0.5:
            return np.array([0, 1, 0, 0])
        else:
            return np.array([0, 0, 0, 1])
    else:
        raise ValueError(f"Un-recogonized scattering result pattern: {R_last=}, {P_last=}.")
    
def map_dynamics_to_scatter_result_ehrenfest(R_last: float, P_last: float, adiabatic_populations_0_last: float):
    """ [RL, TL, RU, TU] """
    if (R_last<0) and (P_last<0):
        return np.array([adiabatic_populations_0_last, 0, 1-adiabatic_populations_0_last, 0])
    elif (R_last>0) and (P_last>0):
        return np.array([0, adiabatic_populations_0_last, 0, 1-adiabatic_populations_0_last])
    else:
        raise ValueError(f"Un-recogonized scattering result pattern: {R_last=}, {P_last=}.")
    
def calculate_scatter(traj_ncfiles: List[netcdf_file], scatter_type: ScatterOutputType) -> np.ndarray[np.float64]:
    ntrajs = len(traj_ncfiles)
    res = np.zeros((ntrajs, 4))
    map_dynamics_to_scatter_result = map_dynamics_to_scatter_result_ehrenfest if scatter_type == ScatterOutputType.EHRENFEST else map_dynamics_to_scatter_result_fssh
    for ii, ncfile in enumerate(traj_ncfiles):
        _, R, P, pop, _, _, _ = load_single_pymddrive_nc(ncfile)
        res[ii, :] = map_dynamics_to_scatter_result(R[-1], P[-1], pop[-1][0])
        
    # Ensemble average
    scatter_result = res.mean(axis=0)
    return scatter_result

def ensemble_avg(traj_ncfiles) -> Tuple[np.ndarray]:
    
    # get_populations_list
    ntrajs = len(traj_ncfiles)
    t_longest = None
    adiabatic_populations_list = []
    diabatic_populations_list = []
    KE_list = []
    PE_list = []
    R_list = []
    P_list = []
    
    # load data and append
    for ncfile in traj_ncfiles:
        t, R, P, apop, dpop, KE, PE = load_single_pymddrive_nc(ncfile)
        adiabatic_populations_list.append(apop)
        diabatic_populations_list.append(dpop)
        KE_list.append(KE)
        PE_list.append(PE)
        R_list.append(R)
        P_list.append(P)
        if t_longest is None:
            t_longest = np.copy(t)
        else:
            if len(t) > len(t_longest):
                t_longest = np.copy(t)
                
    # create result array
    a_pop_ensemble = np.zeros((ntrajs, t_longest.shape[0], adiabatic_populations_list[0].shape[1]))
    d_pop_ensemble = np.zeros((ntrajs, t_longest.shape[0], adiabatic_populations_list[0].shape[1]))
    ke_ensemble = np.zeros((ntrajs, t_longest.shape[0]))    
    pe_ensemble = np.zeros((ntrajs, t_longest.shape[0]))
    r_ensemble = np.zeros((ntrajs, t_longest.shape[0]))
    p_ensemble = np.zeros((ntrajs, t_longest.shape[0]))
    
    for ii, (a_pop, d_pop) in enumerate(zip(adiabatic_populations_list, diabatic_populations_list)):
        len_ = len(a_pop)
        a_pop_ensemble[ii, :len_, :] = a_pop
        a_pop_ensemble[ii, len_:, :] = a_pop[-1]
        d_pop_ensemble[ii, :len_, :] = d_pop
        d_pop_ensemble[ii, len_:, :] = d_pop[-1]
        ke_ensemble[ii, :len_] = KE_list[ii]
        ke_ensemble[ii, len_:] = KE_list[ii][-1]
        pe_ensemble[ii, :len_] = PE_list[ii]
        pe_ensemble[ii, len_:] = PE_list[ii][-1]
        r_ensemble[ii, :len_] = R_list[ii][:, 0] # caveat: 1D nuclei
        r_ensemble[ii, len_:] = np.nan
        p_ensemble[ii, :len_] = P_list[ii][:, 0] # caveat: 1D nuclei
        p_ensemble[ii, len_:] = np.nan
    
    R_avg = np.nanmean(r_ensemble, axis=0)
    P_avg = np.nanmean(p_ensemble, axis=0)    
    a_pop_avg = a_pop_ensemble.mean(axis=0)
    d_pop_avg = d_pop_ensemble.mean(axis=0)
    ke_avg = ke_ensemble.mean(axis=0)
    pe_avg = pe_ensemble.mean(axis=0)   
    return t_longest, R_avg, P_avg, a_pop_avg, d_pop_avg, ke_avg, pe_avg

def post_process(project_dir: str) -> Tuple[np.ndarray]:
    traj_ncfiles, scatter_type = load_trajectories(project_dir)
    scatter_result = calculate_scatter(traj_ncfiles, scatter_type)
    t_longest, R_avg, P_avg, a_pop_avg, d_pop_avg, ke_avg, pe_avg = ensemble_avg(traj_ncfiles)
    
    scatter_header = "{:>10s} {:>12s} {:>12s} {:>12s}".format("RL", "TL", "RU", "TU")
    scatter_outfile = os.path.join(project_dir, "scatter_result.dat")
    np.savetxt(scatter_outfile, scatter_result.reshape(1, -1), header=scatter_header, fmt="%12.6f")
    
    dim = a_pop_avg.shape[1] 
    traj_header_list = ["time", "R", "P"] + [f"adiabatic_{i}" for i in range(dim)] + [f"diabatic_{i}" for i in range(dim)] + ["KE", "PE"]
    traj_header = ""
    for ii, header in enumerate(traj_header_list):
        if ii == 0:
            traj_header += "{:>10s} ".format(header)
        else:
            traj_header += "{:>12s} ".format(header)
    traj_outfile = os.path.join(project_dir, "traj.dat")
    np.savetxt(traj_outfile, np.column_stack((t_longest, R_avg, P_avg, a_pop_avg, d_pop_avg, ke_avg, pe_avg)), header=traj_header, fmt="%12.6f")
    
if __name__ == "__main__":
    project_dir = "./"
    post_process(project_dir)
    
    
    
    