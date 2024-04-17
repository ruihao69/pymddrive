import numpy as np
from scipy.io import netcdf_file
from joblib import Parallel, delayed

from pymddrive.utils import get_ncpus

import os
import glob
from typing import List, Tuple
from enum import Enum

def check_filenames(files: List[str]) -> bool:
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
        return True
    
def find_trajectories(project_dir: str) -> Tuple[List[str]]:
    traj_files = glob.glob(os.path.join(project_dir, "*.nc")) 
    fileflag = check_filenames(traj_files)
    if not fileflag:
        raise ValueError(f"Files do not have the correct format.")
    # return [netcdf_file(file, 'r', version=1) for file in traj_files], scatter_type
    return traj_files

def load_single_pymddrive_nc(nc_filename: str) -> Tuple[np.ndarray]:
    with open(nc_filename, 'rb') as f:
        nc = netcdf_file(f)
        t = np.array(nc.variables['time'].data)
        R = np.array(nc.variables['R'].data) 
        P = np.array(nc.variables['P'].data) 
        adiabatic_populations = np.array(nc.variables['adiabatic_populations'].data) 
        diabatic_populations = np.array(nc.variables['diabatic_populations'].data) 
        KE = np.array(nc.variables['KE'].data)  
        PE = np.array(nc.variables['PE'].data) 
    return (t, R, P, adiabatic_populations, diabatic_populations, KE, PE)

def load_all_pymddrive_nc(nc_files: List[str]) -> Tuple[np.ndarray]:
    all_data = Parallel(n_jobs=get_ncpus())(
        delayed(load_single_pymddrive_nc)(ncfile) for ncfile in nc_files
    )
    return all_data
    

def map_dynamics_to_scatter_result(R_last: float, P_last: float, adiabatic_populations_0_last: float):
    """ [RL, TL, RU, TU] """
    if (R_last<0) and (P_last<0):
        return np.array([adiabatic_populations_0_last, 0, 1-adiabatic_populations_0_last, 0])
    elif (R_last>0) and (P_last>0):
        return np.array([0, adiabatic_populations_0_last, 0, 1-adiabatic_populations_0_last])
    else:
        raise ValueError(f"Un-recogonized scattering result pattern: {R_last=}, {P_last=}.")
    
def calculate_scatter(all_data: List[Tuple[np.ndarray]]) -> np.ndarray[np.float64]:
    
    def scatter_one_traj(one_data: Tuple[np.ndarray]) -> np.ndarray:
        _, R, P, pop, _, _, _ = one_data 
        return map_dynamics_to_scatter_result(R[-1], P[-1], pop[-1][0])
    
    res = Parallel(n_jobs=get_ncpus())(
        delayed(scatter_one_traj)(ncfile) for ncfile in all_data
    )
    
    # Ensemble average
    scatter_result = np.array(res).mean(axis=0)
    return scatter_result

def get_t_longest_len(all_data: List[Tuple[np.ndarray]]) -> Tuple[int, np.ndarray]:
    def get_t_single_traj(one_data: str) -> np.ndarray:
        t, _, _, _, _, _, _ = one_data
        return t
    len_list = np.array([len(get_t_single_traj(data)) for data in all_data])
    argmax = np.argmax(len_list)
    t_longest = all_data[argmax]
    return len_list[argmax], t_longest

def ensemble_avg(all_data: List[Tuple[np.ndarray]]) -> Tuple[np.ndarray]:
    ntrajs = len(all_data)
    dim = all_data[0][3].shape[1]
    n_longest, t_longest = get_t_longest_len(all_data)
    
    def get_output_array(n_longest: int, dim: int) -> Tuple[np.ndarray]:
        R_out = np.zeros((n_longest, 1))
        P_out = np.zeros((n_longest, 1))
        apop_out = np.zeros((n_longest, dim))
        dpop_out = np.zeros((n_longest, dim))
        KE_out = np.zeros((n_longest, 1))
        PE_out = np.zeros((n_longest, 1))
        return R_out, P_out, apop_out, dpop_out, KE_out, PE_out
    
    def fill_R_or_P_one_traj(X_out: np.ndarray, X: np.ndarray) -> None:
        X_out[:X.shape[0], ...] = X
        X_out[X.shape[0]:, ...] = np.nan
        
    def fill_pop_or_E_one_traj(X_out: np.ndarray, X: np.ndarray) -> None:
        X_out[:X.shape[0], :] = X
        X_out[X.shape[0]:, :] = X[-1]
    
    def load_one_ensemble(one_data: Tuple[np.ndarray]) -> Tuple[np.ndarray]:
        _, R, P, apop, dpop, KE, PE = one_data
        R_out, P_out, apop_out, dpop_out, KE_out, PE_out = get_output_array(n_longest, dim)
        fill_R_or_P_one_traj(R_out, R)
        fill_R_or_P_one_traj(P_out, P)
        fill_pop_or_E_one_traj(apop_out, apop)
        fill_pop_or_E_one_traj(dpop_out, dpop)
        fill_pop_or_E_one_traj(KE_out, KE)
        fill_pop_or_E_one_traj(PE_out, PE)
        return R_out, P_out, apop_out, dpop_out, KE_out, PE_out
    
    ensembles = Parallel(n_jobs=get_ncpus())(
        delayed(load_one_ensemble)(data) for data in all_data
    )
        
    
    # R_avg = np.nanmean(r_ensemble, axis=0)
    # P_avg = np.nanmean(p_ensemble, axis=0)    
    # a_pop_avg = a_pop_ensemble.mean(axis=0)
    # d_pop_avg = d_pop_ensemble.mean(axis=0)
    # ke_avg = ke_ensemble.mean(axis=0)
    # pe_avg = pe_ensemble.mean(axis=0)   
    # return t_longest, R_avg, P_avg, a_pop_avg, d_pop_avg, ke_avg, pe_avg

def post_process(project_dir: str) -> Tuple[np.ndarray]:
    traj_ncfiles = find_trajectories(project_dir)
    all_data = load_all_pymddrive_nc(traj_ncfiles)
    
    # compute scatter result
    scatter_result = calculate_scatter(all_data)
    
    # compute ensemble average
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
    
    
    
    