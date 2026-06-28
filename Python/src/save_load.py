#!/usr/bin/env python3

# saving and loading procedures

import numpy as np
from multiprocessing import Process

# meglio mettere in parallelo il salvataggio...

# save the status
def save_status(dir_name, file_name:str, u:np.array, w:np.array, z:np.array, t:float):
    file_name = (dir_name / file_name).with_suffix('.npz')
    mp_save = Process(target=np.savez_compressed, args=(file_name, u, w, z, t))
    mp_save.start()

# Load status from disk
def load_status(file_name):

    file_name = file_name.with_suffix('.npz')
    npzf = np.load(file_name)
    u = npzf['arr_0']
    w = npzf['arr_1']
    z = npzf['arr_2']
    time = npzf['arr_3']
    npzf.close()
    
    return u, w, z, time

# save mass
def save_mass(dir_name, times: np.array, mass_u:np.array, mass_w:np.array, mass_z:np.array, 
              l_infty_u:np.array, l_infty_w:np.array, l_infty_z:np.array):
    
    file_name = (dir_name / 'Mass').with_suffix('.npz')
    np.savez_compressed(file_name, times, mass_u, mass_w, mass_z, l_infty_u, l_infty_w, l_infty_z)

# load mass
def load_mass(file_name):
    file_name = file_name.with_suffix('.npz')
    npzf = np.load(file_name)
    times = npzf['arr_0']
    mass_u = npzf['arr_1']
    mass_w = npzf['arr_2']
    mass_z = npzf['arr_3']
    l_infty_u = npzf['arr_4']
    l_infty_w = npzf['arr_5']
    l_infty_z = npzf['arr_6']
    npzf.close()
    
    return times, mass_u, mass_w, mass_z, l_infty_u, l_infty_w, l_infty_z