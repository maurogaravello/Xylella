#!/usr/bin/env python3

# saving and loading procedures
from pathlib import Path
from numpy.typing import NDArray
from typing import Tuple
import numpy as np
from multiprocessing import Process

# meglio mettere in parallelo il salvataggio...

# save the status
def save_status(dir_name:Path, file_name:str, u:NDArray, w:NDArray, z:NDArray, t:float) -> None:
    """
    Save simulation state to compressed NumPy format.
    
    Parameters
    ----------
    dir_name : Path
        Output directory path
    file_name : str
        Base filename (without extension)
    u : NDArray
        Juvenile population distribution
    w : NDArray
        Adult population distribution
    z : NDArray
        Tree density distribution
    t : float
        Current simulation time
    """

    file_path = (dir_name / file_name).with_suffix('.npz')
    mp_save = Process(target=np.savez_compressed, args=(file_path, u, w, z, t))
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


def save_mass(dir_name: Path, times: NDArray, mass_u: NDArray, mass_w: NDArray, mass_z: NDArray,
              l_infty_u: NDArray, l_infty_w: NDArray, l_infty_z: NDArray) -> None:
    """Save conservation metrics to disk."""
    file_path = (dir_name / 'Mass').with_suffix('.npz')
    np.savez_compressed(
        file_path,
        times=times,
        mass_u=mass_u,
        mass_w=mass_w,
        mass_z=mass_z,
        l_infty_u=l_infty_u,
        l_infty_w=l_infty_w,
        l_infty_z=l_infty_z
    )

def load_mass(file_name: Path) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Load conservation metrics from disk."""
    file_path = file_name.with_suffix('.npz')
    with np.load(file_path) as data:
        return (
            data['times'],
            data['mass_u'],
            data['mass_w'],
            data['mass_z'],
            data['l_infty_u'],
            data['l_infty_w'],
            data['l_infty_z']
        )

