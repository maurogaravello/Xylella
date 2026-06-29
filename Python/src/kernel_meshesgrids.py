#!/usr/bin/env python3

### kernel_meshesgrids.py

import numpy as np
from numpy.typing import NDArray
from typing import Tuple

# 2D meshgrid creation for kernels
def create_2Dmesh_kernels(x1_kernel_limits:Tuple[float, float], x2_kernel_limits:Tuple[float, float], 
                          dx1:float, dx2:float) -> Tuple[NDArray, NDArray]:
    
    xx1, xx2 = np.meshgrid(create_1Dmesh_kernels(x1_kernel_limits[0], x1_kernel_limits[1], dx1),
                           create_1Dmesh_kernels(x2_kernel_limits[0], x2_kernel_limits[1], dx2),
                           indexing='ij')
    return xx1, xx2


# 1D mesh for kernels
def create_1Dmesh_kernels(a:float, b:float, mesh_size:float) -> NDArray:
    """
    Create the 1D mesh for the kernel functions

    Parameters
    ----------
    a : float
        starting point from left
    b : float
        ending point from right
    mesh_size : float
        step size

    Returns
    -------
    np.array
        points in the mesh
    """
    x1 = np.arange(0, b, mesh_size)
    x2 = np.arange(-mesh_size, a, -mesh_size)
    res = np.concatenate((np.flip(x2), x1))

    return res