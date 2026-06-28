#!/usr/bin/env python3

### meshes.py

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Tuple

# spatial meshes
@dataclass
class SpaceMeshes:
    # spatial mesh for x1
    x1_min: float
    x1_max: float
    x1: NDArray
    dx1: float
    N_x1: int

    # spatial mesh for x2
    x2_min: float
    x2_max: float
    x2: NDArray
    dx2: float
    N_x2: int


## spatial mesh creation
def spatial_mesh(x_limits:Tuple[float, float], y_limits:Tuple[float, float], N_x:int, N_y:int) -> SpaceMeshes:
    x1_min, x1_max = x_limits
    x2_min, x2_max = y_limits
    # x and y 1D meshes
    x1, dx1 = np.linspace(x1_min, x1_max, num=N_x, retstep=True)
    x2, dx2 = np.linspace(x2_min, x2_max, num=N_y, retstep=True)
   
    return SpaceMeshes(x1_min=x1_min, x1_max=x1_max, x1=x1, dx1=float(dx1), N_x1=x1.shape[0],
                        x2_min=x2_min, x2_max=x2_max, x2=x2, dx2=float(dx2), N_x2=x2.shape[0])


# age meshes
@dataclass
class AgeMeshes:
    # age mesh for juveniles
    a_J_min: float
    a_J_max: float
    a_J: NDArray
    da_J: float
    N_a_J: int

    # age mesh for adults
    a_A_min: float
    a_A_max: float
    a_A: NDArray
    da_A: float
    N_a_A: int

def age_mesh(a_limits:Tuple[float, float, float], N_a:int) -> AgeMeshes:

    a_min, a_T, a_max = a_limits
    a_tmp, da = np.linspace(a_min, a_max, N_a, endpoint=True, retstep=True)
    a_juvenile = np.extract(a_tmp < a_T, a_tmp)
    a_adult = np.extract(a_tmp >= a_T, a_tmp) # a_T is the adult age
    N_a_1 = a_juvenile.shape[0] # age dimension - juveniles
    N_a_2 = a_adult.shape[0] # age dimension - adults
    del(a_tmp)

    return AgeMeshes(a_J_min=a_min, a_J_max=a_T, a_J=a_juvenile, da_J=float(da), N_a_J=N_a_1,
                     a_A_min=a_T, a_A_max=a_max, a_A=a_adult, da_A=float(da), N_a_A=N_a_2)

# 2D meshgrid creation
def meshgrid_2D(x1:NDArray, x2:NDArray) -> Tuple[NDArray, NDArray]:
    xx1, xx2 = np.meshgrid(x1, x2, indexing='ij')
    return xx1, xx2

# 3D meshgrid creation
def meshgrids_3D(age_mesh:NDArray, x1:NDArray, x2:NDArray) -> Tuple[NDArray, NDArray, NDArray]:

    aa, xx1, xx2 = np.meshgrid(age_mesh, x1, x2, indexing='ij')
    return aa, xx1, xx2