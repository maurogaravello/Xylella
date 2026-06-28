"""
Finite Volume Methods for solving conservation laws.

This module provides implementations of 1-D finite volume methods
with various numerical flux functions.
"""

import numpy as np
from numpy.typing import NDArray


def lax_friedrichs_for_x(augmented_state: NDArray, 
                         augmented_velocity_x: NDArray, 
                         x_mesh_size:float,
                         alpha: float) -> NDArray:
    """
    Lax-Friedrichs numerical flux for the x-direction.
    
    
    Parameters
    ----------
    augmented_state : NDArray (3D: a, x, y). 
                      The first and last line for x are the boundary conditions.
    augmented_velocity_x : NDArray (2D: x, y). 
                      The first and last line for x are the boundary conditions.
    x_mesh_size : float
        The mesh size in x.
    alpha : float
        The numerical flux parameter.
    
    Returns
    -------
    NDArray
        Numerical fluxes at interfaces.
    """
    first_term = augmented_velocity_x[2:, :] * augmented_state[:, 2:, :] \
        - augmented_velocity_x[:-2, :] * augmented_state[:, :-2, :]
    second_term = alpha * (augmented_state[:, 2:, :] -2 * augmented_state[:, 1:-1, :] 
                           + augmented_state[:, :-2, :])
    return (first_term - second_term) / (2*x_mesh_size)

def lax_friedrichs_for_y(augmented_state: NDArray, 
                         augmented_velocity_y: NDArray, 
                         y_mesh_size:float,
                         alpha: float) -> NDArray:
    """
    Lax-Friedrichs numerical flux for the x-direction.
    
    
    Parameters
    ----------
    augmented_state : NDArray (3D: a, x, y). 
                      The first and last line for y are the boundary conditions.
    augmented_velocity_y : NDArray (2D: x, y). 
                      The first and last line for y are the boundary conditions.
    y_mesh_size : float
        The mesh size in y.
    alpha : float
        The numerical flux parameter.
    
    Returns
    -------
    NDArray
        Numerical fluxes at interfaces.
    """
    first_term = augmented_velocity_y[:, 2:] * augmented_state[:, :, 2:] \
        - augmented_velocity_y[:, :-2] * augmented_state[:, :, :-2]
    second_term = alpha * (augmented_state[:, :, 2:] -2 * augmented_state[:, :, 1:-1] 
                           + augmented_state[:, :, :-2])
    return (first_term - second_term) / (2*y_mesh_size)


def upwind_for_age(augmented_state: NDArray, age_mesh_size:float) -> NDArray:
    """
    Upwind numerical flux for age structure.
    
    Computes the term related to 'partial_a u':
    
    Parameters
    ----------
    augmented_state : NDArray (3D: a, x, y). 
                      The first line a=0 is the boundary condition.
    age_mesh_size : float
        The mesh size in age strcture.
    
    Returns
    -------
    NDArray
        Numerical term related to partial_a u.
    """
    
    return (augmented_state[1:, :, :] - augmented_state[:-1, :, :]) / age_mesh_size

def upwind_for_x(augmented_state: NDArray, 
                 augmented_velocity_x: NDArray, 
                 x_mesh_size:float) -> NDArray:
    """
    Upwind numerical flux for x-direction.
    
    Computes the term related to 'partial_x (xi_x u)':
    
    Parameters
    ----------
    augmented_state : NDArray (3D: a, x, y). 
                      The first and last line for x are the boundary conditions.
    augmented_velocity_x : NDArray (2D: x, y). 
                      The first and last line for x are the boundary conditions.
    x_mesh_size : float
        The mesh size in x.
    
    Returns
    -------
    NDArray
        Numerical term related to partial_x (xi_x u).
    """
    first_term = augmented_velocity_x[2:, :] * augmented_state[:, 2:, :] \
        - augmented_velocity_x[:-2, :] * augmented_state[:, :-2, :]
    second_term = -0.5 * np.absolute(augmented_velocity_x[1:-1, :] + augmented_velocity_x[2:, :]) \
        * (augmented_state[:, 2:, :] - augmented_state[:, 1:-1, :])
    third_term = -0.5 * np.absolute(augmented_velocity_x[1:-1, :] + augmented_velocity_x[:-2, :]) \
        * (augmented_state[:, :-2, :] - augmented_state[:, 1:-1, :])
    
    return (first_term + second_term + third_term) / (2*x_mesh_size)
    
def upwind_for_y(augmented_state: NDArray, 
                 augmented_velocity_y: NDArray, 
                 y_mesh_size:float) -> NDArray:
    """
    Upwind numerical flux for y-direction.
    
    Computes the term related to 'partial_y (xi_y u)':
    
    Parameters
    ----------
    augmented_state : NDArray (3D: a, x, y). 
                      The first and last line for y are the boundary conditions.
    augmented_velocity_y : NDArray (2D: x, y). 
                      The first and last line for y are the boundary conditions.
    y_mesh_size : float
        The mesh size in y.
    
    Returns
    -------
    NDArray
        Numerical term related to partial_y (xi_y u).
    """
    first_term = augmented_velocity_y[:, 2:] * augmented_state[:, :, 2:] \
        - augmented_velocity_y[:, :-2] * augmented_state[:, :, :-2]
    second_term = -0.5 * np.absolute(augmented_velocity_y[:, 1:-1] + augmented_velocity_y[:, 2:]) \
        * (augmented_state[:, :, 2:] - augmented_state[:, :, 1:-1])
    third_term = -0.5 * np.absolute(augmented_velocity_y[:, 1:-1] + augmented_velocity_y[:, :-2]) \
        * (augmented_state[:, :, :-2] - augmented_state[:, :, 1:-1])
    
    return (first_term + second_term + third_term) / (2*y_mesh_size)
    
