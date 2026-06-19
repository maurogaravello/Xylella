"""
Finite Volume Methods for solving conservation laws.

This module provides implementations of 1-D finite volume methods
with various numerical flux functions.
"""

import numpy as np
from typing import Callable
from numpy.typing import NDArray

def finite_volume_1d(
    u0_augmented: NDArray,
    flux_fn: Callable[[NDArray, NDArray], NDArray],
    dt: float,
    dx: float,
) -> NDArray:
    """
    1-D finite volume method with generic flux function.
    
    Solves the conservation law: u_t + f(u)_x = 0
    using the finite volume method with forward Euler time stepping.
    
    Parameters
    ----------
    u0_augmented : NDArray
        Initial condition (cell averages).
    flux_fn : Callable[[NDArray, NDArray], NDArray]
        Numerical flux function that takes left and right states and returns fluxes.
        Signature: flux_fn(u_left, u_right) -> fluxes at cell interfaces.
    dt : float
        Time step size.
    dx : float
        Cell width.
    
    Returns
    -------
    u : NDArray
        Solution at time t_final. NOT AUGMENTED!
    """
    
        
    # Compute fluxes at cell interfaces
    # u[i-1/2] left state: u[i-1], u[i-1/2] right state: u[i]
    n = len(u)
    
    
    # Compute fluxes at all interfaces
    u_left = u0_augmented[:-1]   # States to the left of interfaces
    u_right = u0_augmented[1:]   # States to the right of interfaces
    fluxes = flux_fn(u_left, u_right)
    
    # Update cell averages
    u_new = u0_augmented[1:-1] - (dt / dx) * (fluxes[1:n+1] - fluxes[:-n])
    
    u = u_new
    
    return u


def lax_friedrichs_flux(
    u_left: NDArray,
    u_right: NDArray,
    flux_physical: Callable[[NDArray], NDArray],
    wave_speed: float,
) -> NDArray:
    """
    Lax-Friedrichs numerical flux.
    
    Computes the Lax-Friedrichs flux: 
    F(u_L, u_R) = (f(u_L) + f(u_R))/2 - α(u_R - u_L)/2
    
    where α is an estimate of the maximum wave speed.
    
    Parameters
    ----------
    u_left : NDArray
        Left state(s) at cell interfaces.
    u_right : NDArray
        Right state(s) at cell interfaces.
    flux_physical : Callable[[NDArray], NDArray]
        Physical flux function f(u).
    wave_speed : float
        Maximum wave speed (viscosity parameter).
    
    Returns
    -------
    fluxes : NDArray
        Numerical fluxes at interfaces.
    """
    f_left = flux_physical(u_left)
    f_right = flux_physical(u_right)
    
    fluxes = (f_left + f_right) / 2.0 - wave_speed * (u_right - u_left) / 2.0
    
    return fluxes


def upwind_flux(
    u_left: NDArray,
    u_right: NDArray,
    flux_physical: Callable[[NDArray], NDArray],
    wave_speed_fn: Callable[[NDArray], NDArray],
) -> NDArray:
    """
    Upwind numerical flux.
    
    Computes the upwind flux based on the sign of the local wave speed:
    - If a(u) > 0: F = f(u_L)  (flow from left)
    - If a(u) < 0: F = f(u_R)  (flow from right)
    - If a(u) = 0: F = (f(u_L) + f(u_R))/2  (no flow)
    
    Parameters
    ----------
    u_left : NDArray
        Left state(s) at cell interfaces.
    u_right : NDArray
        Right state(s) at cell interfaces.
    flux_physical : Callable[[NDArray], NDArray]
        Physical flux function f(u).
    wave_speed_fn : Callable[[NDArray], NDArray]
        Wave speed function a(u) = df/du.
    
    Returns
    -------
    fluxes : NDArray
        Numerical fluxes at interfaces.
    """
    f_left = flux_physical(u_left)
    f_right = flux_physical(u_right)
    
    # Compute wave speeds
    a_left = wave_speed_fn(u_left)
    a_right = wave_speed_fn(u_right)
    
    # Average wave speed at interface
    a_avg = (a_left + a_right) / 2.0
    
    # Upwind selection
    fluxes = np.where(a_avg > 0, f_left, np.where(a_avg < 0, f_right, (f_left + f_right) / 2.0))
    
    return fluxes
