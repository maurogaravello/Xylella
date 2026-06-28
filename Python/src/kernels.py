#!/usr/bin/env python3

### kernels.py

from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Tuple, Callable


@dataclass
class Kernels:
    # gradient of $\eta$
    grad_eta: Tuple[NDArray, NDArray]

    # kernel $\eta_z$
    eta_z: NDArray


## kernels creation
def kernels(g_eta: Callable[[NDArray, NDArray], Tuple[NDArray, NDArray]], 
            eta_zeta: Callable[[NDArray, NDArray], NDArray],
            xx1_grad_eta: NDArray, xx2_grad_eta: NDArray, 
            xx1_eta_z: NDArray, xx2_eta_z: NDArray) -> Kernels: 
    
    """
    Create kernels for the given input arrays.

    Parameters:
        g_eta (Callable): A function that takes two input arrays 
                          and returns a tuple of two output arrays representing 
                          the gradient of $\eta$.
        eta_zeta (Callable): A function that takes two input arrays and returns 
                             an output array representing the kernel $\eta_z$.
        xx1_grad_eta (NDArray): Meshgrid for the gradient of $\eta$.
        xx2_grad_eta (NDArray): Meshgrid for the gradient of $\eta$.
        xx1_eta_z (NDArray): Meshgrid for the gradient of $\eta_z$.
        xx2_eta_z (NDArray): Meshgrid for the gradient of $\eta_z$.

    Returns:
        Kernels: An instance of the Kernels dataclass containing 
                 the gradient of $\eta$ and the kernel $\eta_z$.

    """
    
    
    grad_eta = g_eta(xx1_grad_eta, xx2_grad_eta)
    eta_z = eta_zeta(xx1_eta_z, xx2_eta_z)
 
    return Kernels(grad_eta=grad_eta, eta_z=eta_z)