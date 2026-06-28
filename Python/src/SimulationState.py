"""SimulationState.py"""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class SimulationState:
    # Juveniles distribution (augmented with ghost cells and not augmented)
    u_augmented: NDArray 
    u: NDArray
    # translated views
    u_x1_right: NDArray
    u_x1_left: NDArray
    u_x2_right: NDArray
    u_x2_left: NDArray

    # Adults distribution (augmented with ghost cells and not augmented)
    w_augmented: NDArray 
    w: NDArray
    # translated views
    w_x1_right: NDArray
    w_x1_left: NDArray
    w_x2_right: NDArray
    w_x2_left: NDArray
    
    # Trees distribution (augmented with ghost cells and not augmented)
    z_augmented: NDArray
    z: NDArray
    # translated views
    z_x1_right: NDArray
    z_x1_left: NDArray
    z_x2_right: NDArray
    z_x2_left: NDArray



def initialize_simulation_state(NaJ:int, NaA:int, Nx1:int, Nx2:int) -> SimulationState:
    # augmented with ghost cells
    
    u_augmented = np.zeros((NaJ, Nx1 + 2, Nx2 + 2))
    w_augmented = np.zeros((NaA, Nx1 + 2, Nx2 + 2))
    z_augmented = np.zeros((Nx1 + 2, Nx2 + 2))

    return SimulationState(u_augmented=u_augmented, 
                           u=u_augmented[:, 1:-1, 1:-1],
                           u_x1_right=u_augmented[:, 1:, 1:-1],
                           u_x1_left=u_augmented[:, :-1, 1:-1],
                           u_x2_right=u_augmented[:, 1:-1, 1:],
                           u_x2_left=u_augmented[:, 1:-1, :-1],
                           w_augmented=w_augmented, 
                           w=w_augmented[:, 1:-1, 1:-1],
                           w_x1_right=w_augmented[:, 1:, 1:-1],
                           w_x1_left=w_augmented[:, :-1, 1:-1],
                           w_x2_right=w_augmented[:, 1:-1, 1:],
                           w_x2_left=w_augmented[:, 1:-1, :-1],
                           z_augmented=z_augmented, 
                           z=z_augmented[1:-1, 1:-1],
                           z_x1_right=z_augmented[1:, 1:-1],
                           z_x1_left=z_augmented[:-1, 1:-1],
                           z_x2_right=z_augmented[1:-1, 1:],
                           z_x2_left=z_augmented[1:-1, :-1])