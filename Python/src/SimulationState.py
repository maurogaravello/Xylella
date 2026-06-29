"""SimulationState.py"""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

# dataclass for u and w
@dataclass
class State3D:
    # augmented distribution
    augmented_density: NDArray
    # not augmented distribution
    density: NDArray

    # partially augmented distributions
    augmented_density_a: NDArray
    augmented_density_x: NDArray
    augmented_density_y: NDArray

    # translated views
    density_a_left: NDArray
    density_x_right: NDArray
    density_x_left: NDArray
    density_y_right: NDArray
    density_y_left: NDArray

    # sum of all values of the density
    def sum(self) -> float:
        return float(np.sum(self.density))
    
    # sum over age
    def sum_over_age(self) -> NDArray:
        return np.sum(self.density, axis=0)
    
    # L^\infty norm
    def inf_norm(self) -> float:
        return float(np.max(np.abs(self.density)))
    
    # puts boundary conditions in the ghost cells
    def boundary(self, boundary_a:NDArray) -> None:
        self.augmented_density[:, 0, :] = 0.
        self.augmented_density[:, -1, :] = 0.
        self.augmented_density[:, :, 0] = 0.
        self.augmented_density[:, :, -1] = 0.
        self.augmented_density_a[0, :, :] = boundary_a


def initialize_state_3D(N_a:int, N_x:int, N_y:int) -> State3D:
    # augmented density with ghost cells
    augmented = np.zeros((N_a+1, N_x+2, N_y+2))

    return State3D(augmented_density=augmented,
                   density=augmented[1:, 1:-1, 1:-1],
                   augmented_density_a=augmented[:, 1:-1, 1:-1],
                   augmented_density_x=augmented[1:, :, 1:-1],
                   augmented_density_y=augmented[1:, 1:-1, :],
                   density_a_left=augmented[:-1, 1:-1, 1:-1],
                   density_x_right=augmented[1:, 2:, 1:-1],
                   density_x_left=augmented[1:, :-2, 1:-1],
                   density_y_right=augmented[1:, 1:-1, 2:],
                   density_y_left=augmented[1:, 1:-1, :-2] )

# dataclass for z
@dataclass
class State2D:
    # augmented distribution
    augmented_density: NDArray
    # not augmented distribution
    density: NDArray

    # partially augmented distributions
    augmented_density_x: NDArray
    augmented_density_y: NDArray

    # translated views
    density_x_right: NDArray
    density_x_left: NDArray
    density_y_right: NDArray
    density_y_left: NDArray

    # sum of all values of the density
    def sum(self) -> float:
        return float(np.sum(self.density))

    # L^\infty norm
    def inf_norm(self) -> float:
        return float(np.max(np.abs(self.density)))
    
    # puts zeros in the ghost cells
    def zero_ghost_cells(self) -> None:
        self.augmented_density[0, :] = 0.
        self.augmented_density[-1, :] = 0.
        self.augmented_density[:, 0] = 0.
        self.augmented_density[:, -1] = 0.


def initialize_state_2D(N_x:int, N_y:int) -> State2D:
    # augmented density with ghost cells
    augmented = np.zeros((N_x+2, N_y+2))

    return State2D(augmented_density=augmented,
                   density=augmented[1:-1, 1:-1],
                   augmented_density_x=augmented[:, 1:-1],
                   augmented_density_y=augmented[1:-1, :],
                   density_x_right=augmented[2:, 1:-1],
                   density_x_left=augmented[:-2, 1:-1],
                   density_y_right=augmented[1:-1, 2:],
                   density_y_left=augmented[1:-1, :-2])

@dataclass
class SimulationState:
    # Juveniles distribution state
    u: State3D

    # Adults distribution state
    w: State3D
    
    # Trees distribution state
    z: State2D

    # Velocity for u
    u_velocity_x: State2D
    u_velocity_y: State2D

    # Velocity for w
    w_velocity_x: State2D
    w_velocity_y: State2D

def initialize_states(N_aJ:int, N_aA:int, N_x:int, N_y:int) -> SimulationState:
    
    u = initialize_state_3D(N_aJ, N_x, N_y)
    w = initialize_state_3D(N_aA, N_x, N_y)
    z = initialize_state_2D(N_x, N_y)
    vel_u_x = initialize_state_2D(N_x, N_y)
    vel_u_y = initialize_state_2D(N_x, N_y)
    vel_w_x = initialize_state_2D(N_x, N_y)
    vel_w_y = initialize_state_2D(N_x, N_y)

    return SimulationState(u=u, w=w, z=z, 
                           u_velocity_x=vel_u_x, u_velocity_y=vel_u_y,
                           w_velocity_x=vel_w_x, w_velocity_y=vel_w_y)
