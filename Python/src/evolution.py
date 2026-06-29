#!/usr/bin/env python3


import numpy as np
from scipy import signal
from multiprocessing.pool import Pool as PoolClass

from src.SimulationState import SimulationState
from src.kernels import Kernels
from src.meshes import SpaceMeshes, AgeMeshes
from src.mortality import MortalityParams
from src.params_scheme import SimulationParams
from src import finite_volume_methods as fvm




## one-step evolution
def one_step_evolution(time:float,
                       spatial_meshes:SpaceMeshes,
                       age_meshes:AgeMeshes,
                       states:SimulationState, 
                       mortalities:MortalityParams,
                       parameters:SimulationParams,
                       kernels:Kernels,
                       pools:PoolClass, convolution=signal.oaconvolve) -> float:
    
    finalTime = parameters.T
    dx1, dx2, da_J, da_A = spatial_meshes.dx1, spatial_meshes.dx2, age_meshes.da_J, age_meshes.da_A
    cfl = parameters.cfl
    total_juveniles = states.u.sum_over_age() * da_J
    total_adults = states.w.sum_over_age() * da_A

    # convolutions (done in parallel)
    # convolutions: eta_2 -> eta_z, eta_1 -> eta
    #               line 1: z * eta_2                 Indices: 0
    #               line 2: u * grad_eta_1              Indices: 1, 2
    #               line 3: w * grad_eta_1            Indices: 3, 4
    conv = [pools.apply_async(convolution, (states.z.density, b), {'mode': 'same'}) for b in (kernels.eta_z, )] \
            + [pools.apply_async(convolution, (total_juveniles, b), {'mode': 'same'}) for b in kernels.grad_eta] \
            + [pools.apply_async(convolution, (total_adults , b), {'mode': 'same'}) for b in kernels.grad_eta]
    conv = [p.get()*(dx1*dx2) for p in conv]
   
    # velocity vector fields (not augmented)
    velocity_u = parameters.velocity_juveniles(-parameters.weeds_convolved * conv[1], 
                                               -parameters.weeds_convolved * conv[2])
    max_velocity_u = np.amax(velocity_u[0]**2 + velocity_u[1]**2)**0.5
    np.copyto(states.u_velocity_x.density, velocity_u[0])
    np.copyto(states.u_velocity_y.density, velocity_u[1])
    states.u_velocity_x.zero_ghost_cells()
    states.u_velocity_y.zero_ghost_cells()
    
    sum_convolutions = -(parameters.weeds_convolved + conv[0])
    velocity_w = parameters.velocity_adults(sum_convolutions * conv[3], 
                                            sum_convolutions *conv[4])
    del sum_convolutions, conv
    max_velocity_w = np.amax(velocity_w[0]**2 + velocity_w[1]**2)**0.5
    np.copyto(states.w_velocity_x.density, velocity_w[0])
    np.copyto(states.w_velocity_y.density, velocity_w[1])
    states.w_velocity_x.zero_ghost_cells()
    states.w_velocity_y.zero_ghost_cells()

    # CFL and dt (time step)
    alpha = float(max(max_velocity_u, max_velocity_w, 1))
    dt = cfl * min(dx1, dx2, da_J, da_A) / alpha
    dt = float(dt)
    # dt if possible has to coincide with the age grid
    nn = int((time + dt) / da_J)
    if nn * da_J > time:
        dt = nn * da_J - time
    # dt if possible has to coincide with the final time
    if time + dt > finalTime and time <= finalTime:
        dt = finalTime - time

    # update u
    states.u.boundary(parameters.beta(time))
    rhs = -mortalities.delta_u * states.u.density \
        - fvm.upwind_for_age(states.u.augmented_density_a, da_J)
    if parameters.LF:
        # Lax-Friedrichs
        rhs += - fvm.lax_friedrichs_for_x(states.u.augmented_density_x, 
                                          states.u_velocity_x.augmented_density_x, 
                                          dx1, alpha) \
            - fvm.lax_friedrichs_for_y(states.u.augmented_density_y, 
                                       states.u_velocity_y.augmented_density_y, 
                                       dx2, alpha)
    else:
        # upwind
        rhs += - fvm.upwind_for_x(states.u.augmented_density_x, 
                                  states.u_velocity_x.augmented_density_x, dx1) \
            - fvm.upwind_for_y(states.u.augmented_density_y, 
                               states.u_velocity_y.augmented_density_y, dx2)
    # time derivative...        
    new_u = states.u.density + dt * rhs
    np.copyto(states.u.density, new_u)

    # update w
    states.w.boundary(states.u.density[-1, :, :])
    rhs = -mortalities.delta_w * states.w.density \
        - fvm.upwind_for_age(states.w.augmented_density_a, da_A)
    if parameters.LF:
        # Lax-Friedrichs
        rhs += - fvm.lax_friedrichs_for_x(states.w.augmented_density_x, 
                                          states.w_velocity_x.augmented_density_x, 
                                          dx1, alpha) \
            - fvm.lax_friedrichs_for_y(states.w.augmented_density_y, 
                                       states.w_velocity_y.augmented_density_y, 
                                       dx2, alpha)
    else:
        # upwind
        rhs += - fvm.upwind_for_x(states.w.augmented_density_x, 
                                  states.w_velocity_x.augmented_density_x, dx1) \
            - fvm.upwind_for_y(states.w.augmented_density_y, 
                               states.w_velocity_y.augmented_density_y, dx2)
    # time derivative...        
    new_w = states.w.density + dt * rhs
    np.copyto(states.w.density, new_w)

    # update z
    new_z = (1 - dt * mortalities.lmbda * mortalities.delta_z(total_adults)) \
        * states.z.density
    np.copyto(states.z.density, new_z)
    
    return time + dt


