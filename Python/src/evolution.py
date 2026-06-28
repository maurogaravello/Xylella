#!/usr/bin/env python3


import numpy as np
from scipy import signal
from numpy.typing import NDArray

# import logging

# from datetime import datetime
# from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolClass
from typing import Tuple
from src.SimulationState import SimulationState
from src.kernels import Kernels
from src.meshes import SpaceMeshes, AgeMeshes
from src.mortality import MortalityParams
from src.params_scheme import SimulationParams


# importing functions for Lax-Friedrichs
from src import finite_volume_methods




## one-step evolution
## This is the numerical scheme for the convective part
## U_n^{i,j,k} := \frac{1}{\Delta a \Delta x_1 \Delta x_2} \int_{C^{i,j,k}} u(t_n, a, x_1 x_2) \dd a \dd x_1 \dd x_2
##
## U_{n+1}^{i,j,k} = U_{n}^{i,j,k} 
##                   + \frac{\Delta t}{\Delta a} (U_n^{i-1,j,k} - U_n^{i,j,k})
##                   + \frac{\Delta t}{\Delta x_1} (- \abs{V_{1,n}^{j,k}} U_n^{i,j,k} + \max(V_{1,n}^{j-1,k}, 0) U_n^{i,j-1,k} - \min(V_{1,n}^{j+1,k}, 0) U_n^{i,j+1,k})
##                   + \frac{\Delta t}{\Delta x_2} (- \abs{V_{2,n}^{i,k}} U_n^{i,j,k} + \max(V_{2,n}^{i,k-1}, 0) U_n^{i,j,k-1} - \min(V_{2,n}^{j,k+1}, 0) U_n^{i,j,k+1})
## 
def one_step_evolution(time:float, last_time_a:float,
                       spatial_meshes:SpaceMeshes,
                       age_meshes:AgeMeshes,
                       states:SimulationState, 
                       mortalities:MortalityParams,
                       parameters:SimulationParams,
                       kernels:Kernels,
                       pools:PoolClass, convolution=signal.oaconvolve) -> Tuple[float, float]:
    
    finalTime = parameters.T
    dx1, dx2, da_J, da_A = spatial_meshes.dx1, spatial_meshes.dx2, age_meshes.da_J, age_meshes.da_A
    cfl = parameters.cfl
    total_adults = np.sum(states.w, axis=0) * da_A

    # logging.info('Convolutions started. Time elapsed = {}'.format(datetime.now() - start))

    # convolutions (done in parallel)
    # convolutions: eta_2 -> eta_z, eta_1 -> eta
    #               line 1: z * eta_2                 Indices: 0
    #               line 2: u * grad_eta_1              Indices: 1, 2
    #               line 3: w * grad_eta_1            Indices: 3, 4
    conv = [pools.apply_async(convolution, (states.z, b), {'mode': 'same'}) for b in (kernels.eta_z, )] \
            + [pools.apply_async(convolution, (np.sum(states.u, axis=0) * da_J, b), {'mode': 'same'}) for b in kernels.grad_eta] \
            + [pools.apply_async(convolution, (total_adults * da_A, b), {'mode': 'same'}) for b in kernels.grad_eta]
    conv = [p.get()*(dx1*dx2) for p in conv]

    # logging.info('Convolutions completed. Total time elapsed = {}'.format(datetime.now() - start))
    
    # velocity vector fields (not augmented) ARRIVATO QUI
    velocity_u = parameters.velocity_juveniles(-parameters.weeds_convolved * conv[1], 
                                               -parameters.weeds_convolved * conv[2])
    sum_convolutions = -(parameters.weeds_convolved + conv[0])
    velocity_w = (sum_convolutions * conv[3], sum_convolutions *conv[4])
    del sum_convolutions, conv

    # maximum velocities
    max_velocity_u = np.amax(velocity_u[0]**2 + velocity_u[1]**2)**0.5
    max_velocity_w = np.amax(velocity_w[0]**2 + velocity_w[1]**2)**0.5

    # CFL and dt (time step)
    dt = cfl * min(dx1, dx2, da_J, da_A) / max(max_velocity_u, max_velocity_w, 1)
    dt = float(dt)
    if time + dt > finalTime and time <= finalTime:
        dt = finalTime - time

    # flag for convection age part
    if time + dt >= last_time_a + da_A:
        flag_a = True
        last_time_a += da_A
    else:
        flag_a = False

    # update u
    N_u = states.u_augmented.shape[0]
    np.copyto(states.density_trash_2[:N_u], states.u_augmented) # make a copy of u (necessary for the source term)
    convective_scheme(states.u_augmented, velocity_u, 
                      dt, dx1, dx2, 
                      parameters.beta(time), 
                      states.density_trash_1[:N_u], 
                      states.velocity_trash, 
                      states.flux_trash[:N_u], 
                      flag_a)
    src.source(states.u, states.density_trash_2[:N_u, 1:-1, 1:-1], mortalities.delta_u, dt)

    # update w
    N_w = states.w_augmented.shape[0]
    np.copyto(states.density_trash_2[:N_w], states.w_augmented) # make a copy of w (necessary for the source term)
    convective_scheme(states.w_augmented, velocity_w, 
                      dt, dx1, dx2, 
                      states.u_augmented[-1, 1:-1, 1:-1], 
                      states.density_trash_1[:N_w],
                      states.velocity_trash, 
                      states.flux_trash[:N_w], 
                      flag_a)
    src.source(states.w, states.density_trash_2[:N_w, 1:-1, 1:-1], mortalities.delta_w, dt)

    # update z
    src.source_trees(states.z, mortalities.delta_z(total_adults), 
                     mortalities.lmbda, parameters.s(time), states.velocity_trash[1:-1, 1:-1], dt)

    return time + dt, last_time_a


## Convective scheme (here velocity is NOT augmented)
def convective_scheme(density:NDArray, velocity:Tuple[NDArray, NDArray], 
                      dt:float, dx1:float, dx2:float, 
                      boundary:NDArray,
                      density_trash:NDArray,
                      velocity_trash:NDArray,
                      flux_trash:NDArray,
                      flag_a:bool=False) -> None:

    # density is augmented
    # velocity is not augmented
    # trashes are augmented

    # setting boundary conditions for density (all zeros)
    density[:, 0, :] = 0.
    density[:, :, 0] = 0.
    density[:, :, -1] = 0.    
    density[:, -1, :] = 0.

    # setting boundary conditions for velocity trash (all zeros)
    velocity_trash[0, :] = 0.
    velocity_trash[-1, :] = 0.
    velocity_trash[:, 0] = 0.
    velocity_trash[:, -1] = 0.

    # setting boundary conditions for flux trash (all zeros)
    flux_trash[0, :, :] = 0.
    flux_trash[-1, :, :] = 0.
    flux_trash[:, 0, :] = 0.
    flux_trash[:, -1, :] = 0.    
    flux_trash[:, :, 0] = 0.    
    flux_trash[:, :, -1] = 0.

    V_1, V_2 = velocity

    #
    # Dimensional splitting (we are in 2D)
    #

    # In the first part it is important the extension in x direction
    # Select the correct array first with slicing (augmented only in x directions)
    density_aug_x = density[:, :, 1:-1]
    out_v = velocity_trash[:, 1:-1] # augmentation of the velocity V_1
    out_flux = flux_trash[:, :, 1:-1] # augmentation of the flux
    # Fill the center (density is ok)
    out_v[1:-1] = V_1
 
    # Calculate the flux (put in out_flux)
    np.multiply(density_aug_x, out_v, out=out_flux)

    # out_d as a view of density_trash (to put the result of the Lax_Friedrichs_scheme)
    out_d = density_trash[:, 1:-1, :] # augmented in y direction
    LF.Lax_Friedrichs_scheme_preallocated(density_aug_x[:, 2:, :], density_aug_x[:, :-2, :], 
                          out_flux[:, 2:, :], out_flux[:, :-2, :], dt/dx1, out_d[:, :, 1:-1])


    # In the second part of the dimensional splitting we revert the roles of the density...
    # the input is out_d the output density (to avoid unnecessary copies)
    # extension along y
    out_v = velocity_trash[1:-1, :]
    out_flux = flux_trash[:, 1:-1, :]
    # Fill the center (density is ok)
    out_v[:, 1:-1] = V_2

    # The flux
    np.multiply(out_d, out_v, out=out_flux)

    LF.Lax_Friedrichs_scheme_preallocated(out_d[:, :, 2:], out_d[:, :, :-2], out_flux[:, :, 2:], out_flux[:, :, :-2], dt/dx2, 
                          out=density[:, 1:-1, 1:-1])


    #
    # Finally the age component...
    #
    if flag_a:
        # convection in a
        density[1:] = density[:-1]
        density[0, 1:-1, 1:-1] = boundary