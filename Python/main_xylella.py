#!/usr/bin/env python3
###
### main_xylella.py
###
### Simulation performed for the xylella nonlocal system

import argparse
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import signal
import importlib.util
from multiprocessing import Pool
from multiprocessing.pool import Pool as PoolClass # For type hinting

from src import evolution
from src import kernel_meshesgrids
from src import kernels
from src import meshes
from src import mortality
from src import params_scheme
from src import plot_functions as plot
from src import save_load as sal
from src import SimulationState

# logging...
def start_log(file_name):
    try:
        file_name.unlink()
    except FileNotFoundError:
        pass

    logging.basicConfig(filename=file_name, filemode='a', level='INFO')
    start_time = datetime.now()
    logging.info('Started  at {}'.format(start_time))
    return start_time

def create_directory(directory):
    try:
        directory.mkdir()
    except FileExistsError:
        [file.unlink() for file in directory.glob('**/*') if file.is_file()]


if __name__ == '__main__':

    desc = """LF_xylella_main.py: it performs the simulation for the model of xylella using 
    the Lax-Friedrichs scheme and the dimensional splitting method.
    """

    #
    # parser
    #
    parser = argparse.ArgumentParser(description = desc, prog = "main_xylella.py")
    parser.add_argument('dir_name', type=str, help="Enter the name of the directory")

    # simulation
    parser.add_argument('-s', '--simulation', dest='only_simulation',
                        action='store_true',
                        help='Do the simulation.')

    parser.add_argument('-p', '--plot', dest='only_plot',
                        action='store_true',
                        help='Plot after simulation.')

    parser.add_argument('-pn', '--processors_number', type=int,
                        help='Number of processors to use for plotting and/or simulation', default=1)

    parser.add_argument('-m', '--movie', dest='movie',
                        action='store_true',
                        help='Create the movie')
        
    parser.add_argument('-cb', '--color_bar', dest='color_bar',
                        action='store_true', 
                        help='Add a color bar to contour plot')
        
    args = parser.parse_args()
    procs = args.processors_number
    #
    # end of parser
    #

    # Reads all parameters, Initial Datum, Flow and MaxCharSpeed
    dir_name = Path(args.dir_name)
    file_parameters = dir_name / 'parameters.py'

    # module import
    spec = importlib.util.spec_from_file_location('parameters', file_parameters)
    if spec is not None and spec.loader is not None:
        param = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(param)
    else:
        raise ImportError("Failed to load the parameters module")

    # directory of the simulation
    dir_simulation = dir_name / 'Simulation'
    
    #
    # Spatial mesh creation: x_1 and x_2
    #
    spatial_meshes = meshes.spatial_mesh(param.x1_limits, param.x2_limits, param.N_x1, param.N_x2)

    #
    # Age meshes
    #
    age_meshes = meshes.age_mesh(param.a_limits, param.N_a)
    
   
    #
    # Simulation
    #
    if args.only_simulation:

        # directory of the simulation
        create_directory(dir_simulation)

        # starting logging
        filelog = dir_simulation / 'log_Simulation.txt'
        start_time = start_log(filelog)

        #
        # Creation of simulation variables
        #
        states = SimulationState.initialize_states(age_meshes.N_a_J, 
                                                   age_meshes.N_a_A,
                                                   spatial_meshes.N_x1, 
                                                   spatial_meshes.N_x2)
        logging.info('Simulation variables created')
    
        #
        # meshgrids
        #
        xx1_2D, xx2_2D = meshes.meshgrid_2D(spatial_meshes.x1, spatial_meshes.x2)
        aa_J, xx1_J, xx2_J = meshes.meshgrids_3D(age_meshes.a_J, spatial_meshes.x1, spatial_meshes.x2)
        aa_A, xx1_A, xx2_A = meshes.meshgrids_3D(age_meshes.a_A, spatial_meshes.x1, spatial_meshes.x2)
        logging.info('2D and 3D Meshgrids created')

        #
        # Setting u, w, and z with their initial conditions
        #
        np.copyto(states.u.density, param.u0(aa_J,  xx1_J, xx2_J))
        np.copyto(states.w.density, param.w0(aa_A, xx1_A, xx2_A))
        np.copyto(states.z.density, param.z0(xx1_2D, xx2_2D))
        logging.info('Initial conditions set for u, w, and z')
    
        # mortalities (not dependent on time) and lambda
        delta_u = param.delta_u(aa_J, xx1_J, xx2_J)
        delta_w = param.delta_w(aa_A, xx1_A, xx2_A)
        delta_z = lambda w_1: param.delta_z(xx1_2D, xx2_2D, w_1)
        lmbda = param.lmbda(xx1_2D, xx2_2D)
        mortalities = mortality.set_mortality_params(delta_u, delta_w, delta_z, lmbda)
        logging.info('Mortality parameters set')

        #
        # weeds
        # 
        weeds = param.weeds(xx1_2D, xx2_2D)
        xx1_weed, xx2_weed = kernel_meshesgrids.create_2Dmesh_kernels(param.x1_eta_mu_limits, 
                                                                      param.x2_eta_mu_limits,
                                                                      spatial_meshes.dx1, 
                                                                      spatial_meshes.dx2)
        weed_convolution = signal.oaconvolve(weeds, param.eta_mu(xx1_weed, xx2_weed), mode='same')
        del xx1_weed, xx2_weed, weeds

        # fertility of adults (boundary data for juveniles)
        beta = lambda t: param.beta(t, xx1_2D, xx2_2D)

        # source term for new olive trees
        s = lambda t: param.s(t, xx1_2D, xx2_2D)

        # set parameters in a dataclass
        param_simulation = params_scheme.set_parameters(time=param.T, weeds_conv=weed_convolution,
                                                        s=s, beta=beta, 
                                                        velocity_juveniles=param.velocity_juveniles,
                                                        velocity_adults=param.velocity_adults, 
                                                        number_pictures=param.N_pictures, cfl=param.cfl,
                                                        LF=param.LF)
        logging.info('Simulation parameters set')

        #
        # removing the meshgrids
        #
        del xx1_J, xx2_J, xx1_A, xx2_A, aa_J, aa_A
        logging.info('Meshgrids removed from memory')

        #
        # Kernels meshgrids and definitions
        #
        xx1_grad_eta, xx2_grad_eta = kernel_meshesgrids.create_2Dmesh_kernels(param.x1_grad_eta_limits, 
                                                                              param.x2_grad_eta_limits,
                                                                              spatial_meshes.dx1, 
                                                                              spatial_meshes.dx2)   
        xx1_eta_z, xx2_eta_z = kernel_meshesgrids.create_2Dmesh_kernels(param.x1_eta_z_limits, 
                                                                        param.x2_eta_z_limits,
                                                                        spatial_meshes.dx1, 
                                                                        spatial_meshes.dx2)   
        kernels_arrays = kernels.kernels(param.grad_eta, param.eta_z, xx1_grad_eta, xx2_grad_eta, xx1_eta_z, xx2_eta_z)
        logging.info('Kernels arrays created')
        del xx1_grad_eta, xx2_grad_eta, xx1_eta_z, xx2_eta_z
        logging.info('Kernels meshgrids removed from memory')

        # initial time
        time = 0.
        times = [time] # list of times
        last_time_a = time
            
        # times for pictures
        pic = 0
        step_pic = param_simulation.T /(param_simulation.N_pictures-1)

        # Method: Lax-Friedrichs or upwind
        if param_simulation.LF:
            logging.info('It is used the Lax-Friedrichs scheme')
        else:
            logging.info('It is used the upwind scheme')

        # Choice of the convolution method
        # default colvolution method is `signal.oaconvolve`
        logging.info('Choice of the convolution method: possible alternatives are signal.convolve2d or signal.oaconvolve')
        convolution = param.convolution if hasattr(param, 'convolution') else signal.oaconvolve
        logging.info('Here we use {}'.format(convolution.__name__))

        # masses: L^1 norms
        mass_u = [states.u.sum() * spatial_meshes.dx1 
                  * spatial_meshes.dx2 * age_meshes.da_J]
        mass_w = [states.w.sum() * spatial_meshes.dx1 
                  * spatial_meshes.dx2 * age_meshes.da_A]
        mass_z = [states.z.sum() * spatial_meshes.dx1 * spatial_meshes.dx2]
        logging.info('Initial mass for u = {}'.format(mass_u[0]))
        logging.info('Initial mass for w = {}'.format(mass_w[0]))
        logging.info('Initial mass for z = {}'.format(mass_z[0]))

        # L^\infty norms
        l_infty_u = [states.u.inf_norm()]
        l_infty_w = [states.w.inf_norm()]
        l_infty_z = [states.z.inf_norm()]
        logging.info(r'Initial $L^\infty$ norm for u = {}'.format(l_infty_u[0]))
        logging.info(r'Initial $L^\infty$ norm for w = {}'.format(l_infty_w[0]))
        logging.info(r'Initial $L^\infty$ norm for z = {}'.format(l_infty_z[0]))


        # Construction of the pool (just one-time for speed up)
        pools:PoolClass = Pool(processes=procs)
        logging.info('Pool created with {} processors'.format(procs))

        while (time < param.T):

            # saving the status (for plotting)
            if time >= step_pic*pic:
                logging.info('Saving file_{:03d} at time {}'.format(pic, datetime.now()))
                sal.save_status(dir_simulation, 'saving_{:03d}'.format(pic), 
                                states.u.density, 
                                states.w.density, 
                                states.z.density, 
                                time)
                pic += 1        

            # one step evolution
            time = evolution.one_step_evolution(time, 
                                                spatial_meshes, age_meshes,
                                                states,
                                                mortalities,
                                                param_simulation,
                                                kernels_arrays,
                                                pools,
                                                convolution=convolution)
            
            # masses updates
            mass_u.append(states.u.sum() * spatial_meshes.dx1 
                          *spatial_meshes.dx2 * age_meshes.da_J)
            mass_w.append(states.w.sum() * spatial_meshes.dx1 
                          *spatial_meshes.dx2 * age_meshes.da_A)
            mass_z.append(states.z.sum() * spatial_meshes.dx1 *spatial_meshes.dx2)

            # l_infty norms updates
            l_infty_u.append(states.u.inf_norm())
            l_infty_w.append(states.w.inf_norm())
            l_infty_z.append(states.z.inf_norm())

            # times updates
            times.append(time)

        # last saving
        logging.info('Saving file_{:03d} at time {}'.format(pic, datetime.now()))
        sal.save_status(dir_simulation, 'saving_{:03d}'.format(pic), 
                        states.u.density, 
                        states.w.density, 
                        states.z.density, 
                        time)

        sal.save_mass(dir_simulation, np.array(times), 
                      np.array(mass_u), np.array(mass_w), np.array(mass_z), 
                      np.array(l_infty_u), np.array(l_infty_w), np.array(l_infty_z))

        # closing the log
        logging.info('')
        logging.info('Initial mass for u = {}'.format(mass_u[0]))
        logging.info('Final mass for u = {}'.format(mass_u[-1]))
        logging.info('')
        logging.info('Initial mass for w = {}'.format(mass_w[0]))
        logging.info('Final mass for w = {}'.format(mass_w[-1]))
        logging.info('')
        logging.info('Initial mass for z = {}'.format(mass_z[0]))
        logging.info('Final mass for z = {}'.format(mass_z[-1]))
        logging.info('')
        logging.info(r'Initial $L^\infty$ norm for u = {}'.format(l_infty_u[0]))
        logging.info(r'Final $L^\infty$ norm for u = {}'.format(l_infty_u[-1]))
        logging.info('')
        logging.info(r'Initial $L^\infty$ norm for w = {}'.format(l_infty_w[0]))
        logging.info(r'Final $L^\infty$ norm for w = {}'.format(l_infty_w[-1]))
        logging.info('')
        logging.info(r'Initial $L^\infty$ norm for z = {}'.format(l_infty_z[0]))
        logging.info(r'Final $L^\infty$ norm for z = {}'.format(l_infty_z[-1]))
        logging.info('')
        finish_time = datetime.now()
        logging.info('Finished  at {}'.format(finish_time))
        logging.info('Total time = {}'.format(finish_time - start_time))
        logging.info('-------------')


    #
    # Plots
    #
    elif args.only_plot:

        # import multiprocessing   
        from multiprocessing import Process

        # rcount and ccount (for the surface plot)
        rcount = param.rcount if hasattr(param, 'rcount') else 50
        ccount = param.ccount if hasattr(param, 'ccount') else 50        

        # starting logging
        filelog = dir_simulation / 'Plots_log.txt'
        start_time = start_log(filelog)

        try:
            times, mass_u, mass_w, mass_z, l_infty_u, l_infty_w, l_infty_z = sal.load_mass(dir_simulation / 'Mass.npz')
            # plot the mass for u, w, z
            plot.plot_mass(dir_simulation, times, mass_u, saving=True,
                           title="Juveniles' mass", file_name='Mass_u.png')
            plot.plot_mass(dir_simulation, times, mass_w, saving=True,
                           title="Adults' mass", file_name='Mass_w.png')
            plot.plot_mass(dir_simulation, times, mass_z, saving=True,
                           title="Trees' biomass", file_name='Mass_z.png')
            
            # plot the L^\infty norms for u, w, z
            plot.plot_mass(dir_simulation, times, l_infty_u, saving=True, 
                           title=r'$L^\infty$ norm of juveniles', 
                           file_name='L_infty_norm_u.png')
            plot.plot_mass(dir_simulation, times, l_infty_w, saving=True, 
                           title=r'$L^\infty$ norm of adults', 
                           file_name='L_infty_norm_w.png')
            plot.plot_mass(dir_simulation, times, l_infty_z, saving=True, 
                           title=r'$L^\infty$ norm of trees density', 
                           file_name='L_infty_norm_z.png')
            
        except Exception as e:
            logging.error('Error while loading the mass: {}'.format(e))

        # create list of savings        
        files_list = sorted(dir_simulation.glob('saving_*.npz'))

        pp = list(range(procs))
        start = datetime.now()

        logging.info('Plotting with {} processors started at {}'.format(procs, start))
        
        for j in range(len(files_list)):
            elem = files_list[j]
            if (j >= procs):
                pp[j % procs].join()
            pp[j % procs] = Process(target=plot.plot_density, 
                                    args=(dir_simulation, spatial_meshes.x1, 
                                          spatial_meshes.x2, j,
                                           param.density_limits,
                                           age_meshes.a_J, age_meshes.a_A), 
                                           kwargs={'debug':args.debug, 'color_bar':args.color_bar,
                                                   'no_pictures':args.no_pictures,
                                                   'rcount':rcount, 'ccount':ccount})
            pp[j % procs].start()

        end = datetime.now()

        logging.info('')
        logging.info('Time elapsed for plotting, using {} processor(s): {}'.format(procs,
                                                                                   end-start))
        logging.info('-------------')
        logging.info('')
        finish_time = datetime.now()
        logging.info('Finished  at {}'.format(finish_time))
        logging.info('Total time = {}'.format(finish_time - start_time))
    
    #
    # Movies
    #    
    elif args.movie:
        # starting logging
        filelog = dir_simulation / 'Movie_log.txt'
        start_log(filelog)

        logging.info('Movie started at {}'.format(datetime.now()))
        plot.movie(dir_simulation)
        logging.info('Movie finished at {}'.format(datetime.now()))