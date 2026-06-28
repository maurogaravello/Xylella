#!/usr/bin/env python3
###
### plot_functions.py
### function for producing plots

import numpy as np
import logging
import os
from lib import save_and_load as sav

#
# Plot q
#
def plot_density(dir_name, x:np.array, y:np.array,
                 pic:int, q_range:list, 
                 da_juveniles:float, da_adults:float, **kwargs):

    debug = kwargs.get('debug', False)
    color_bar = kwargs.get('color_bar', True)
    no_pictures = kwargs.get('no_pictures', False)
    rcount = kwargs.get('rcount', 50)
    ccount = kwargs.get('ccount', 50)

    juveniles_range = q_range[0]
    adult_range = q_range[1]
    tree_range = q_range[2]

    if no_pictures:
        pass
    else:
        file_name = (dir_name / 'saving_{:03d}'.format(pic)).with_suffix('.npz')
        u, w, z, time = sav.load_status(file_name)

        # Plot juveniles contour
        u = np.sum(u, axis=0) * da_juveniles
        file_name = (dir_name / 'plot_contour_juveniles_{:03d}'.format(pic)).with_suffix('.png')
        plot_contour(x, y, u, file_name=file_name,
                    title='Juveniles at time = {0:.2f}'.format(time),
                    label=('$x_1$', '$x_2$'),
                    lower_value=juveniles_range[0],
                    upper_value=juveniles_range[1],
                    saving=True, color_bar=color_bar)
        
        # Plot adults contour
        w = np.sum(w, axis=0) * da_adults
        file_name = (dir_name / 'plot_contour_adults_{:03d}'.format(pic)).with_suffix('.png')
        plot_contour(x, y, w, file_name=file_name,
                    title='Adults at time = {0:.2f}'.format(time),
                    label=('$x_1$', '$x_2$'),
                    lower_value=adult_range[0],
                    upper_value=adult_range[1],
                    saving=True, color_bar=color_bar)

        # Plot trees contour
        file_name = (dir_name / 'plot_contour_trees_{:03d}'.format(pic)).with_suffix('.png')
        plot_contour(x, y, z, file_name=file_name,
                    title='Trees at time = {0:.2f}'.format(time),
                    label=('$x_1$', '$x_2$'),
                    lower_value=tree_range[0],
                    upper_value=tree_range[1],
                    saving=True, color_bar=color_bar)

        # Plot juveniles surface  
        file_name = (dir_name / 'plot_surface_juveniles_{:03d}'.format(pic)).with_suffix('.png')
        plot_surface(x, y, u, file_name=file_name,
                    title='Juveniles at time = {0:.2f}'.format(time),
                    label=('$x_1$', '$x_2$'),
                    lower_value=juveniles_range[0],
                    upper_value=juveniles_range[1],
                    saving=True, rcount=rcount, ccount=ccount)

        # Plot adults surface  
        file_name = (dir_name / 'plot_surface_adults_{:03d}'.format(pic)).with_suffix('.png')
        plot_surface(x, y, w, file_name=file_name,
                    title='Adults at time = {0:.2f}'.format(time),
                    label=('$x_1$', '$x_2$'),
                    lower_value=adult_range[0],
                    upper_value=adult_range[1],
                    saving=True, rcount=rcount, ccount=ccount)
        
        # Plot trees surface  
        file_name = (dir_name / 'plot_surface_trees_{:03d}'.format(pic)).with_suffix('.png')
        plot_surface(x, y, z, file_name=file_name,
                    title='Trees at time = {0:.2f}'.format(time),
                    label=('$x_1$', '$x_2$'),
                    lower_value=tree_range[0],
                    upper_value=tree_range[1],
                    saving=True, rcount=rcount, ccount=ccount)



# Movie
def movie(base_directory):

    base_directory = base_directory.as_posix()

    commands = []

    commands.append("mencoder 'mf://{}/plot_surface_tree*.png' -mf type=png:fps=5 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o {}/movie_surface_trees.avi".format(base_directory, base_directory))
    commands.append("mencoder 'mf://{}/plot_surface_adults*.png' -mf type=png:fps=5 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o {}/movie_surface_adults.avi".format(base_directory, base_directory))
    commands.append("mencoder 'mf://{}/plot_surface_juven*.png' -mf type=png:fps=5 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o {}/movie_surface_juveniles.avi".format(base_directory, base_directory))

    commands.append("mencoder 'mf://{}/plot_contour_tree*.png' -mf type=png:fps=5 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o {}/movie_contour_trees.avi".format(base_directory, base_directory))
    commands.append("mencoder 'mf://{}/plot_contour_adults*.png' -mf type=png:fps=5 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o {}/movie_contour_adults.avi".format(base_directory, base_directory))
    commands.append("mencoder 'mf://{}/plot_contour_juven*.png' -mf type=png:fps=5 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o {}/movie_contour_juveniles.avi".format(base_directory, base_directory))

    try:
        for cmd in commands:
            os.system(cmd)
        logging.info('Movie created')
    except:
        logging.info('Movie creation failed')
        logging.info("Do you have 'mencoder'?")

#
# plot a contour
#
def plot_contour(t1, t2, matrix, **kwargs):

    """ Function drawing the contour of the solution

    :param t1: numpy array describing the horizontal domain
    :param t2: numpy array describing the vertical domain
    :param matrix: tuple containing 4 numpy arrays describing the solution

    :param n_color: int. Number of colors of the plot
    :param DirName: str. Name of the directory

    This function write the file Contour.png

    """

    i = kwargs.get('fig', 1)
    title = kwargs.get('title', '')
    subtitle = kwargs.get('subtitle', False)
    xlabel, ylabel = kwargs.get('label', ('$x$', '$t$'))
    saving = kwargs.get('saving', False)
    file_name = kwargs.get('file_name', 'Contour.png')
    n_color = kwargs.get('n_color', 256)
    lower_value = kwargs.get('lower_value', None)
    upper_value = kwargs.get('upper_value', None)
    c_bar = kwargs.get('color_bar', False)

    if lower_value != None and upper_value != None:
        v = np.linspace(lower_value, upper_value, n_color)
    else:
        v = None

    if saving:
        import matplotlib
        matplotlib.use('AGG') # NOT GOOD! Only for plotting remotely

    import matplotlib.pyplot as plt
    if saving:
        plt.ioff()

    fig = plt.figure(i)
    ax = fig.add_subplot(111)

    if subtitle:
        ax.set_title(title, loc='left').set_size('xx-large')
        ax.set_title(subtitle, loc='right').set_size('small')
    else:
        ax.set_title(title).set_size('xx-large')
    ax.set_xlabel(xlabel).set_size('xx-large')
    ax.set_ylabel(ylabel, rotation='horizontal').set_size('xx-large')

    xx, yy = np.meshgrid(t1, t2, indexing='ij')

    c = ax.contourf(xx, yy, matrix, n_color, extend='max', vmin=lower_value,
                    vmax=upper_value, levels=v, cmap='jet')

    if c_bar:
        cbar = plt.colorbar(mappable=c, format='%.2f', ax=ax)
        # cbar.set_ticks([])
    else:
        plt.gca().set_aspect('equal')


    if saving:
        fig.savefig(file_name.with_suffix('.png').as_posix(), format='png')
    else:
        plt.show()

    plt.close(fig)


#
# plot the mass
#
def plot_mass(directory, times:np.array, mass:np.array, **kwargs):

    """ Function drawing the mass of the solution versus the time

    :param times: numpy array describing the times
    :param mass: numpy array describing the mass

    This function write the file Mass.png

    """

    i = kwargs.get('fig', 1)
    title = kwargs.get('title', 'Mass versus time')
    subtitle = kwargs.get('subtitle', False)
    xlabel, ylabel = kwargs.get('label', ('$t$', ''))
    saving = kwargs.get('saving', False)
    file_name = kwargs.get('file_name', 'Mass.png')

    if saving:
        import matplotlib
        matplotlib.use('AGG') # NOT GOOD! Only for plotting remotely

    import matplotlib.pyplot as plt
    if saving:
        plt.ioff()

    fig = plt.figure(i)
    ax = fig.add_subplot(111)

    if subtitle:
        ax.set_title(title, loc='left').set_size('xx-large')
        ax.set_title(subtitle, loc='right').set_size('small')
    else:
        ax.set_title(title).set_size('xx-large')
    ax.set_xlabel(xlabel).set_size('xx-large')
    ax.set_ylabel(ylabel, rotation = 'horizontal').set_size('xx-large')

    c = ax.plot(times, mass)

    if saving:
        fig.savefig((directory / file_name).with_suffix('.png').as_posix(), format='png')
    else:
        plt.show()

    plt.close(fig)


#
# plot a surface
#
def plot_surface(t1, t2, matrix, **kwargs):

    """ Function drawing the surface of the solution

    :param t1: numpy array describing the horizontal domain
    :param t2: numpy array describing the vertical domain
    :param matrix: tuple. The value of the function to be plotted
    :param DirName: str. Name of the directory

    This function write the file Surface.png

    """

    i = kwargs.get('fig', 1)
    title = kwargs.get('title', '')
    subtitle = kwargs.get('subtitle', False)
    xlabel, ylabel = kwargs.get('label', ('$x$', '$t$'))
    saving = kwargs.get('saving', False)
    file_name = kwargs.get('file_name', 'Surface.png')
    n_color = kwargs.get('n_color', 256)
    lower_value = kwargs.get('lower_value', None)
    upper_value = kwargs.get('upper_value', None)
    rcount = kwargs.get('rcount', 50)
    ccount = kwargs.get('ccount', 50)

    if lower_value != None and upper_value != None:
        v = np.linspace(lower_value, upper_value, n_color)
    else:
        v = None

    if saving:
        import matplotlib
        matplotlib.use('AGG') # NOT GOOD! Only for plotting remotely

    import matplotlib.pyplot as plt
    if saving:
        plt.ioff()

    fig = plt.figure(i)
    ax = fig.add_subplot(111, projection='3d')

    if subtitle:
        ax.set_title(title, loc='left').set_size('xx-large')
        ax.set_title(subtitle, loc='right').set_size('small')
    else:
        ax.set_title(title).set_size('xx-large')
    ax.set_xlabel(xlabel).set_size('xx-large')
    ax.set_ylabel(ylabel, rotation = 'horizontal').set_size('xx-large')

    xx, yy = np.meshgrid(t1, t2, indexing='ij')

    # fix the scale
    ax.set_zlim3d(lower_value, upper_value)

    c = ax.plot_surface(xx, yy, matrix, vmin=lower_value,
                        vmax=upper_value, cmap='jet',
                        rcount=rcount, ccount=ccount)

    if saving:
        fig.savefig(file_name.with_suffix('.png').as_posix(), format='png')
    else:
        plt.show()

    plt.close(fig)