#!/usr/bin/env python3
###
### plot_functions.py
### function for producing plots

import numpy as np
import logging
import subprocess
import matplotlib
import matplotlib.pyplot as plt
from collections.abc import Sequence
from src import save_load as sav
from numpy.typing import NDArray
from pathlib import Path

# Constants for plotting and movie generation
MENCODER_FPS = 5
MENCODER_CODEC = 'wmv2'
DEFAULT_RCOUNT = 50
DEFAULT_CCOUNT = 50
DEFAULT_N_COLOR = 256


def _prepare_matplotlib(saving: bool) -> None:
    """Configure matplotlib for saving mode if requested."""
    if saving:
        matplotlib.use('AGG')
        plt.ioff()

#
# Plot q
#
def plot_density(dir_name:Path, x:NDArray, y:NDArray,
                 pic:int, q_range:Sequence[tuple[float, float]], 
                 da_juveniles:float, da_adults:float, **kwargs) -> None:

    color_bar = kwargs.get('color_bar', True)
    no_pictures = kwargs.get('no_pictures', False)
    rcount = kwargs.get('rcount', DEFAULT_RCOUNT)
    ccount = kwargs.get('ccount', DEFAULT_CCOUNT)

    juveniles_range = q_range[0]
    adult_range = q_range[1]
    tree_range = q_range[2]

    if no_pictures:
        return

    file_name = (dir_name / f'saving_{pic:03d}').with_suffix('.npz')
    u, w, z, time = sav.load_status(file_name)

    densities = [
        ('juveniles', np.sum(u, axis=0) * da_juveniles, juveniles_range),
        ('adults', np.sum(w, axis=0) * da_adults, adult_range),
        ('trees', z, tree_range),
    ]

    for name, data, value_range in densities:
        contour_file = (dir_name / f'plot_contour_{name}_{pic:03d}').with_suffix('.png')
        plot_contour(
            x,
            y,
            data,
            file_name=contour_file,
            title=f'{name.capitalize()} at time = {time:.2f}',
            label=('$x_1$', '$x_2$'),
            lower_value=value_range[0],
            upper_value=value_range[1],
            saving=True,
            color_bar=color_bar,
        )

        surface_file = (dir_name / f'plot_surface_{name}_{pic:03d}').with_suffix('.png')
        plot_surface(
            x,
            y,
            data,
            file_name=surface_file,
            title=f'{name.capitalize()} at time = {time:.2f}',
            label=('$x_1$', '$x_2$'),
            lower_value=value_range[0],
            upper_value=value_range[1],
            saving=True,
            rcount=rcount,
            ccount=ccount,
        )



# Movie
def movie(base_directory_1:Path) -> None:

    base_directory = base_directory_1.as_posix()

    commands = []

    commands.append(f"mencoder 'mf://{base_directory}/plot_surface_tree*.png' -mf type=png:fps={MENCODER_FPS} -ovc lavc -lavcopts vcodec={MENCODER_CODEC} -oac copy -o {base_directory}/movie_surface_trees.avi")
    commands.append(f"mencoder 'mf://{base_directory}/plot_surface_adults*.png' -mf type=png:fps={MENCODER_FPS} -ovc lavc -lavcopts vcodec={MENCODER_CODEC} -oac copy -o {base_directory}/movie_surface_adults.avi")
    commands.append(f"mencoder 'mf://{base_directory}/plot_surface_juven*.png' -mf type=png:fps={MENCODER_FPS} -ovc lavc -lavcopts vcodec={MENCODER_CODEC} -oac copy -o {base_directory}/movie_surface_juveniles.avi")

    commands.append(f"mencoder 'mf://{base_directory}/plot_contour_tree*.png' -mf type=png:fps={MENCODER_FPS} -ovc lavc -lavcopts vcodec={MENCODER_CODEC} -oac copy -o {base_directory}/movie_contour_trees.avi")
    commands.append(f"mencoder 'mf://{base_directory}/plot_contour_adults*.png' -mf type=png:fps={MENCODER_FPS} -ovc lavc -lavcopts vcodec={MENCODER_CODEC} -oac copy -o {base_directory}/movie_contour_adults.avi")
    commands.append(f"mencoder 'mf://{base_directory}/plot_contour_juven*.png' -mf type=png:fps={MENCODER_FPS} -ovc lavc -lavcopts vcodec={MENCODER_CODEC} -oac copy -o {base_directory}/movie_contour_juveniles.avi")

    try:
        for cmd in commands:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
        logging.info('Movie created')
    except subprocess.CalledProcessError as e:
        logging.info('Movie creation failed')
        logging.info("Do you have 'mencoder'?")
        logging.info(f"Error message: {e}")
    except Exception as e:
        logging.info('Movie creation failed')
        logging.info(f"Error message: {e}")

#
# plot a contour
#
def plot_contour(t1:NDArray, t2:NDArray, matrix:NDArray, **kwargs) -> None:

    """ Function drawing the contour of the solution

    :param t1: numpy 1D array describing the horizontal domain
    :param t2: numpy 1D array describing the vertical domain
    :param matrix: numpy 2D array describing the solution values
    :param title: str, plot title
    :param label: tuple of str, axis labels (xlabel, ylabel)
    :param saving: bool, whether to save the plot to file
    :param file_name: Path, output filename
    :param n_color: int, number of color levels in the contour plot
    :param lower_value: float, minimum value for color scale
    :param upper_value: float, maximum value for color scale
    :param color_bar: bool, whether to display a colorbar

    Saves plot as PNG file or displays interactively.

    """

    i = kwargs.get('fig', 1)
    title = kwargs.get('title', '')
    subtitle = kwargs.get('subtitle', False)
    xlabel, ylabel = kwargs.get('label', ('$x$', '$t$'))
    saving = kwargs.get('saving', False)
    file_name = kwargs.get('file_name', 'Contour.png')
    n_color = kwargs.get('n_color', DEFAULT_N_COLOR)
    lower_value = kwargs.get('lower_value', None)
    upper_value = kwargs.get('upper_value', None)
    c_bar = kwargs.get('color_bar', False)

    if lower_value is not None and upper_value is not None:
        v = np.linspace(lower_value, upper_value, n_color)
    else:
        v = None

    _prepare_matplotlib(saving)

    fig = plt.figure(i)
    ax = fig.add_subplot(111)

    if subtitle:
        ax.set_title(title, loc='left').set_fontsize('xx-large')
        ax.set_title(subtitle, loc='right').set_fontsize('small')
    else:
        ax.set_title(title).set_fontsize('xx-large')
    ax.set_xlabel(xlabel, fontsize='xx-large')
    ax.set_ylabel(ylabel, rotation='horizontal', fontsize='xx-large')

    xx, yy = np.meshgrid(t1, t2, indexing='ij')

    c = ax.contourf(xx, yy, matrix, n_color, extend='max', vmin=lower_value,
                    vmax=upper_value, levels=v, cmap='jet')

    if c_bar:
        cbar = plt.colorbar(mappable=c, format='%.2f', ax=ax)
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
def plot_mass(directory:Path, times:NDArray, mass:NDArray, **kwargs) -> None:

    """ Function drawing the mass of the solution versus time

    :param directory: Path, output directory for saving the plot
    :param times: numpy 1D array of time values
    :param mass: numpy 1D array of mass values
    :param title: str, plot title (default: 'Mass versus time')
    :param label: tuple of str, axis labels (xlabel, ylabel)
    :param saving: bool, whether to save the plot to file
    :param file_name: str, output filename (default: 'Mass.png')

    Saves plot as PNG file or displays interactively.

    """

    i = kwargs.get('fig', 1)
    title = kwargs.get('title', 'Mass versus time')
    subtitle = kwargs.get('subtitle', False)
    xlabel, ylabel = kwargs.get('label', ('$t$', ''))
    saving = kwargs.get('saving', False)
    file_name = kwargs.get('file_name', 'Mass.png')

    _prepare_matplotlib(saving)

    fig = plt.figure(i)
    ax = fig.add_subplot(111)

    if subtitle:
        ax.set_title(title, loc='left').set_fontsize('xx-large')
        ax.set_title(subtitle, loc='right').set_fontsize('small')
    else:
        ax.set_title(title).set_fontsize('xx-large')
    ax.set_xlabel(xlabel, fontsize='xx-large')
    ax.set_ylabel(ylabel, rotation = 'horizontal', fontsize='xx-large')

    ax.plot(times, mass)

    if saving:
        fig.savefig((directory / file_name).with_suffix('.png').as_posix(), format='png')
    else:
        plt.show()

    plt.close(fig)


#
# plot a surface
#
def plot_surface(t1:NDArray, t2:NDArray, matrix:NDArray, **kwargs) -> None:

    """ Function drawing the surface of the solution

    :param t1: numpy 1D array describing the horizontal domain
    :param t2: numpy 1D array describing the vertical domain
    :param matrix: numpy 2D array of values to be plotted as surface
    :param title: str, plot title
    :param label: tuple of str, axis labels (xlabel, ylabel)
    :param saving: bool, whether to save the plot to file
    :param file_name: str, output filename
    :param n_color: int, number of color levels
    :param lower_value: float, minimum value for color scale
    :param upper_value: float, maximum value for color scale
    :param rcount: int, number of rows in surface mesh
    :param ccount: int, number of columns in surface mesh

    Saves plot as PNG file or displays interactively.

    """

    i = kwargs.get('fig', 1)
    title = kwargs.get('title', '')
    subtitle = kwargs.get('subtitle', False)
    xlabel, ylabel = kwargs.get('label', ('$x$', '$t$'))
    saving = kwargs.get('saving', False)
    file_name = kwargs.get('file_name', 'Surface.png')
    n_color = kwargs.get('n_color', DEFAULT_N_COLOR)
    lower_value = kwargs.get('lower_value', None)
    upper_value = kwargs.get('upper_value', None)
    rcount = kwargs.get('rcount', DEFAULT_RCOUNT)
    ccount = kwargs.get('ccount', DEFAULT_CCOUNT)

    if lower_value is not None and upper_value is not None:
        v = np.linspace(lower_value, upper_value, n_color)
    else:
        v = None

    _prepare_matplotlib(saving)

    fig = plt.figure(i)
    ax = fig.add_subplot(111, projection='3d')

    if subtitle:
        ax.set_title(title, loc='left').set_fontsize('xx-large')
        ax.set_title(subtitle, loc='right').set_fontsize('small')
    else:
        ax.set_title(title).set_fontsize('xx-large')
    ax.set_xlabel(xlabel).set_fontsize('xx-large')
    ax.set_ylabel(ylabel, rotation = 'horizontal').set_fontsize('xx-large')

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