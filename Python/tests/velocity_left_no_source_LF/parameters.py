import numpy as np
from typing import Tuple
from numpy.typing import NDArray

######################################################
## Numerical domain
##
## a_limits = (a_min, a_T, a_max)
## x1_limits = (x1_min, x1_max)
## x2_limits = (x2_min, x2_max)
##
## 2*N_a: approximative number of mesh points for age
## N_x1: approximative number of mesh points for x1
## N_x2: approximative number of mesh points for x2
##
######################################################

a_limits = (0, 2, 12) # age limits
x1_limits = (-50., 50.) # horizontal limits
x2_limits = (-50., 50.) # vertical limits

N_a = 41 # 250
N_x1 = 61 # 320
N_x2 = 74 # 320

######################################################
## Weeds
######################################################

def weeds(xx1:NDArray, xx2:NDArray) -> NDArray:
    """
    Weeds distribution

    Parameters
    ----------
    xx1 : meshgrid
        2D array of x1 coordinates
    xx2 : meshgrid
        2D array of x2 coordinates

    Returns
    -------
    array_like
        2D array of weeds distribution
    """
    sn = ((xx1 <= -19) & (xx1 >= -40) & (np.abs(xx2) <= 10)).astype(float)
    dx = ((xx1 <= 10) & (xx1 >= -16) & (np.abs(xx2) <= 10)).astype(float)

    return 5.*(sn+dx)

######################################################
## Mollifiers
# eta_mu,
# eta (serve solo il gradiente)
# eta_z,
######################################################

# visual horizon of u and w w.r.t. weed
vhweed = 5.0
# visual horizon of u and w w.r.t. u and w
vhuw = 2.0
# visual horizon of w w.r.t. trees
vhwz = 5.0

# It represents \eta_\mu. the gradient is not important!!!
def eta_mu(xx1:NDArray, xx2:NDArray) -> NDArray:
    """
    Kernel $eta_mu$

    Parameters
    ----------
    xx1 : meshgrid
        2D array of x1 coordinates
    xx2 : meshgrid
        2D array of x2 coordinates

    Returns
    -------
    array_like
        2D array of kernel values
    """

    
    tmp = 1 - (xx1**2 + xx2**2)/(vhweed**2)

    res = np.maximum(0, tmp)
    return res

x1_eta_mu_limits = (-vhweed*1.2, vhweed*1.2)
x2_eta_mu_limits = (-vhweed*1.2, vhweed*1.2)

# It represents \eta. only the gradient is important!!!
def grad_eta(xx1:NDArray, xx2:NDArray) -> Tuple[NDArray, NDArray]:
    """
    Gradient of the kernel $eta$

    Parameters
    ----------
    xx1 : meshgrid
        2D array of x1 coordinates
    xx2 : meshgrid
        2D array of x2 coordinates

    Returns
    -------
    a tuple of two 2D arrays
    """

    res_x = -2 * xx1 / (vhuw**2)
    res_y = -2 * xx2 / (vhuw**2)

    return  (res_x, res_y)


x1_grad_eta_limits = (-vhuw*1.2, vhuw*1.2)
x2_grad_eta_limits = (-vhuw*1.2, vhuw*1.2)

# It represents \eta_z. the gradient is NOT important!!!
def eta_z(xx1:NDArray, xx2:NDArray) -> NDArray:
    """
    Kernel $eta_z$

    Parameters
    ----------
    xx1 : meshgrid
        2D array of x1 coordinates
    xx2 : meshgrid
        2D array of x2 coordinates

    Returns
    -------
    array_like
        2D array of kernel values
    """

    tmp = 1 - (xx1**2 + xx2**2)/(vhwz**2)

    res = np.maximum(0, tmp)
    return res

x1_eta_z_limits = (-vhwz*1.2, vhwz*1.2)
x2_eta_z_limits = (-vhwz*1.2, vhwz*1.2)

######################################################
## Mortality functions
######################################################

def delta_u(aa:NDArray, xx1:NDArray, xx2:NDArray) -> NDArray|float:
    """
    Mortality function $delta_u$

    Parameters
    ----------
    aa : meshgrid
        3D array of age coordinates
    xx1 : meshgrid
        3D array of x1 coordinates
    xx2 : meshgrid
        3D array of x2 coordinates

    Returns
    -------
    array_like
        3D array of kernel values
    """
    
    return np.zeros_like(xx1)

def delta_w(aa:NDArray, xx1:NDArray, xx2:NDArray) -> NDArray|float:
    """
    Mortality function $delta_w$

    Parameters
    ----------
    aa : meshgrid
        3D array of age coordinates
    xx1 : meshgrid
        3D array of x1 coordinates
    xx2 : meshgrid
        3D array of x2 coordinates

    Returns
    -------
    array_like
        3D array of kernel values
    """
    return np.zeros_like(xx1)

def delta_z(xx1:NDArray, xx2:NDArray, int_w:NDArray) -> NDArray:
    """
    Mortality function $delta_z$

    Parameters
    ----------
    xx1 : meshgrid
        2D array of x1 coordinates
    xx2 : meshgrid
        2D array of x2 coordinates

    Returns
    -------
    array_like
        2D array of kernel values
    """

    # same output dimension of xx1
    chi = 0.01
    return (int_w / (1 + chi*int_w)) * np.zeros_like(xx1)

######################################################
## Parameter lambda: susceptibility
######################################################

def lmbda(xx1:NDArray, xx2:NDArray) -> NDArray:
    """
    Function $lambda$

    Parameters
    ----------
    xx1 : meshgrid
        2D array of x1 coordinates
    xx2 : meshgrid
        2D array of x2 coordinates

    Returns
    -------
    array_like
        2D array of kernel values
    """
# same output dimension of xx1
    return 0.4 * np.ones_like(xx1)

######################################################
## New olive trees: s
######################################################

def s(t:float, xx1:NDArray, xx2:NDArray) -> NDArray|float:
    """
    Function $s$ (source term for new olive trees)

    Parameters
    ----------
    t : float
        time
    xx1 : meshgrid
        2D array of x1 coordinates
    xx2 : meshgrid
        2D array of x2 coordinates

    Returns
    -------
    array_like
        2D array of kernel values
    """
    # same output dimension of xx1
    return np.zeros_like(xx1)

######################################################
## fertility (boundary conditions for juveniles)
######################################################

def beta(t:float, xx1:NDArray, xx2:NDArray) -> NDArray:
    """
    Function $\beta$

    Parameters
    ----------
    t : float
        time
    xx1 : meshgrid
        2D array of x1 coordinates
    xx2 : meshgrid
        2D array of x2 coordinates

    Returns
    -------
    array_like
        2D array of kernel values
    """
    int_time = 20. * max( (t-2) * (3-t), 0)
    ret_space = (weeds(xx1,xx2) > 0)
    # return int_time * ret_space * (xx1>-16)
    return np.zeros_like(xx1)

######################################################
## Initial conditions
######################################################

def u0(aa:NDArray, xx1:NDArray, xx2:NDArray) -> NDArray:
    """
    Initial condition for $u$

    Parameters
    ----------
    aa : meshgrid
        3D array of age coordinates
    xx1 : meshgrid
        3D array of x1 coordinates
    xx2 : meshgrid
        3D array of x2 coordinates

    Returns
    -------
    array_like
        3D array of kernel values
    """
    return 10. * (xx1**2 + xx2**2 <= 20**2) * (aa <= 0.2)

def w0(aa:NDArray, xx1:NDArray, xx2:NDArray) -> NDArray:
    """
    Initial condition for $w$

    Parameters
    ----------
    aa : meshgrid
        3D array of age coordinates
    xx1 : meshgrid
        3D array of x1 coordinates
    xx2 : meshgrid
        3D array of x2 coordinates

    Returns
    -------
    array_like
        3D array of kernel values
    """
    return np.zeros_like(aa)

def z0(xx1:NDArray, xx2:NDArray) -> NDArray:
    """
    Initial condition for $z$

    Parameters
    ----------
    xx1 : meshgrid
        2D array of x1 coordinates
    xx2 : meshgrid
        2D array of x2 coordinates

    Returns
    -------
    array_like
        2D array of kernel values
    """

    middle = 2. * (np.abs(xx1) <= 10) * (np.abs(xx2) <= 10)
    top = 2. * (np.abs(xx1) <= 10) * (xx2 <= 42) * (xx2 >= 13)
    bottom = 2. * (np.abs(xx1) <= 10) * (xx2 <= -20) * (xx2 >= -42)
    left = 2. * (xx1 <= -25) * (xx1 >= -42) * (np.abs(xx2) <= 10)
    right = 2. * (xx1 <= 42) * (xx1 >= 25) * (np.abs(xx2) <= 10)

    return middle + top + bottom + left + right


######################################################
## Time of the simulation
######################################################

T = 12.

######################################################
## Plotting ranges
######################################################

density_u_limits = (0.1, 4.) # (0.0001, 4.)
density_w_limits = (0.1, 4.) # (0.0001, 4.)
density_z_limits = (1., 2.) # (0.0001, 2.)
density_limits = (density_u_limits, density_w_limits, density_z_limits)

######################################################
## Number of pictures (approx.)
######################################################
N_pictures = 121

######################################################
## CFL
######################################################
cfl = 0.95

######################################################
## Method
######################################################
LF = True
upwind = not LF

######################################################
## Velocity function
######################################################

def velocity_juveniles(xx1:NDArray, xx2:NDArray) -> Tuple[NDArray, NDArray]:
    """
    Velocity function for juveniles

    Parameters
    ----------
    xx1 : meshgrid
        2D array of x1 coordinates
    xx2 : meshgrid
        2D array of x2 coordinates

    Returns
    -------
    array_like
        Tuple of 2D arrays of kernel values
    """
    return -3.*np.ones_like(xx1), np.zeros_like(xx2)

def velocity_adults(xx1:NDArray, xx2:NDArray) -> Tuple[NDArray, NDArray]:
    """
    Velocity function for adults

    Parameters
    ----------
    xx1 : meshgrid
        2D array of x1 coordinates
    xx2 : meshgrid
        2D array of x2 coordinates

    Returns
    -------
    array_like
        Tuple of 2D arrays of kernel values
    """
    return -3.*np.ones_like(xx1), np.zeros_like(xx2)
