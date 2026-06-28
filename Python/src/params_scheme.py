# params_scheme.py
from dataclasses import dataclass
from typing import Callable
from numpy.typing import NDArray

@dataclass
class SimulationParams:
    # Time of simulation
    T: float

    # Callable parameters 
    weeds_convolved: NDArray
    s:       Callable[[float], NDArray|float]
    beta:    Callable[[float], NDArray]
    velocity_juveniles: Callable[[NDArray, NDArray], tuple[NDArray, NDArray]]
    velocity_adults: Callable[[NDArray, NDArray], tuple[NDArray, NDArray]]

    # number of pictures to be taken during the simulation
    N_pictures: int

    # CFL number
    cfl: float
    # Lax-Friedrichs method
    LF: bool
    upwind: bool

def set_parameters(time:float, weeds_conv:NDArray,
                   s:Callable[[float], NDArray], beta:Callable[[float], NDArray],
                   velocity_juveniles:Callable[[NDArray, NDArray], tuple[NDArray, NDArray]],
                   velocity_adults:Callable[[NDArray, NDArray], tuple[NDArray, NDArray]],
                   number_pictures:int = 121, cfl:float = 0.95, LF:bool=True) -> SimulationParams:
    
    return SimulationParams(T=time, weeds_convolved=weeds_conv, s=s, beta=beta, 
                            velocity_juveniles=velocity_juveniles, velocity_adults=velocity_adults,
                            N_pictures=number_pictures, cfl=cfl, LF=LF, upwind=not(LF))