# mortality.py
from dataclasses import dataclass
from typing import Callable
from numpy.typing import NDArray

@dataclass
class MortalityParams:
    # mortality rate of juveniles
    delta_u: NDArray|float

    # mortality rate of adults
    delta_w: NDArray|float

    # mortality rate of trees
    delta_z: Callable[[NDArray], NDArray]

    # susceptibility of trees to the disease
    lmbda: NDArray


def set_mortality_params(delta_u:NDArray|float, delta_w:NDArray|float, delta_z:Callable[[NDArray], NDArray], 
                         lmbda:NDArray) -> MortalityParams:

    return MortalityParams(delta_u=delta_u, delta_w=delta_w, delta_z=delta_z, lmbda=lmbda)