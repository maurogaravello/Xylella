"""
Fast repeated convolution: x1 is fixed, x2 changes every call.

Replicates scipy.signal.oaconvolve(x1, x2, mode='same') but caches the FFT
of x1 so it's computed only once (and recomputed automatically only if the
required FFT size changes, e.g. because x2's shape changed).
"""

import numpy as np
from scipy.fft import rfftn, irfftn, next_fast_len
from numpy.typing import ArrayLike, NDArray


class FastConvolveSame:
    def __init__(self, x1:ArrayLike):
        self.x1 = np.asarray(x1)
        self._fft_shape:list[int] = [1, 2]
        self._full_shape:list[int] = [1, 4]
        self._X1:NDArray = np.copy(self.x1)

    def _prepare(self, x2_shape:tuple[int, ...]):
        full_shape:list[int] = [n1 + n2 - 1 for n1, n2 in zip(self.x1.shape, x2_shape)]
        fft_shape:list[int] = [next_fast_len(n) for n in full_shape]

        # Only recompute FFT(x1) if the padded size actually changed
        if fft_shape != self._fft_shape:
            self._fft_shape = fft_shape
            self._full_shape = full_shape
            self._X1 = rfftn(self.x1, fft_shape)

        return self._fft_shape, self._full_shape

    def __call__(self, x2_input:ArrayLike):
        x2 = np.asarray(x2_input)
        fft_shape, full_shape = self._prepare(x2.shape)

        X2 = rfftn(x2, fft_shape)
        full = irfftn(self._X1 * X2, s=fft_shape)
        full = full[tuple(slice(0, s) for s in full_shape)]

        return _centered(full, self.x1.shape)


def _centered(arr, newshape):
    """Crop the centered part of arr to newshape (same convention scipy uses)."""
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    sl = tuple(slice(startind[k], endind[k]) for k in range(len(endind)))
    return arr[sl]


