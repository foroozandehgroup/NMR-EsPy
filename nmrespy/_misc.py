#!/usr/bin/python3
# _misc.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

import copy
import os

import numpy as np
from scipy.integrate import simps

import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama


def aligned_tabular(columns, titles=None):
    """Tabularises a list of lists, with the option of including titles.

    Parameters
    ----------
    columns : list
        A list of lists, representing the columns of the table. Each list
        must be of the same length.

    titles : None or list, default: None
        Titles for the table. If desired, the ``titles`` should be of the same
        length as all of the lists in ``columns``.

    Returns
    -------
    msg : str
        A string with the contents of ``columns`` tabularised.
    """

    if titles:
        sep = ' │'
        for i,(title, column) in enumerate(zip(titles, columns)):
            columns[i] = [title] + column

    else:
        sep = ' '

    pads = []
    for column in columns:
        pads.append(max(len(element) for element in column))

    msg = ''
    for i, row in enumerate(zip(*columns)):
        for j, (pad, e1, e2) in enumerate(zip(pads, row, row[1:])):
            p = pad - len(e1)
            if j == 0:
                msg += f"{e1}{p*' '}{sep}{e2}"
            else:
                msg += f"{p*' '}{sep}{e2}"
        if titles and i == 0:
            for i, pad in enumerate(pads):
                if i == 0:
                    msg += f"\n{(pad+1)*'─'}┼"
                else:
                    msg += f"{(pad+1)*'─'}┼"
            msg = msg[:-1]
        msg += '\n'

    return msg


def make_fid(parameters, n, sw, offset=None):
    """Constructs a discrete time-domain signal (FID), as a summation of
    exponentially damped complex sinusoids, along with the corresponding
    time-points at which the signal was sampled.

    Parameters
    ----------
    parameters : numpy.ndarray
        Parameter array, with ``parameters.shape == (M, 4)`` for a 1D FID,
        or ``parameters.shape == (M, 4)`` for a 2D FID, where `M` is the number
        of oscillators.

    n : [int], [int, int]
        Number of points to construct signal from in each dimension.

    sw : [float], [float, float]
        Sweep width in each dimension, in Hz.

    offset : [float], [float, float], or None, default: None
        Transmitter offset frequency in each dimension, in Hz. If set to
        `None`, the offset frequency will be set to 0Hz in each dimension.


    Returns
    fid : numpy.ndarray
        The synthetic time-domain signal.

    tp : [numpy.ndarray], [numpy.ndarray, numpy.ndarray]
        The time points the FID is sampled at in each dimension.
    """

    if not isinstance(parameters, np.ndarray) or parameters.ndim != 2:
        raise TypeError(
            f'{cols.R}parameters should be a numpy ndarray with 2'
            f' dimesions.{cols.END}'
        )

    if not parameters.shape[1] in [4, 6]:
        raise ValueError(
            f'{cols.R}parameters should statisfy parameters.shape[1] == 4'
            f' (1D FID) or parameters.shape[1] == 6 (2D FID).{cols.END}'
        )

    # FID dimension
    dim = int(parameters.shape[1] / 2) - 1

    if offset is None:
        offset = [0.] * dim

    def _check_valid_arg(value, name, type_, dim):

        errmsg = (
            f'{cols.R}{name} should be a list of length {dim} with values of'
            f' type {type_.__name__}{cols.END}'
        )

        if isinstance(value, type_) and dim == 1:
            return [value]
        elif isinstance(value, list) and len(n) == dim:
            for elem in value:
                if not isinstance(elem, type_):
                    raise TypeError(errmsg)
            return value
        else:
            raise TypeError(errmsg)

    n = _check_valid_arg(n, 'n', int, dim)
    offset = _check_valid_arg(offset, 'offset', float, dim)
    sw = _check_valid_arg(sw, 'sw', float, dim)

    amp = parameters[:, 0]
    phase = parameters[:, 1]
    freq = [parameters[:, 2+i] + offset[i] for i in range(dim)]
    damp = [parameters[:, dim+2+i] for i in range(dim)]
    tp = [np.linspace(0, float(n_) / sw_, n_) for n_, sw_ in zip(n, sw)]

    if dim == 1:
        # Vandermonde matrix of poles
        Z = np.exp(np.outer(tp[0], (1j*2*np.pi*freq[0] - damp[0])))

        # vector of complex ampltiudes
        alpha = amp * np.exp(1j * phase)

        fid = Z @ alpha

    if dim == 2:
        # Vandermonde matrices
        Z1 = np.exp(np.outer(tp[0], (1j*2*np.pi*freq[0] - damp[0])))
        Z2t = np.exp(np.outer((1j*2*np.pi*freq[1] - damp[1]), tp[1]))

        # diagonal matrix of complex amplitudes
        A = np.diag(amp * np.exp(1j * phase))

        fid = Z1 @ A @ Z2t

    return fid
