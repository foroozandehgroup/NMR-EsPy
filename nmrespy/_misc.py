#!/usr/bin/python3
#io.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

# DESCRIPION GOES HERE

from copy import deepcopy
import os

import numpy as np
from scipy.integrate import simps

from ._cols import *
if USE_COLORAMA:
    import colorama


def get_yn(prompt):
    """
    get_yn(prompt)

    ───Description─────────────────────────────
    Gets user to enter either 'y' (yes) or 'n' (no), in response to a
    prompt. If 'n' is enterred, the program quits. If 'y' is enterred, the
    program proceeds. If another input is given, the user is prompted
    again.

    ───Parameters──────────────────────────────
    prompt - str
        A message for the user to respond to
    """
    inp = input(prompt)
    if inp == 'y' or inp == 'Y':
        pass
    elif inp == 'n' or inp == 'N':
        print(f'{R}exiting program...{END}')
        exit()
    else:
        get_yn(f'{R}Invalid input. Please enter [y] or [n]:{END} ')


def check_path(fname, dir, force_overwrite):
    """
    check_path(fname, dir, force_overwrite)

    ───Description─────────────────────────────
    Checks:
    1) whether a given path, specified by a directory and filename,
       exists already. If so, ask the user whether they want to overwrite
       the file. If not:
    2) whether the directory specified exists.

    ───Parameters──────────────────────────────
    fname - str
        Name of the file.
    dir - str
        Path to the directory to store the file in.
    force_overwrite - Bool
        If False, if the file specified already exists, the user is asked
        if they are happy to proceed with overwriting it. If True, no such
        check is made - the file is automatically overwritten.

    ───Returns─────────────────────────────────
    path - str
        If all checks are successful, the path corresponding to dir/fname
        is returned.
    """
    path = os.path.join(dir, fname)
    if os.path.isfile(path):
        if force_overwrite:
            pass
        else:
            prompt = f'{O}The file {path} already exists. Do you' \
                     + f' wish to overwrite it? [y/n]: {END}'
            get_yn(prompt)

    elif os.path.isdir(os.path.split(path)[0]):
        pass

    else:
        raise IOError(f'{R}{dir} is not a directory!{END}')

    return path


def conv_ppm_idx(p, sw_p, offset_p, n, direction='ppm->idx'):
    """
    ppm_to_index(p, sw, offset, N)
    *** ONLY 1D PARAMETERS SUPPORTED AT THE MOMENT ***

    ───Description─────────────────────────────
    Converts a parameter from ppm units to the corresponding array index,
    or vice versa

    ───Parameters──────────────────────────────
    p - float or int
        The parameter of interest, in ppm.
    sw_p - float
        Sweep width, in ppm.
    offset_p - float
        Transmitter offset frequency, in ppm
    n - int
        Tne number of time-points the signal is composed of
    direction - str
        Should be either 'ppm->idx' (default), or 'idx->ppm'. If
        'ppm->idx', p should be a in units of ppm, and it will be
        converted to the corresponding array index. If 'idx->ppm',
        p should be an array index, and it will be converted to
        the corresponding ppm value

    ───Returns─────────────────────────────────
    conv_p - float or int
        The converted parameter
    """
    if direction == 'ppm->idx':
        if isinstance(p, float):
            return int(round((offset_p + (sw_p / 2) - p) * (n / sw_p)))
        else:
            msg = f'\n{R}p should be a float if converting from ppm to' \
                  + f' index. Got {type(p)} instead{END}'
            raise TypeError(msg)

    elif direction == 'idx->ppm':
        if isinstance(p, int):
            return float(offset_p + (sw_p / 2) - ((p * sw_p) / n))
        else:
            msg = f'\n{R}p should be a int if converting from index to' \
                  + f' ppm. Got {type(p)} instead{END}'
            raise TypeError(msg)

    else:
        msg = f'\n{R}direction should be \'ppm->idx\' or \'idx->ppm\'{END}'
        raise ValueError(msg)


def mkfid(para, n, sw, offset, dim):
    """
    mkfid(para, n, sw, offset, dim)

    ───Description─────────────────────────────
    Constructs a discrete time-domain signal (FID), as a summation of
    exponentially damped complex sinusoids, along with the corresponding
    time-points at which the signal was sampled.

    ───Parameters──────────────────────────────
    para - numpy.ndarray
        Parameter array, of shape (M, 4) or (M, 6).
    n - int
        Number of points to construct signals from.
    sw - float
        Sweep width (Hz) in each dimension.
    offset - float
        Offset frequency (Hz) in each dimension.
    dim - int
        Signal dimension. Should be 1 or 2

    ───Returns─────────────────────────────────
    fid - numpy.ndarray
        The synthetic time-domain signal.
    tp - numpy.ndarray or tuple
        The time points the FID is sampled at in each dimension. If
        dim is 1, this is an ndarray. If dim is 2, this is a tuple of
        two ndarrays.
    """

    if dim == 1:
        # time points
        tp = np.linspace(0, float(n[0]-1) / sw[0], int(n[0]))

        para_new = deepcopy(para)
        # shift to have centre frequency at zero
        para_new[..., 2] = -para_new[..., 2] + offset[0]
        Z = np.exp(np.outer(tp, (1j * 2 * np.pi * para_new[..., 2] -
                   para_new[..., 3])))
        alpha = para_new[..., 0] * np.exp(1j * para[..., 1])
        fid = np.matmul(Z, alpha).flatten()

    if dim == 2:
        # time points
        tp1 = np.linspace(0, float(n[0]-1 / sw[0]), n[0])
        tp2 = np.linspace(0, float(n[1]-1 / sw[1]), n[1])

        # adjust based on transmitter offset
        para[..., 2] = para[..., 2] - offset[0]
        para[..., 3] = para[..., 3] - offset[1]
        Z1 = np.exp(np.outer(tp1, (1j * 2 * np.pi * para[..., 2] -
                                      para[..., 4])))
        Z2t = np.exp(np.outer((1j * 2 * np.pi * para[..., 3] -
                               para[..., 5]), tp2))
        A = np.diag(para[..., 0] * np.exp(1j * para[..., 1]))
        fid = np.matmul(Z1, np.matmul(A, Z2t))

    return fid
