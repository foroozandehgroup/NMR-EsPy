#!/usr/bin/python3
# _misc.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

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

def isiterable(obj):
    try:
        iterable = iter(obj)
        return True
    except TypeError:
        return False



def convert(value, sw, off, n, sfo, conversion):

    if conversion == 'idx->hz':
        return float(off + (sw / 2) - ((value * sw) / n))

    elif conversion == 'idx->ppm':
        return float((off + sw / 2 - value * sw / n) / sfo)

    elif conversion == 'ppm->idx':
        return int(round((off + (sw / 2) - sfo * value) * (n / sw)))

    elif conversion == 'ppm->hz':
        return value * sfo

    elif conversion == 'hz->idx':
        return int((n / sw) * (off + (sw / 2) - value))

    elif conversion == 'hz->ppm':
        return value / sfo


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
