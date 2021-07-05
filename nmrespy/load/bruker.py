# load.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Provides functionality for importing NMR data."""

import functools
import operator
from pathlib import Path
import re

import numpy as np
from numpy.fft import ifft, ifftshift

import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
    colorama.init()
import nmrespy._errors as errors
from nmrespy._misc import get_yes_no


def get_params_from_jcampdx(names, path):
    """Retrieve parameters from files written in JCAMP-DX format.

    Parameters
    ----------
    names: [str]
        The names of the parameters.

    path: pathlib.Path
        The path to the parameter file.

    Returns
    -------
    params: [str]
        The values of the desired parameters as a string.
    """
    with open(path, 'r') as fh:
        txt = fh.read()

    params = []
    for name in names:
        pattern = r'##\$' + name + r'= (\(\d+\.\.\d+\)\n)?(.+?)#'
        try:
            params.append(
                re.search(pattern, txt, re.DOTALL)
                    .groups()[-1]
                    .replace('\n', ' ')
                    .rstrip()
            )

        except AttributeError:
            # re.search returned None
            raise errors.ParameterNotFoundError(name, path)

    return params

def get_binary_format(info):
    """Determine the formatting of the binary files.

    Parameters
    ----------
    info : dict
        See :py:func:`determine_bruker_data_type` for a description of items.

    Returns
    -------
    binary_format : {'>i4', '<i4', '>f8', '<f8'}
        The binary file format.

    """
    if info['dtype'] == 'pdata':
        names, file = ['DTYPP', 'BYTORDP'], info['param']['procs']
    else:
        names, file = ['DTYPA', 'BYTORDA'], info['param']['acqus']

    encoding, endian = [int(p) for p in get_params_from_jcampdx(names, file)]

    return (('<' if endian == 0 else '>') +
            ('i4' if encoding == 0 else 'f8'))


def determine_bruker_data_type(directory):
    """Given a directory, determines what type of Bruker data is stored.

    This function is used to determine a) whether the specified data is
    time-domain or pdata, and b) the dimension of the data (checks up to
    3D). If it matches the required criteria, a dictionary of information
    will be returned (See Returns). Otherwise, it will return ``None``.

    Parameters
    ----------
    directory : pathlib.Path
        The path to the directory of interest.

    Returns
    -------
    info : dict or None
        If `info` is a dict, it will contain the following items:

        * **'dim'** : *int*, The dimension of the data.
        * **'dtype'** : *{'fid', 'pdata'}*, The type of data (raw time-domain
          or pdata).
        * **'param'** : *dict*, a dictionary of all parameter file paths of
          relevance.
        * **'bin'** : *dict*, a dictionary of all binary file paths of
          relevance.

        If the directory does not satisfy the requirements, `info` will be
        ``None``.
    """

    options = compile_bruker_path_options(directory)
    idx = None
    for i, option in enumerate(options):
        if check_for_bruker_files(option):
            idx = i
    return options[idx] if isinstance(idx, int) else None


def check_for_bruker_files(info):
    """Asserts whether all the paths specified in ``info['bin']`` and
    ``info['param']`` exist.

    Parameters
    ----------
    info : dict
        See :py:func:`determine_bruker_data_type` for a description of items.

    Returns
    -------
    all_exist : bool
    """

    # Check binary file  and parameter file path(s) path exists
    files_to_check = list(info['param'].values()) + list(info['bin'].values())
    if not all([f.is_file() for f in files_to_check]):
        # At least one parameter file is not found.
        # Implies no entries in path_info will be valid.
        return False
    # An entry in path_info has all the requisite files!
    return True


def compile_bruker_path_options(directory):
    """Generates a dictionary of information relating to each experiment type,
    given a certain path.

    Parameters
    ----------
    directory : pathlib.Path
        Path to the directory of interest.

    Returns
    -------
    options : [dict]
        See :py:func:`determine_bruker_data_type` for a description of items
        in each dict.
    """

    oneback = directory.parents[1]
    return [
        # 1D FID
        {
            'bin': {
                'fid' : directory / 'fid',
            },
            'param': {
                'acqus': directory / 'acqus',
            },
            'dim': 1,
            'dtype': 'fid',
        },
        # 2D FID
        {
            'bin': {
                'ser' : directory / 'ser',
            },
            'param': {
                'acqus': directory / 'acqus',
                'acqu2s': directory / 'acqu2s',
            },
            'dim': 2,
            'dtype': 'fid',
        },
        # 3D FID
        {
            'bin': {
                'ser' : directory / 'ser',
            },
            'param': {
                'acqus': directory / 'acqus',
                'acqu2s': directory / 'acqu2s',
                'acqu3s': directory / 'acqu3s',
            },
            'dim': 3,
            'dtype': 'fid',
        },
        # 1D Processed data
        {
            'bin': {
                '1r' : directory / '1r',
            },
            'param': {
                'acqus': oneback / 'acqus',
                'procs': directory / 'procs',
            },
            'dim': 1,
            'dtype': 'pdata',
        },
        # 2D Processed data
        {
            'bin': {
                '2rr' : directory / '2rr',
            },
            'param': {
                'acqus': oneback / 'acqus',
                'acqu2s': oneback / 'acqu2s',
                'procs': directory / 'procs',
                'proc2s': directory / 'proc2s',
            },
            'dim': 2,
            'dtype': 'pdata',
        },
        # 3D Processed data
        {
            'bin': {
                '3rrr' : directory / '3rrr',
            },
            'param': {
                'acqus': oneback / 'acqus',
                'acqu2s': oneback / 'acqu2s',
                'acqu3s': oneback / 'acqu3s',
                'procs': directory / 'procs',
                'proc2s': directory / 'proc2s',
                'proc3s': directory / 'proc3s',
            },
            'dim': 3,
            'dtype': 'pdata',
        },
    ]


def get_parameters(info):
    """Retrieve various experiment parameters required by NMR-EsPy"""

    shape, nuc, sw, off, sfo, fnmode = [[] for _ in range(6)]
    sizeparam = 'TD' if info['dtype'] == 'fid' else 'SI'

    for d in ['', '2', '3'][:info['dim']]:
        acqusfile = info['param'][f'acqu{d}s']
        if info['dtype'] == 'fid':
            sizefile = acqusfile
        else:
            sizefile = info['param'][f'proc{d}s']

        names = ['NUC1', 'SW_h', 'O1', 'SFO1', 'FnMODE']
        p = get_params_from_jcampdx(names, acqusfile)
        p += get_params_from_jcampdx([sizeparam], sizefile)
        p = iter(p)

        for lst, typ in zip((nuc, sw, off, sfo, fnmode, shape),
                            (str, float, float, float, int, int)):
            lst.append(typ(next(p)))

    nuc = [re.search('<(.+?)>', n).group(1) for n in nuc]
    if sizeparam == 'TD':
        shape[0] //= 2
    print(shape)

    return {
        'shape' : shape,
        'nuc' : nuc,
        'sw' : sw,
        'off' : off,
        'sfo' : sfo,
        'fnmode' : fnmode,
    }


def import_data(paths, binfmt):
    return [np.fromfile(file, dtype=binfmt) for file in paths]


def reshape_multidim_data(data, shape):
    """
    Parameters
    ----------
    data : [numpy.ndarray]
        List of data arrays (initially all flat).

    shape : list
        Size of each dimension in the data array.

    Returns
    -------
    shaped_data : [numpy.ndarray]
        List of reshaped datasets.
    """
    return [d.reshape(*reversed(shape)) for d in data]


def complexify(data):
    """Convert flattened array comprising elements of the form
    [Re, Im, Re, Im, ...] to a complex array"""
    return [d[::2] + 1j * d[1::2] for d in data]


def remove_zeros(data):
    """
    Applying convdta on a 2D signal leads to a block of zeros at
    the end of the data, i.e.

                      ************0000
                      ************0000
                      ************0000
                      ************0000

    This needs to be removed!
    """

    sliceshape = data[0].shape
    directsize = sliceshape[-1]
    for i in range(directsize - 1, -1, -1):
        if not np.array_equal(
            data[0][:, i],
            np.zeros(sliceshape, dtype='complex')
        ):
            break

    return [d[..., :i] for d in data]


def load_bruker(directory, ask_convdta=True):
    """Loads data and relevant parameters from Bruker format.

    Parameters
    ----------
    directory: str
        Absolute path to data directory.

    ask_convdta: bool, optional
        If `True` (default), the user will be warned that the data should
        have its digitial filter removed prior to importing if the data to be
        impoprted is from an `fid` or `ser` file. If `False`, the user is
        not warned.

    Returns
    -------
    result: dict
        A dictionary containing items with the following keys:

        * `'source'` (str) - The type of data imported (`'bruker_fid'` or
          `'bruker_pdata'`).
        * `'data'` (numpy.ndarray) - The data.
        * `'directory'` (pathlib.Path) - The path to the data directory.
        * `'sweep_width'` (`[float]` or `[float, float]`) - The experiemnt
          sweep width in each dimension (Hz).
        * `'offset'` (`[float]` or `[float, float]`) - The transmitter
          offset frequency in each dimension (Hz).
        * `'transmitter_frequency'` (`[float]` or `[float, float]`) - The
          transmitter frequency in each dimension (MHz).
        * `'nuclei'` (`[str]` or `[str, str]`) - The nucelus in each
          dimension.
        * `'binary_format'` ('str') - The format of the binary data file.
          Of the form `'<endian><unitsize>'`, where `'<endian>'` is either
          `'<'` (little endian) or `'>'` (big endian), and `'<unitsize>'`
          is either `'i4'` (32-bit integer) or `'f8'` (64-bit float).

    Notes
    -----
    *Directory Requirements*

    The path specified by `directory` should
    contain the following files depending on whether you are importing 1- or
    2-dimesnional data, and whether you are importing raw FID data or
    processed data:

    * 1D data

      - Raw FID

        + `directory/fid`
        + `directory/acqus`

      - Processed data

        + `directory/1r`
        + `directory/acqus`
        + `directory/procs`


    * 2D data

      - Raw FID

        + `directory/ser`
        + `directory/acqus`
        + `directory/acqu2s`

      - Processed data

        + `directory/2rr`
        + `directory/acqus`
        + `directory/acqu2s`
        + `directory/procs`
        + `directory/proc2s`

    * 3D data

      - Raw FID

        + `directory/ser`
        + `directory/acqus`
        + `directory/acqu2s`
        + `directory/acqu3s`

      - Processed data

        + `directory/3rrr`
        + `directory/acqus`
        + `directory/acqu2s`
        + `directory/acqu3s`
        + `directory/procs`
        + `directory/proc2s`
        + `directory/proc3s`

    *Digital Filters*

    If you are importing raw FID data, make sure the path
    specified corresponds to an `fid` or `ser` file which has had its
    digital filter removed. To do this, open the data you wish to analyse in
    TopSpin, and enter `convdta` in the bottom-left command line. You will be
    prompted to enter a value for the new data directory. It is this value you
    should use in `directory`, not the one corresponding to the original
    (uncorrected) signal. There alternative approaches you could take to
    deal with this. For example, the nmrglue provides the function
    `nmrglue.fileio.bruker.rm_dig_filter <https://nmrglue.readthedocs.io/en/\
    latest/reference/generated/nmrglue.fileio.bruker.rm_dig_filter.html#\
    nmrglue.fileio.bruker.rm_dig_filter>`_

    **For Development**

    .. todo::
       Incorporate functionality to phase correct digitally filtered data.
       This would circumvent the need to ask the user to ensure they have
       performed `convdta` prior to importing.
    """

    # Warning text checking that the data being imported has had
    # `convdta` applied to it.
    if ask_convdta:
        convdta_prompt = (
            f'{cols.OR}WARNING: It is necessary that the FID you import has '
            'been digitally filtered, using the <convdta> command in TopSpin. '
            'If you have not done this, please do so before proceeding\nIf '
            'you wish to proceed, enter [y]\nIf you wish to quit, enter [n]: '
            f'{cols.END} '
        )

    d = Path(directory)
    if not d.is_dir():
        raise IOError(f'\n{cols.R}Directory {directory} doesn\'t exist!'
                      f'{cols.END}')

    info = determine_bruker_data_type(d)
    if info is None:
        raise errors.InvalidDirectoryError(directory)

    # If the data to be imported is a raw FID, alert the user about the
    # necessity to correct for the digital filter
    if info['dtype'] == 'fid' and ask_convdta:
        get_yes_no(convdta_prompt)

    # TODO: 3D data not spported currently. Expect to allow Pseudo-2D data
    # in the future
    if info['dim'] > 2:
        raise errors.MoreThanTwoDimError()

    p = get_parameters(info)
    for k, v in zip(p.keys(), p.values()):
        info[k] = v
    info['binfmt'] = get_binary_format(info)

    data = import_data(paths=info['bin'].values(), binfmt=info['binfmt'])

    if info['dtype'] == 'fid' and info['dim'] > 1:
        data = \
            remove_zeros(
                reshape_multidim_data(
                    complexify(data), info['shape']
                )
            )

    elif info['dtype'] == 'fid':
        data = complexify(data)

    elif info['dim'] > 1:
        reshape_multidim_data(data, info['shape'])
    print(data[0].shape)
    info['data'] = data
    info['path'] = d
    del info['bin']
    del info['param']

    return info


# =================================================
# TODO Correctly handle different detection methods
#
# The crucial value in the parameters files is
# FnMODE, which is an integer value
#
# The folowing schemes are possible:
#
# 0 -> undefined
#
# 1 -> QF
# Successive fids are acquired with incrementing
# time interval without changing the receiver phase.
#
# 2 -> QSEQ
# Successive fids will be acquired with incrementing
# time interval and receiver phases 0 and 90 degrees.
#
# 3 -> TPPI
# Successive fids will be acquired with incrementing
# time interval and receiver phases 0, 90, 180 and
# 270 degrees.
#
# 4 -> States
# Successive fids will be acquired incrementing the
# time interval after every second fid and receiver
# phases 0 and 90 degrees.
#
# 5 -> States-TPPI
# Successive fids will be acquired incrementing the
# time interval after every second fid and receiver
# phases 0,90,180 and 270 degrees.cd
#
# 6 -> Echo-Antiecho
# Special phase handling for gradient controlled
# experiments.
#
# See: http://triton.iqfr.csic.es/guide/man/acqref/fnmode.htm
# =================================================


# # Dealing with processed data.
# else:
#     if info['dim'] == 1:
#         # Does the following things:
#         # 1. Flips the spectrum (order frequency bins from low to high
#         # going left to right)
#         # 2. Performs inverse Fourier Transform
#         # 3. Retrieves the first half of the signal, which is a conjugate
#         # symmetric virtual echo
#         # 4. Doubles to reflect loss of imaginary data
#         # 5. Zero fills back to original signal size
#         # 6. Halves the first point to ensure no baseline shift takes
#         # place
#         data = 2 * np.hstack(
#             (
#                 ifft(ifftshift(data[::-1]))[:int(data.size // 2)],
#                 np.zeros(int(data.size // 2), dtype='complex'),
#             )
#         )
#         data[0] /= 2
#
#     else:
#         # Reshape data, and flip in both dimensions
#         data = data.reshape(
#             n_indirect, int(data.size / n_indirect),
#         )[::-1, ::-1]
#         # Inverse Fourier Tranform in both dimensions
#         for axis in range(2):
#             data = ifft(ifftshift(data, axes=axis), axis=axis)
#         # Slice signal in half in both dimensions
#         data = 4 * data[:int(data.shape[0] // 2), :int(data.shape[1] // 2)]
