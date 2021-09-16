# load.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Provides functionality for importing NMR data."""

from pathlib import Path
import re
from typing import FrozenSet, Iterable, List, NewType, Set, Tuple, Union

import numpy as np

from nmrespy import RED, ORA, END, USE_COLORAMA, ExpInfo
if USE_COLORAMA:
    import colorama
    colorama.init()
import nmrespy._errors as errors
from nmrespy._misc import get_yes_no


# Dimension-specifying tags in Bruker parameter files
# i.e. for a 3D experiment, acqus, acqu2s, and acqu3s files will exist.
TAGS = ['', '2', '3']
ListLike = NewType('ListLike', Union[FrozenSet, List, Tuple, Set])


def parse_jcampdx(path: Union[Path, str]) -> dict:
    """Retrieve parameters from files written in JCAMP-DX format.

    Parameters
    ----------
    path
        The path to the parameter file.

    Returns
    -------
    A dictionary of parameters. N.B. All returned values are either strings or
    lists of strings.
    """
    with open(path, 'r') as fh:
        txt = fh.read()

    params = {}
    array_pattern = r'(?=##\$(.+?)= \(\d+\.\.\d+\)\n([\s\S]+?)##)'
    array_matches = re.finditer(array_pattern, txt)

    for match in array_matches:
        key, value = match.groups()
        params[key] = value.rstrip('\n').replace('\n', ' ').split(' ')

    oneline_pattern = r'(?=##\$(.+?)= (.+?)\n##)'
    oneline_matches = re.finditer(oneline_pattern, txt)

    for match in oneline_matches:
        key, value = match.groups()
        params[key] = value

    return params


class BrukerDataset:
    def __init__(self, dtype: str, dim: int, files: dict) -> None:
        # Warning text checking that the data being imported has had
        # `convdta` applied to it.
        if ask_convdta:
            convdta_prompt = (
                f'{ORA}WARNING: It is necessary that the FID you import has '
                'been digitally filtered, using the <convdta> command in '
                'TopSpin. If you have not done this, please do so before '
                'proceeding\nIf you wish to proceed, enter [y]\nIf you wish '
                f'to quit, enter [n]: {END}'
            )

        d = Path(directory).resolve()
        if not d.is_dir():
            raise IOError(f'\n{RED}Directory {d} doesn\'t exist!{END}')

        try:
            info = determine_bruker_data_type(d)
            dim, dtype, files = [info[k] for k in ['data', 'dtype', 'files']]
            del info
        # If the directory is invalid, None is returned, which does not have 
        # a __getitem__ attribute!
        except TypeError:
            raise errors.InvalidDirectoryError(d)

        # TODO: 3D data not spported currently. Expect to allow Pseudo-2D data
        # in the future
        if info['dim'] > 2:
            raise errors.MoreThanTwoDimError()

        # If the data to be imported is a raw FID, alert the user about the
        # necessity to correct for the digital filter
        if dtype == 'fid' and ask_convdta:
            get_yes_no(convdta_prompt)

        


        self.dim = dim
        self.dtype = dtype
        self.datafile = files.pop('data')
        self.paramfiles = files

    def 

    def get_parameters(
        self, filenames: Union[ListLike[str], str, None] = None
    ) -> dict:
        if isinstance(filenames, str):
            filenames = [filenames]
        elif isinstance(filenames, (list, tuple, set, frozenset)):
            pass
        elif filenames is None:
            filenames = [k for k in self.paramfiles.keys()]
        else:
            raise TypeError(f'{RED}Invalid type for `filenames`{END}.')

        params = {}
        for name in filenames:
            try:
                params[name] = parse_jcampdx(self.paramfiles[name])
            except KeyError:
                raise ValueError(
                    f'{RED}`{name}` is an invalid filename. Valid options '
                    f"are:\n{', '.join([k for k in self.paramfiles.keys()])}."
                    f'{END}'
                )

        return next(iter(params.values())) if len(params) == 1 else params

    def get_expinfo(self) -> ExpInfo:
        revtags = reversed(TAGS[:dim])
        akeys = [f'acqu{x}s' for x in revtags]
        pts, sw, offset, sfo, nuclei = [[] for  _ in range(5)]
        for akey in akeys:
            aparams = self.get_parameters(filenames=akey)
            sw.append(float(aparams['SW_h']))
            offset.append(float(aparams['O1']))
            sfo.append(float(aparams['SFO1']))
            nuclei.append(aparams['NUC1'])

            if self.dtype == 'fid':
                td = int(aparams['TD'])
                if akey == '':
                    td //= 2
                pts.append(td)
            elif self.dtype == 'pdata':
                pkey = akey.replace('acqu', 'proc')
                pparams = self.get_parameters(filenames=pkey)
                pts.append(int(pparams['SI']))

        return ExpInfo(pts, sw, offset, sfo, nuclei)


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


def _determine_data_type(directory: Path) -> Union[BrukerDataset, None]:
    """Determine the type of Bruker data stored in ``directory``.

    This function is used to determine

    a) whether the specified data is time-domain or pdata
    b) the dimension of the data (checks up to 3D).

    If the data satisfies the required criteria for a particular dataset type,
    a dictionary of information will be returned. Otherwise, ``None`` will be
    returned.

    Parameters
    ----------
    directory
        The path to the directory of interest.

    Returns
    -------
    Dictionary with the entries:

    * ``'dim'`` (``int``) The dimension of the data.
    * ``'dtype'`` (``'fid'`` or ``'pdata'``) The type of data (raw
      time-domain or pdata).
    * ``'files'`` (``List[pathlib.Path]``) Paths to data and parameter files.
    """
    for option in _compile_experiment_options(directory):
        files = option['files'].values()
        if all_paths_exist(files):
            return BrukerDataset(
                option['dtype'], option['dim'], option['files']
            )
    return None


def all_paths_exist(files: Iterable) -> bool:
    """Determine if all the paths in ``files`` exist.

    Parameters
    ----------
    files
        File paths to check.
    """
    return all([f.is_file() for f in files])


def _compile_experiment_options(directory: Path) -> List[dict]:
    """Generate information dictionaries for different experiment types.

    Compiles dictionaries of information relavent to each experiment type:

    * ``'files'`` - The expected paths to data and parameter files.
    * ``'dim'`` - The data dimension.
    * ``'dtype'`` - The type of data (time-domain or pdata).

    Parameters
    ----------
    directory
        Path to the directory of interest.
    """
    twoback = directory.parents[1]
    options = []
    for i in range(1, 4):
        acqusnames = [f'acqu{x}s' for x in TAGS[:i]]
        acqusfiles = {
            name: path for (name, path) in
            zip(
                acqusnames,
                (directory / name for name in acqusnames)
            )
        }
        # for n, p in zip(acqusnames, (directory / nm for nm in acqusnames)):
        fidfiles = {
            **{'data': directory / ('fid' if i == 1 else 'ser')},
            **acqusfiles
        }
        options.append(
            {
                'files': fidfiles,
                'dtype': 'fid',
                'dim': i
            }
        )

        acqusfiles = {
            name: path for (name, path) in
            zip(
                acqusnames,
                (twoback / path.name for path in acqusfiles.values())
            )
        }
        procsnames = [f'proc{x}s' for x in TAGS[:i]]
        procsfiles = {
            name: path for (name, path) in
            zip(
                procsnames,
                (directory / name for name in procsnames)
            )
        }
        pdatafiles = {
            **{'data': directory / f"{i}{i * 'r'}"},
            **acqusfiles,
            **procsfiles,
        }

        options.append(
            {
                'files': pdatafiles,
                'dtype': 'pdata',
                'dim': i
            }
        )
    print(options)
    return options


def get_parameters(info):
    """Retrieve various experiment parameters required by NMR-EsPy"""

    shape, nuc, sw, off, sfo, fnmode = [[] for _ in range(6)]
    sizeparam = 'TD' if info['dtype'] == 'fid' else 'SI'

    for d in reversed(['', '2', '3'][:info['dim']]):
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
        shape[-1] //= 2

    return {
        'shape': shape,
        'nuc': nuc,
        'sw': sw,
        'off': off,
        'sfo': sfo,
        'fnmode': fnmode,
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
    return [d.reshape(shape) for d in data]


def complexify(data):
    """Convert flattened array comprising elements of the form
    [Re, Im, Re, Im, ...] to a complex array"""
    return [d[::2] + 1j * d[1::2] for d in data]


def remove_zeros(data, shape):
    """
    Signals with a TD1 that is not a multiple of 256 are padded with zeros
    to ensure each FID take up a multiple of 1024 bytes space.

    .. code::

            *******00 *******00 *******00 *******00 *******00
              FID 1  |  FID 2  |  FID 3  |  FID 4  |  FID 5

    These zeros need to be removed!
    """
    # Toal number of datapoints
    size = data[0].size
    # Number of FIDs
    fid_no = np.prod(np.array(shape[:-1]))
    # Number of points assinged to each FID (inc. zero padding)
    blocksize = size // fid_no
    # Number of points in each FID to slice (i.e. the number of zeros)
    slicesize = blocksize - shape[-1]
    # Create mask to remove zeros
    mask = np.ones(size).astype(bool)
    for n in range(1, fid_no + 1):
        mask[n * blocksize - slicesize: n * blocksize] = False
    return [d[mask] for d in data]


def load_bruker(directory: str, ask_convdta: bool = True) -> ExpInfo:
    """Load data and parameters from Bruker format.

    Parameters
    ----------
    directory
        Absolute path to data directory.

    ask_convdta
        If ``True``, the user will be warned that the data should have its
        digitial filter removed prior to importing if the data to be impoprted
        is from an ``fid`` or ``ser`` file. If ``False``, the user is not
        warned.

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

    bruker_dataset = BrukerDataset(dtype, dim, files)
    expinfo = get_expinfo(all_params)
    return

    get_parameters(info)
    for k, v in zip(p.keys(), p.values()):
        info[k] = v
    info['binfmt'] = get_binary_format(info)

    data = import_data(paths=info['bin'].values(), binfmt=info['binfmt'])

    if info['dtype'] == 'fid' and info['dim'] > 1:
        data = \
            reshape_multidim_data(
                remove_zeros(
                    complexify(data), info['shape']
                ), info['shape']
            )

    elif info['dtype'] == 'fid':
        data = complexify(data)

    elif info['dim'] > 1:
        data = reshape_multidim_data(
            remove_zeros(data, info['shape']),
            info['shape']
        )

    info['data'] = data
    info['path'] = d
    info['source'] = 'bruker'
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
