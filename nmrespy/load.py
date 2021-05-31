# load.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk

"""Provides functionality for importing NMR data."""

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
    # |convdta| applied to it.
    if ask_convdta:
        convdta_prompt = (
            f'{cols.OR}WARNING: It is necessary that the FID you import has '
            'been digitally filtered, using the <convdta> command in TopSpin. '
            'If you have not done this, please do so before proceeding\nIf '
            'you wish to proceed, enter [y]\nIf you wish to quit, enter [n]: '
            f'{cols.END} '
        )

    # pathlib.Path instance of directory
    d = Path(directory)

    # Check that the directory actually exists...
    if not d.is_dir():
        raise IOError(f'\n{cols.R}directory {directory} doesn\'t exist!'
                      f'{cols.END}')

    def check_file_paths(path_info):
        # Goes through the entries in path_info and determines whether
        # the binary file path and and parameter file path(s) in each
        # entry exist. Returns the path_info entry is all paths are valid.
        # Retruns None if no entries in path_info are valid.
        for info in path_info:
            # Check binary file path exists
            if info['bin'].is_file():
                # Check parameter file path(s) exist
                for param_file in info['param'].values():
                    if param_file.is_file():
                        pass
                    else:
                        # At least one parameter file is not found.
                        # Implies no entries in path_info will be valid.
                        return None
                # An entry in path_info has all the requisite files!
                return info
        # None of the valid binary file names exist in the directory.
        return None

    # Valid combinations of binary file pathname and parameter
    # file pathname(s)
    # 'bin' - binary file pathname
    # 'param' - parameter file pathname
    # 'dim' - data dimension
    # 'dtype' - type of data (raw FID data or processed data)
    path_info = [
        # 1D FID
        {
            'bin': d / 'fid',
            'param': {
                'acqus': d / 'acqus',
            },
            'dim': 1,
            'dtype': 'fid',
        },
        # 2D FID
        {
            'bin': d / 'ser',
            'param': {
                'acqus': d / 'acqus',
                'acqu2s': d / 'acqu2s',
            },
            'dim': 2,
            'dtype': 'fid',
        },
        # 1D Processed data
        {
            'bin': d / '1r',
            'param': {
                'acqus': d.parents[1] / 'acqus',
                'procs': d / 'procs',
            },
            'dim': 1,
            'dtype': 'pdata',
        },
        # 2D Processed data
        {
            'bin': d / '2rr',
            'param': {
                'acqus': d.parents[1] / 'acqus',
                'acqu2s': d.parents[1] / 'acqu2s',
                'procs': d / 'procs',
                'proc2s': d / 'proc2s',
            },
            'dim': 2,
            'dtype': 'pdata',
        },
    ]

    # Either an entry from path_dict or None
    info = check_file_paths(path_info)

    # Required files not present in specified dictionary
    if info is None:
        raise errors.InvalidDirectoryError(directory)

    # If the data to be imported is a raw FID, alert the user about the
    # necessity to correct for the digital filter
    if info['dtype'] == 'fid' and ask_convdta:
        get_yes_no(convdta_prompt)

    # The checks in check_file_paths do not discriminate between
    # 2D FID data and higher-dimensional FID data. All directories
    # will have ser, acqus and acqu2s files. Check if there is an
    # acqu3s file implying the dimension is greater than or equal
    # to 3.
    elif (info['dim'] == 2
          and info['dtype'] == 'fid'
          and (d / 'acqu3s').is_file()):
        raise errors.MoreThanTwoDimError()

    # --- Search parameter files to get required info --------------------
    if info['dtype'] == 'pdata':
        # Processed data - look in procs for binary file structure
        unit_name = 'DTYPP'
        endian_name = 'BYTORDP'
        file = info['param']['procs']

    else:
        # FID data - look in acqus for binary file structure
        unit_name = 'DTYPA'
        endian_name = 'BYTORDA'
        file = info['param']['acqus']

    # Endianess of binary file.
    # Will be 0 (little endian) or 1 (big endian).
    endian = '<' if int(_get_param(endian_name, file)) == 0 else '>'
    # Size of each unit in binary file.
    # Will be 0 (4-byte integer) or 2 (8-byte float).
    size = 'i4' if int(_get_param(unit_name, file)) == 0 else 'f8'
    # Format to be given to numpy fromfile
    # '<i4', '>i4', '<f8' or '>f8'
    fmt = endian + size

    # Other parameters of interest:
    # Nucleus, sweep width, transmitter offset, transmitter frequency.
    nuc, sw, off, sfo = [], [], [], []
    for i in range(1, info['dim'] + 1):
        j = '' if i == 1 else str(i)  # 1 -> '', 2 -> '2'
        file = info['param'][f'acqu{j}s']

        # Nucleus is indicated by <1H>, <13C>, etc.
        nuc.append(re.search('<(.+?)>', _get_param('NUC1', file)).group(1))
        sw.append(float(_get_param('SW_h', file)))
        off.append(float(_get_param('O1', file)))
        sfo.append(float(_get_param('SFO1', file)))

    # If data is 2D, need at least one of the dimension sizes in order
    # to reshape the data after loading from the binary file. Get
    # indirect dimension size.
    if info['dim'] == 2:
        p, file = (('TD', 'acqu2s') if info['dtype'] == 'fid'
                   else ('SI', 'proc2s'))
        n_indirect = int(_get_param(p, info['param'][file]))

    # --- Import and format binary file data -----------------------------
    data = np.fromfile(info['bin'], dtype=fmt)

    if info['dtype'] == 'fid':
        # fid and ser files store real and imaginary points consectively
        # i.e. re - im - re - im - etc.
        # Convert the data from a length 2N array -> length N array of
        # complex numbers.
        data = data[::2] + 1j * data[1::2]

        # Applying convdta on a 2D signal leads to a block of zeros at
        # the end of the data, i.e.
        #
        #                   ************0000
        #                   ************0000
        #                   ************0000
        #                   ************0000
        #
        # This needs to be removed!
        if info['dim'] == 2:
            # Reshape array
            data = data.reshape(n_indirect, int(data.size / n_indirect))

            # # Plots a 3-dimensional wireframe of the data. Used for testing
            # # purposes. Should see a plane of 0's right at the end of the
            # # direct dimension
            # import matplotlib.pyplot as plt
            # from mpl_toolkits.mplot3d import Axes3D
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # x, y = tuple(np.arange(s) for s in data.shape)
            # xx, yy = tuple(arr.T for arr in np.meshgrid(x, y))
            # ax.plot_wireframe(xx, yy, data)

            # Data gets flattened by this operation. Need to reshape it
            # N.B. I am assuming that no data point is exactly zero other
            # that those that are expected to be. An error will occur in
            # reshaping is this is not satisfied. If this is something
            # that will need to be accounted for, we'll have to be more
            # robust
            data = data[data != 0]
            data = data.reshape(n_indirect, int(data.size / n_indirect))

            # # Continuation of testing with plot
            # x, y = tuple(np.arange(s) for s in data.shape)
            # xx, yy = tuple(arr.T for arr in np.meshgrid(x, y))
            # ax.plot_wireframe(xx, yy, data, color='k')
            # print('bye')
            # plt.show()

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
            #
            # We will need to figure out how to preprocess each
            # of these in order to generate correctly formed
            # time-domain data.
            # =================================================

    # Dealing with processed data.
    else:
        if info['dim'] == 1:
            # Does the following things:
            # 1. Flips the spectrum (order frequency bins from low to high
            # going left to right)
            # 2. Performs inverse Fourier Transform
            # 3. Retrieves the first half of the signal, which is a conjugate
            # symmetric virtual echo
            # 4. Doubles to reflect loss of imaginary data
            # 5. Zero fills back to original signal size
            # 6. Halves the first point to ensure no baseline shift takes
            # place
            data = 2 * np.hstack(
                (
                    ifft(ifftshift(data[::-1]))[:int(data.size // 2)],
                    np.zeros(int(data.size // 2), dtype='complex'),
                )
            )
            data[0] /= 2

        else:
            # Reshape data, and flip in both dimensions
            data = data.reshape(
                n_indirect, int(data.size / n_indirect),
            )[::-1, ::-1]
            # Inverse Fourier Tranform in both dimensions
            for axis in range(2):
                data = ifft(ifftshift(data, axes=axis), axis=axis)
            # Slice signal in half in both dimensions
            data = 4 * data[:int(data.shape[0] // 2), :int(data.shape[1] // 2)]

    # Compile a dictionary of parameters
    result = {
        'source': f"bruker_{info['dtype']}",
        'data': data,
        'directory': d,
        'sweep_width': sw,
        'offset': off,
        'transmitter_frequency': sfo,
        'nuclei': nuc,
        'binary_format': fmt,
    }

    return result


def _get_param(name, path):
    """Retrieve parameter from Bruker acqus or procs file.

    Parameters
    ----------
    name: str
        The name of the parameter.

    path: pathlib.Path
        The path to the parameter file.

    Returns
    -------
    param: str
        The value of the desired parameter as a string.

    Notes
    -----
    .. warning::
       This will not work on every parameter found in acqus/procs.
       Any parameters that are defined over multiple lines will not
       be correctly returned. There is no need for this in nmrespy (yet),
       so including this functionality has been neglected.
    """
    # Open the parameter file and read in lines
    try:
        with open(path, 'r') as fh:
            lines = fh.readlines()

    except Exception as e:
        raise e

    # Lines are of the format:
    # '##$name= param\n'
    # Search each line for ##$name
    motif = f'##${name}'
    for line in lines:
        if motif in line:

            def _cat_list(list):
                """Concatenates elements of a list to a string"""
                for string in list:
                    try:
                        result += string
                    except NameError:
                        result = string

                return result

            # Split line at whitespace and discard first component:
            # '##$name= param\n' -> 'param\n'
            # Remove trailing newline character:
            # 'param\n' -> 'param'
            return re.sub('\n', '', _cat_list(line.split(' ')[1:]))

    # '##$name' was not found in the file
    raise errors.ParameterNotFoundError(name, path)
