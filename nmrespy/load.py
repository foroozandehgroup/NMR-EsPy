import os
from pathlib import Path
import pickle
import re

import numpy as np
from numpy.fft import ifftshift, ifft

from .core import NMREsPyBruker
import nmrespy._cols as cols
if cols.USE_COLORAMA:
    import colorama
import nmrespy._errors as errors
import nmrespy._misc as misc


def import_bruker(directory, ask_convdta=True):
    """ Create a new estimation instance containing Bruker data

    Parameters
    ----------
    directory : str
        Absolute path to data directory.

    ask_convdta : bool, optional
        If ``True`` (default), the user will be warned that the data should
        have its digitial filter removed prior to importing if the data to be
        impoprted is from an ``fid`` or ``ser`` file. If ``False``, the user is
        not warned.

    Returns
    -------
    :py:class:`nmrespy.core.NMREsPyBruker`

    Notes
    -----
    .. note::
       2-dimension data is not supported yet - but will be soon!

    *Directory Requirements* \ The path specified by `directory` should
    contain the following files depending on whether you are importing 1- or
    2-dimesnional data, and whether you are importing raw FID data or
    processed data:

    * 1D data

      - Raw FID

        + ``directory/fid``
        + ``directory/acqus``

      - Processed data

        + ``directory/1r``
        + ``directory/acqus``
        + ``directory/procs``


    * 2D data

      - Raw FID

        + ``directory/ser``
        + ``directory/acqus``
        + ``directory/acqu2s``

      - Processed data

        + ``directory/2rr``
        + ``directory/acqus``
        + ``directory/acqu2s``
        + ``directory/procs``
        + ``directory/proc2s``

    *Digital Filters* \ If you are importing raw FID data, make sure the path
    specified corresponds to an ``fid`` or ``ser`` file which has had its
    digital filter removed. To do this, open the data you wish to analyse in
    TopSpin, and enter ``convdta`` in the bottom-left command line. You will be
    prompted to enter a value for the new data directory. It is this value you
    should use in `directory`, not the one corresponding to the original
    (uncorrected) signal.
    """

    # Show a warning checking that the data being imported has had
    # |convdta| applied to it. Prompts user to proceed or quit
    if ask_convdta:
        prompt = f'{cols.O}WARNING: It is necessary that the FID you' \
                 + ' import has been digitally filtered, using the <convdta>' \
                 + ' command in TopSpin. If you have not done this, please do' \
                 + ' so before proceeding\nIf you wish to proceed,' \
                 + f' enter [y]\nIf you wish to quit, enter [n]:{cols.END} '

    dir_ = Path(directory)

    if not dir_.is_dir():
        raise IOError(f'\n{cols.R}directory {directory} doesn\'t exist!'
                      f'{cols.END}')


    # 1D data - check for fid file
    if (dir_ / 'fid').is_file():
        if ask_convdta:
            misc.get_yn(prompt)
        dim, dtype = 1, 'fid'
        binf = dir_ / 'fid'
        param_files = {'acqus': dir_ / 'acqus'}

    # 1D data - check for 1r file
    elif (dir_ / '1r').is_file():
        dim, dtype = 1, 'pdata'
        binf = dir_ / '1r'
        param_files = {
            'acqus': dir_.parents[1] / 'acqus',
            'procs': dir_ / 'procs',
        }

    # 2D data - check for ser file
    elif (dir_ / 'ser').is_file():
        # if > 2D, raise error
        if (dir_ / 'acqu3s').is_file():
            raise errors.MoreThanTwoDimError()
        if ask_convdta:
            misc.get_yn(prompt)
        dim, dtype = 2, 'fid'
        binf = dir_ / 'ser'
        param_files = {
            'acqus': dir_ / 'acqus',
            'acqu2s': dir_ / 'acqu2s',
        }

    # 2D data - check for 2rr file
    elif (dir_ / '2rr').is_file():
            dim, dtype = 2, 'pdata'
            binf = dir_ / '1r'
            param_files = {
                'acqus': dir_.parents[1] / 'acqus',
                'acqu2s': dir_.parents[1] / 'acqu2s',
                'procs': dir_ / 'procs',
                'proc2s': dir_ / 'proc2s',
            }

    # > 2D pdata directory: raise error
    elif (dir_ / '3rrr').is_file() or (dir_ / '4rrrr').is_file():
        raise errors.MoreThanTwoDimError()

    else:
        raise errors.InvalidDirectoryError(directory)

    # check all necessary parameter files are present
    _check_param_files(param_files)

    if 'procs' in param_files.keys():
        # endianess of binary file
        bytorda = _get_param('BYTORDP', param_files['procs'], type_=int)
        # data type (4-bit int or 8-bit float) of binary files
        dtypa = _get_param('DTYPP', param_files['procs'], type_=int)

    else:
        bytorda = _get_param('BYTORDA', param_files['acqus'], type_=int)
        dtypa = _get_param('DTYPA', param_files['acqus'], type_=int)

    nuc = [
        re.search('<(.+?)>', _get_param('NUC1', param_files['acqus'])).group(1)
    ]
    sw = [_get_param('SW_h', param_files['acqus'], type_=float)]
    off = [_get_param('O1', param_files['acqus'], type_=float)]
    sfo = [_get_param('SFO1', param_files['acqus'], type_=float)]

    if dim == 2:
        # points in indirect dimension
        # need to get this in order the reshape the array after importing
        # from binary file
        if dtype == 'fid':
            n_ind = _get_param('TD', param_files['acqu2s'], type_=int)
        else:
            n_ind = _get_param('SI', param_files['proc2s'], type_=int)

        nuc.append(
            re.search(
                '<(.+?)>', _get_param('NUC1', param_files['acqu2s'])
            ).group(1)
        )
        sw.append(_get_param('SW_h', param_files['acqu2s'], type_=float))
        off.append(_get_param('O2', param_files['acqu2s'], type_=float))
        sfo.append(_get_param('SFO2', param_files['acqu2s'], type_=float))

    # check data format:
    # 8-bit float or 4-bit int (dtypa)
    # big or little endian (bytorda)
    if bytorda == 1 and dtypa == 2:
        fmt = '>f8'
    elif bytorda == 1 and dtypa == 0:
        fmt = '>i4'
    elif bytorda == 0 and dtypa == 2:
        fmt = '<f8'
    elif bytorda == 0 and dtypa == 0:
        fmt = '<i4'
    else:
        # this should not happen...
        raise ValueError('bytorda and/or dtypa not anticipated...')

    # import binary file as numpy array
    with open(binf, 'rb') as fh:
        data = np.frombuffer(fh.read(), dtype=fmt)


    if dtype == 'fid':
        # recast 2N-length array into N-length complex array
        data = data[::2] + 1j * data[1::2]

    if dim == 1:
        if dtype == 'pdata':
            # IFT flipped real spectrum and halve signal size
            data = ifft(ifftshift(data[::-1]))[:int(data.size // 2)]


    elif dim == 2:
        # total no. of data points (dim1 x dim2)
        n_tot = data.shape[0]
        # number of data points in direct dimension
        n_dir = int(n_tot / n_ind)
        data = data.reshape(n_ind, n_dir)

        if dtype == 'fid':
            data = _trim_convdta(data)

        elif dtype == 'pdata':
            # ================================================
            # TODO: Need to convert pdata array to time-domain
            # ================================================
            raise errors.TwoDimUnsupportedError()

    n = list(data.shape)

    return NMREsPyBruker(
        dtype, data, dir_, sw, off, n, sfo, nuc, bytorda, dtypa, dim,
    )


def pickle_load(fname, directory='.'):
    """Deserialises and imports a byte stream contained in a specified file,
    using Python's "Pickling" protocol.

    Parameters
    ----------
    fname : str
        Name of the file containing the serialised object. The extension
        '.pkl' may be included or omitted.
    directory : str, deafult: '.'
        Name of the directory the file is contained in.

    Returns
    -------
    :py:class:`nmrespy.core.NMREsPyBruker`

    Notes
    -----
    .. warning::
       `From the Python docs:`

       "The pickle module is not secure. Only unpickle data you trust.
       It is possible to construct malicious pickle data which will execute
       arbitrary code during unpickling. Never unpickle data that could have
       come from an untrusted source, or that could have been tampered with."

       You should only use :py:func:`~nmrespy.load.pickle_load` on files that
       you are 100% certain were generated using
       :py:meth:`~nmrespy.core.NMREsPyBruker.pickle_save`. If you use
       :py:func:`~nmrespy.load.pickle_load` on a .pkl file, and the resulting
       output is not an instance of :py:class:`~nmrespy.core.NMREsPyBruker`,
       you will be warned.
    """

    if fname[-4:] == '.pkl':
        pass
    elif '.' in fname:
        raise ValueError(f'{R}fname: {fname} - Unexpected file'
                         f' extension.{END}')
    else:
        fname += '.pkl'

    dir_ = Path(directory)

    path = dir_ / fname

    if path.is_file():
        with open(path, 'rb') as fh:
            obj = pickle.load(fh)

        if isinstance(obj, NMREsPyBruker):
            print(f'{cols.G}Loaded contents of {path}{cols.END}')
            return obj
        else:
            print(f'{cols.O}The contents of {path} have been successfully'
                  f' loaded, though the object returned is not an intance of'
                  f' NMREsPyBruker. I cannot guarantee this will behave as'
                  f' desired. You have been warned...{cols.END}')
            return obj

    else:
        raise FileNotFoundError(f'{cols.R}Does not exist!{cols.END}')



def _trim_convdta(fid):
    m, n = fid.shape
    # indices where first slice in direct domain are zero
    idx = np.where(fid[0, :] == 0)[0]
    for i in idx:
        # check if all elements are consectutive integers
        if np.all(np.diff(idx) == 1):
            if np.array_equal(np.zeros((m,n-i)), fid[:, i:]):
                return fid[:, :i]

    return fid


def _check_param_files(paths):
    """Check a list of paths exist.

    Parameters
    ----------
    paths : list
        List of `pathlib.Path <https://docs.python.org/3/library/\
        pathlib.html#pathlib.Path>`_ instances.

    Raises
    ------
    IOError
        If any of entries in ``paths`` does not correspond to a valid path on
        the system.
    """

    fails = []
    for path in paths.values():
        if not path.is_file():
            fails.append(str(path))
    # if one or more parameter files were not found, raise error
    if fails:
        msg = f'\n{R}The following necessary parameter files were not' \
              + f'found:\n' + '\n'.join(fails) + f'{END}'
        raise IOError(msg)


def _get_param(param_name, path, type_=str):
    """
    _get_param(param, path, type_=str)

    Decription
    ----------
    Retrieve parameter from Bruker parameter file. Note a string is returned by
    default.

    Parameters
    ----------
    param_name : str
        The name of the parameter
    path : str
        The path to the parameter file
    type_ : class
        Desired type of the result. Defaults to str.

    Returns
    -------
    param - str
        The value of the desired parameter
    """

    with open(path, 'r') as fh:
        lines = fh.readlines()

    motif = r'##$' + param_name
    for l in lines:
        if motif in l:
            # lines are of format '##$param= VALUE'
            param = re.sub('\n', '', l.split(' ')[1])
            return type_(param)

    raise errors.ParameterNotFoundError(param_name, path)
