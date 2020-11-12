import os
import pickle
import re

import numpy as np

from .core import NMREsPyBruker
from ._cols import *
if USE_COLORAMA:
    import colorama
from ._errors import *
from ._misc import get_yn

def import_bruker_fid(dir, ask_convdta=True):
    """ Create a new estimation instance containing Bruker FID data

    Parameters
    ----------
    dir : str
        Absolute path to data directory.

    ask_convdta : bool, optional
        If ``True`` (default), the user will be warned that the data should
        have its digitial filter removed prior to importing. If ``False``,
        the user is not warned.

    Returns
    -------
    :py:class:`nmrespy.core.NMREsPyBruker`

    Notes
    -----
    *Directory Requirements* \ The path specified by `dir` should contain the
    following files depending on whether you are importing 1- or 2-dimesnional
    data:

    * 1D data

      - ``dir/fid``
      - ``dir/acqus``

    * 2D data

      - ``dir/ser``
      - ``dir/acqus``
      - ``dir/acqu2s``

    *Digital Filters* \ Make sure the path specified corresponds to data
    which has had its digital filter removed.
    To do this, open the data you wish to analyse in TopSpin,
    and enter ``convdta`` in the bottom-left command line.
    You will be prompted to enter a value for the new data
    directory. It is this value you should use in `dir`,
    not the one corresponding to the original (digitally filtered)
    signal.
    """

    # Show a warning checking that the data being imported has had
    # |convdta| applied to it. Prompts user to proceed or quit
    if ask_convdta:
        prompt = f'{O}WARNING: It is necessary that the raw data you' \
                 + ' import has been digitally filtered, using the <convdta>' \
                 + ' command in TopSpin. If you have not done this, please do' \
                 + ' so before proceeding\nIf you wish to still proceed,' \
                 + f' enter [y]\nIf you wish to quit, enter [n]:{END} '

        get_yn(prompt)

    if not os.path.isdir(dir):
        raise IOError(f'\n{R}directory {dir} doesn\'t exist!{END}')

    # paths to binary files and acquisition parameter files
    bin_fid = os.path.join(dir, 'fid')
    bin_ser = os.path.join(dir, 'ser')
    acqus = os.path.join(dir, 'acqus')
    acqu2s = os.path.join(dir, 'acqu2s')
    acqu3s = os.path.join(dir, 'acqu3s')

    # 1D data
    if os.path.isfile(bin_fid):
        dim = 1
        binf = bin_fid
        acqus_files = [acqus]

    # 2D data
    elif os.path.isfile(bin_ser):
        dim = 2
        binf = bin_ser
        acqus_files = [acqus, acqu2s]

    else:
        raise InvalidDirectoryError(dir)

    # >2D data (not supported)
    if os.path.isfile(acqu3s):
        raise MoreThanTwoDimError()

    # check all necessary parameter files are present
    _check_param_files(acqus_files)

    # endianess of binary files
    bytorda = _get_param('BYTORDA', acqus, type_=int)
    # data type (4-bit int or 8-bit float) of binary files
    dtypa = _get_param('DTYPA', acqus, type_=int)
    nuc = re.search('<(.+?)>', _get_param('NUC1', acqus)).group(1),
    sw = _get_param('SW_h', acqus, type_=float),
    off = _get_param('O1', acqus, type_=float),
    sfo = _get_param('SFO1', acqus, type_=float),

    if dim == 2:
        n_ind = _get_param('TD', acqu2s, type_=int)
        nuc += re.search('<(.+?)>', _get_param('NUC1', acqu2s)).group(1),
        sw += _get_param('SW_h', acqu2s, type_=float),
        off += _get_param('O2', acqus, type_=float),
        sfo += _get_param('SFO2', acqus, type_=float),

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

    # import binary file as numpy array and append to info
    with open(binf, 'rb') as fh:
        fid = np.frombuffer(fh.read(), dtype=fmt)

    fid = fid[::2] + 1j * fid[1::2]
    if dim == 1:
        n = fid.shape
    if dim == 2:
        # total no. of data points (dim1 x dim2)
        n_tot = fid.shape[0]
        # number of data points in direct dimension
        n_dir = int(n_tot / n_ind)
        fid = fid.reshape(n_ind, n_dir)
        # remove block of zeros that is typical after convdta of 2D signal
        fid = _trim_convdta(fid)
        n = fid.shape

    return NMREsPyBruker('raw', fid, dir, sw, off, n, sfo, nuc, bytorda, dtypa,
                         dim)


def import_bruker_pdata(dir):
    """ Create a new estimation instance containing Bruker processed data

    Parameters
    ----------
    dir : str
        Absolute path to data directory.

    Returns
    -------
    :py:class:`nmrespy.core.NMREsPyBruker`

    Notes
    -----
    *Directory Requirements* \ The path specified by `dir` should contain the
    following files depending on whether you are importing 1- or 2-dimesnional
    data:

    * 1D data

      - ``dir/1r``
      - ``dir/1i`` (optional)
      - ``dir/../../acqus``
      - ``dir/procs``

    * 2D data

      - ``dir/2rr``
      - ``dir/2ri`` (optional)
      - ``dir/2ir`` (optional)
      - ``dir/2ii`` (optional)
      - ``dir/../../acqus``
      - ``dir/../../acqu2s``
      - ``dir/procs``
      - ``dir/proc2s``
    """

    if os.path.isdir(dir) is not True:
        raise IOError(f'\n{R}directory {dir} doesn\'t exist!{END}')

    # directory that acqus and acqu2s found relative to the pdata dir
    twoup = os.path.dirname(os.path.dirname(dir))

    # check data dimension and get paths to binary files and parameter
    # files (acqus, acqu2s, procs, proc2s)
    bin_1r = os.path.join(dir, '1r')
    bin_2rr = os.path.join(dir, '2rr')
    bin_3rrr = os.path.join(dir, '3rrr')
    bin_4rrrr = os.path.join(dir, '4rrrr')

    # 1D data
    if os.path.isfile(bin_1r):
        dim = 1
        # binary files dictionary
        bin_files = {'1r': bin_1r}
        # check for imaginary data and append to dictionary if found
        bin_1i = os.path.join(dir, '1i')
        if os.path.isfile(bin_1i):
            bin_files['1i'] = bin_1i

        # parameter file paths
        acqus = os.path.join(twoup, 'acqus')
        procs = os.path.join(dir, 'procs')
        param_files = [acqus, procs]

    # 2D data
    elif os.path.isfile(bin_2rr):
        dim = 2
        bin_files = {'2rr': bin_2rr}
        # check whether other components (imaginary parts) are present
        labels = ['2ir', '2ri', '2ii']
        paths = [os.path.join(dir, lab) for lab in labels]
        for label, path in zip(labels, paths):
            if os.path.isfile(path):
                # if file present, append to dictionary of paths
                bin_files[label] = path

        # parameter file paths
        acqus = os.path.join(twoup, 'acqus')
        acqu2s = os.path.join(twoup, 'acqu2s')
        procs = os.path.join(dir, 'procs')
        proc2s = os.path.join(dir, 'proc2s')
        param_files = [acqus, acqu2s, procs, proc2s]

    # >2D data (not supported)
    elif os.path.isfile(bin_3rrr) or os.path.isfile(bin_4rrrr):
        raise MoreThanTwoDimError()

    else:
        raise InvalidDirectoryError(dir)

    # check all necessary parameter files are present
    _check_param_files(param_files)

    # get necessary parameters from files
    bytordp = _get_param('BYTORDP', procs, type_=int)
    dtypp = _get_param('DTYPP', procs, type_=int)
    nuc = re.search('<(.+?)>', _get_param('NUC1', acqus)).group(1),
    sw = _get_param('SW_h', acqus, type_=float),
    off = _get_param('O1', acqus, type_=float),
    sfo = _get_param('SFO1', acqus, type_=float),
    si = _get_param('SI', procs, type_=int),

    if dim == 2:
        nuc += re.search('<(.+?)>', _get_param('NUC1', acqu2s)).group(1),
        sw += _get_param('SW_h', acqu2s, type_=float),
        off += _get_param('O2', acqus, type_=float),
        sfo += _get_param('SFO2', acqus, type_=float),
        si += _get_param('SI', proc2s, type_=int),

    if bytordp == 1 and dtypp == 2:
        dtype = '>f8'
    elif bytordp == 1 and dtypp == 0:
        dtype = '>i4'
    if bytordp == 0 and dtypp == 2:
        dtype = '<f8'
    elif bytordp == 0 and dtypp == 0:
        dtype = '<i4'
    # There should not be any possibility...

    # for each binary file, import as numpy array and append to info
    pdata = {}
    for key in bin_files:
        path = bin_files[key]

        with open(path, 'rb') as fh:
            pdata[key] = np.frombuffer(fh.read(), dtype=dtype)
        if dim == 2:
            pdata[key] = pdata[key].reshape(si[0], si[1])

    return NMREsPyBruker('pdata', pdata, dir, sw, off, si, sfo, nuc, bytordp,
                         dtypp, dim)


def pickle_load(fname, dir='.'):
    """Deserialises and imports a byte stream contained in a specified file, using
    Python's "Pickling" protocol.

    Parameters
    ----------
    fname : str
        Name of the file containing the serialised object. The extension
        '.pkl' may be included or omitted.
    dir : str, deafult: '.'
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

    path_ = os.path.join(dir, fname)

    if os.path.isfile(path_):
        with open(path_, 'rb') as file:
            inst = pickle.load(file)
        if isinstance(inst, NMREsPyBruker):
            print(f'{G}Loaded contents of {path_}{END}')
            return inst
        else:
            print(f'{O}The contents of {path_} how been successfully loaded,'
                  f' though the object returned is not an intance of'
                  f' NMREsPyBruker. I cannot guarantte this will behave as'
                  f' desired. You have been warned...{END}')
            return inst

    else:
        raise FileNotFoundError(f'{R}Does not exist!{END}')



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

    fails = []
    for path in paths:
        if not os.path.isfile(path):
            fails.append(path)
    # if one or more parameter files were not found, raise error
    if fails:
        msg = f'\n{R}The following necessary parameter files were not' \
              + f'found:\n' + '\n'.join(fails) + f'{END}'
        raise IOError(msg)


def _get_param(param, path, type_=str):
    """
    _get_param(param, path, type_=str)

    Decription
    ----------
    Retrieve parameter from Bruker file. Note a string is returned by
    default.

    Parameters
    ----------
    param - str
        The name of the parameter
    path - str
        The path to the parameter file
    type_ - class
        Desired type of the result. Defaults to str.

    Returns
    -------
    par - str
        The value of the desired parameter
    """

    f = open(path)
    lines = f.readlines()
    f.close()

    motif = r'##$' + param
    for l in lines:
        if motif in l:
            # lines are of format '##$param= VALUE'
            par = l.split(' ')[1]
            return type_(par)

    msg = f'{R}Could not find parameter {param} in file {path}{END}'
    raise ParameterNotFoundError(msg)
