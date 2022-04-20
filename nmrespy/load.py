# load.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 24 Mar 2022 18:02:41 GMT

"""Provides functionality for importing NMR data."""

import re
from typing import Tuple

import numpy as np
import bruker_utils

from nmrespy import ExpInfo
from nmrespy._colors import RED, ORA, END, USE_COLORAMA
from nmrespy._errors import MoreThanTwoDimError
from nmrespy._misc import get_yes_no

if USE_COLORAMA:
    import colorama
    colorama.init()


class _BrukerDatasetForNmrespy(bruker_utils.BrukerDataset):
    def __init__(self, directory: str) -> None:
        super().__init__(directory)

    @property
    def expinfo(self) -> ExpInfo:
        """Extract :py:class:`ExpInfo` from the dataset."""
        acqusfiles = sorted([f for f in self.valid_parameter_filenames if "acqu" in f])
        sw, offset, sfo, nuclei = [[] for _ in range(4)]
        for f in acqusfiles:
            params = self.get_parameters(filenames=f)[f]
            sw.append(float(params["SW_h"]))
            offset.append(float(params["O1"]))
            sfo.append(float(params["SFO1"]))
            nuclei.append(re.match("^<(.+?)>$", params["NUC1"]).group(1))

        return ExpInfo(
            dim=len(acqusfiles),
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuclei=nuclei,
            **{"parameters": self.get_parameters()},
        )


def load_bruker(directory: str, ask_convdta: bool = True) -> Tuple[np.ndarray, ExpInfo]:
    """Load data and experiment information from Bruker format.

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
    data
        The associated data.

    expinfo
        Experiment information. As well as the normal associated properties
        (see :py:class:`nmrespy.ExpInfo`), the ``parameters`` attribute is a
        dict with parameters from all acqus and procs files associated with the
        dataset.

    Raises
    ------
    OSError
        If the directory specified does not exist.

    ValueError
        If the directroy specified does not have the requisite data and
        parameter files expected for a Bruker dataset.

    MoreThanTwoDimError
        If the dataset specified has more than 2 dimensions.

    Notes
    -----
    **Directory Requirements**

    There are certain file paths expected to be found relative to ``directory``
    which contain the data and parameter files. Here is an extensive list of
    the paths expected to exist, for different data dimensions, and
    data types (raw time-domain data or processed data):

    * 1D data

      - Raw FID

        + ``directory/fid``
        + ``directory/acqus``

      - Processed data

        + ``directory/1r``
        + ``directory/../../acqus``
        + ``directory/procs``


    * 2D data

      - Raw FID

        + ``directory/ser``
        + ``directory/acqus``
        + ``directory/acqu2s``

      - Processed data

        + ``directory/2rr``
        + ``directory/../../acqus``
        + ``directory/../../acqu2s``
        + ``directory/procs``
        + ``directory/proc2s``

    **Digital Filters**

    If you are importing raw FID data, make sure the path
    specified corresponds to an ``fid`` or ``ser`` file which has had its
    group delay artefact removed. To do this, open the data you wish to analyse in
    TopSpin, and enter ``convdta`` in the bottom-left command line. You will be
    prompted to enter a value for the new data directory. It is this value you
    should use in ``directory``, not the one corresponding to the original
    (uncorrected) signal.

    **For Development**

    .. todo::

        Incorporate functionality to phase correct data to remove group delay.
        This would circumvent the need to ask the user to ensure they have
        performed ``convdta`` prior to importing.

    Example
    -------
    .. code:: pycon

        >>> from nmrespy.load import load_bruker
        >>> load_bruker
        <function load_bruker at 0x7f21fd526bf8>
        >>> path = 'tests/data/1'
        >>> data, expinfo = load_bruker(path)
        You should ensure you data has had its group delay artefact removed.
        Prior to dealing with the data in NMR-EsPy, you should call the
        `CONVDTA` command on the dataset. If this has already been done, feel
        free to proceed. If not, quit the program.
        Continue?
        Enter [y] or [n]
        y
        >>> data.shape
        (32692,)
        >>> expinfo.sw
        (5494.50549450549,)
        >>> expinfo.offset
        (2249.20599998768,)
        >>> expinfo.sfo
        (500.132249206,)
        >>> expinfo.nuclei
        ('1H',)
        >>> for key in expinfo.parameters.keys():
        ...     print(key)
        ...
        acqus
        >>> for key, value in expinfo.parameters['acqus'].items():
        ...     print(f'{key} --> {value}')
        ...
        ACQT0 --> -2.26890760309556
        AMP --> [100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                 100, 100]
        AMPCOIL --> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0]
        AQSEQ --> 0
        AQ_mod --> 3
        AUNM --> <au_zg>
        AUTOPOS --> <>
        BF1 --> 500.13
        BF2 --> 500.13
        --snip--
        ZL3 --> 120
        ZL4 --> 120
        scaledByNS --> no
        scaledByRG --> no
    """
    try:
        dataset = _BrukerDatasetForNmrespy(directory)

    except Exception as exc:
        raise type(exc)(f"{RED}{str(exc)}{END}")

    if dataset.dim > 2:
        raise MoreThanTwoDimError()

    if dataset.dtype == "fid" and ask_convdta:
        msg = (
            f"{ORA}You should ensure you data has had its group delay "
            "artefact removed. Prior to dealing with the data in NMR-EsPy, "
            "you should call the `CONVDTA` command on the dataset. If this "
            "has already been done, feel free to proceed. If not, quit the "
            f"program.\nContinue?{END}"
        )

        response = get_yes_no(msg)
        if not response:
            exit()

    data = dataset.data
    expinfo = dataset.expinfo
    expinfo._default_pts = data.shape

    return data, expinfo


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
