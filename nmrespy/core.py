# core.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 04 Feb 2022 14:12:50 GMT

from pathlib import Path
import numpy as np
from nmrespy import ExpInfo
from nmrespy._misc import FrequencyConverter


class Estimator:
    """Estimation class

    .. note::
       The methods :py:meth:`new_bruker`, :py:meth:`new_synthetic_from_data`
       and :py:meth:`new_synthetic_from_parameters` generate instances
       of the class. The method :py:meth:`from_pickle` loads an estimator
       instance that was previously saved using :py:meth:`to_pickle`.
       While you can manually input the listed parameters
       as arguments to initialise the class, it is more straightforward
       to use one of these.

    Parameters
    ----------
    data
        The data associated with the binary file in `path`.

    datapath
        The path to the directory containing the NMR data.

    expinfo
        Experiment information.
    """
    def __init__(
        self, data: np.ndarray, datapath: Path, expinfo: ExpInfo,
    ) -> None:
        self._data = data
        self._datapath = datapath
        self._expinfo = expinfo
        self._converter = FrequencyConverter(self._expinfo, data.shape)
