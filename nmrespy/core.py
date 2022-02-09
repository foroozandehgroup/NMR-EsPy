# core.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 07 Feb 2022 13:45:41 GMT

from __future__ import annotations
import datetime
import functools
import inspect
from pathlib import Path
from typing import Iterable

import numpy as np

from nmrespy import ExpInfo
from nmrespy._misc import FrequencyConverter
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy.sig import make_fid


class Estimator:
    """Estimation class.

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
        now = datetime.datetime.now()
        self._log = (
            "==============================\n"
            "Logfile for Estimator instance\n"
            "==============================\n"
            f"--> Instance created @ {now.strftime('%d-%m-%y %H:%M:%S')}\n"
        )

    def logger(f: callable) -> callable:
        """Decorator for logging method calls."""
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # The first arg is the class instance.
            # Append to the log text.
            args[0]._log += f"--> `{f.__name__}` {args[1:]} {kwargs}\n"
            return f(*args, **kwargs)
        return wrapper

    @classmethod
    def new_synthetic_from_parameters(
        cls, parameters: np.ndarray, expinfo: ExpInfo, pts: Iterable[int], *,
        snr: float = 30.0,
    ) -> Estimator:
        """Generate an instance of :py:class:`Estimator` from an array of oscillator
        parameters.

        Parameters
        ----------
        parameters
            Parameter array with the following structure:

            * **1-dimensional data:**

              .. code:: python

                 parameters = numpy.array([
                    [a_1, φ_1, f_1, η_1],
                    [a_2, φ_2, f_2, η_2],
                    ...,
                    [a_m, φ_m, f_m, η_m],
                 ])

            * **2-dimensional data:**

              .. code:: python

                 params = numpy.array([
                    [a_1, φ_1, f1_1, f2_1, η1_1, η2_1],
                    [a_2, φ_2, f1_2, f2_2, η1_2, η2_2],
                    ...,
                    [a_m, φ_m, f1_m, f2_m, η1_m, η2_m],
                 ])

        expinfo
            Experiment information

        pts
            The number of points the signal comprises in each dimension.

        snr
            The signal-to-noise ratio. If ``None`` then no noise will be added
            to the FID.

        Returns
        -------
        estimator: :py:class:`Estimator`"""
        func_name = inspect.getframeinfo(inspect.currentframe()).function
        sanity_check(func_name, ("expinfo", expinfo, sfuncs.check_expinfo))

        dim = expinfo.unpack("dim")
        sanity_check(
            func_name,
            ("parameters", parameters, sfuncs.check_parameter_array, (dim,)),
            ("pts", pts, sfuncs.check_points, (dim,)),
            ("snr", snr, sfuncs.check_positive_float, (), True),
        )

        data = make_fid(parameters, expinfo, pts, snr=snr)[0]

    @logger
    def to_pickle(self, path="./estimator", force_overwrite=False, fprint=True):
        """Converts the class instance to a byte stream using Python's
        "Pickling" protocol, and saves it to a .pkl file.

        Parameters
        ----------
        path : str, default: './estimator'
            Path of file to save the byte stream to. **DO NOT INCLUDE A
            `'.pkl'` EXTENSION!** `'.pkl'` is added to the end of the path
            automatically.

        force_overwrite : bool, default: False
            Defines behaviour if ``f'{path}.pkl'`` already exists:

            * If `force_overwrite` is set to `False`, the user will be prompted
              if they are happy overwriting the current file.
            * If `force_overwrite` is set to `True`, the current file will be
              overwritten without prompt.

        fprint : bool, default: True
            Specifies whether or not to print infomation to the terminal.

        Notes
        -----
        This method complements :py:meth:`from_pickle`, in that
        an instance saved using :py:meth:`to_pickle` can be recovered by
        :py:func:`~nmrespy.load.pickle_load`.
        """

        checker = ArgumentChecker()
        checker.stage(
            (path, "path", "str"),
            (force_overwrite, "force_overwrite", "bool"),
            (fprint, "fprint", "bool"),
        )
        checker.check()

        # Get full path
        path = Path(path).resolve()
        # Append extension to file path
        path = path.parent / (path.name + ".pkl")
        # Check path is valid (check directory exists, ask user if they are
        # happy overwriting if file already exists).
        pathres = PathManager(path.name, path.parent).check_file(force_overwrite)
        # Valid path, we are good to proceed
        if pathres == 0:
            pass
        # Overwrite denied by the user. Exit the program
        elif pathres == 1:
            exit()
        # pathres == 2: Directory specified doesn't exist
        else:
            raise ValueError(
                f"{RED}The directory implied by path does not" f"exist{END}"
            )

        with open(path, "wb") as fh:
            pickle.dump(self, fh, pickle.HIGHEST_PROTOCOL)

        if fprint:
            print(f"{GRE}Saved instance of Estimator to {path}{END}")

       return cls(data=data, datapath=None, expinfo=expinfo)
