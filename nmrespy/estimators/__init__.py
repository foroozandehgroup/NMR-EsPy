# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 29 Mar 2022 15:37:01 BST

from __future__ import annotations
from abc import ABCMeta, abstractmethod
import datetime
import functools
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np

from nmrespy import ExpInfo, sig
from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._files import (
    check_saveable_path,
    check_existent_path,
    configure_path,
    open_file,
    save_file,
)
from nmrespy._freqconverter import FrequencyConverter
from nmrespy._misc import copydoc
from nmrespy._result_fetcher import ResultFetcher
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy.freqfilter import Region
from nmrespy.plot import NmrespyPlot, plot_result

if USE_COLORAMA:
    import colorama
    colorama.init()


def logger(f: callable):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        class_instance = args[0]
        class_instance._log += f"--> `{f.__name__}` {args[1:]} {kwargs}\n"
        return f(*args, **kwargs)
    return inner


class Estimator(ExpInfo, metaclass=ABCMeta):
    """Base estimation class."""

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ExpInfo,
        datapath: Optional[Path] = None,
    ) -> None:
        """Initialise a class instance.

        Parameters
        ----------
        data
            The data associated with the binary file in `path`.

        datapath
            The path to the directory containing the NMR data.

        expinfo
            Experiment information.
        """
        self._data = data
        self._datapath = datapath

        super().__init__(
            dim=self._data.ndim,
            sw=expinfo.sw(),
            offset=expinfo.offset(),
            sfo=expinfo.sfo,
            nuclei=expinfo.nuclei,
            fn_mode=expinfo.fn_mode,
        )

        self.default_pts = self._data.shape
        self._results = []
        now = datetime.datetime.now().strftime('%d-%m-%y %H:%M:%S')
        self._log = (
            "=====================\n"
            "Logfile for Estimator\n"
            "=====================\n"
            f"--> Created @ {now}\n"
        )

    @property
    def view_log(self) -> None:
        """View the log for the estimator instance."""
        print(self._log)

    def save_log(
        self,
        path: Union[str, Path] = "./espy_logfile",
        force_overwrite: bool = False,
        fprint: bool = True,
    ) -> None:
        """Save the estimator's log.

        Parameters
        ----------
        path
            The path to save the log to.

        force_overwrite
            If ``path`` already exists, ``force_overwrite`` set to ``True`` will get
            the user to confirm whether they are happy to overwrite the file.
            If ``False``, the file will be overwritten without prompt.

        fprint
            Specifies whether or not to print infomation to the terminal.
        """
        sanity_check(
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
            ("fprint", fprint, sfuncs.check_bool),
        )
        sanity_check(
            ("path", path, check_saveable_path, ("log", force_overwrite)),
        )

        path = configure_path(path, "log")
        save_file(self._log, path, fprint=fprint)

    def converter(self, shape: Optional[Iterable[int]] = None):
        if shape is None:
            shape = self._data.shape
        return FrequencyConverter(self.sfo, self.sw, self.offset, shape)

    @classmethod
    @abstractmethod
    def new_bruker(*args, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def new_synthetic_from_simulation(*args, **kwargs):
        pass

    @logger
    def to_pickle(
        self,
        path: Optional[Union[Path, str]] = None,
        force_overwrite: bool = False,
        fprint: bool = True,
    ) -> None:
        """Save the estimator to a byte stream using Python's pickling protocol.

        Parameters
        ----------
        path
            Path of file to save the byte stream to. `'.pkl'` is added to the end of
            the path if this is not given by the user. If ``None``,
            ``./estimator_<x>.pkl`` will be used, where ``<x>`` is the first number
            that doesn't cause a clash with an already existent file.

        force_overwrite
            Defines behaviour if the specified path already exists:

            * If ``force_overwrite`` is set to ``False``, the user will be prompted
              if they are happy overwriting the current file.
            * If ``force_overwrite`` is set to ``True``, the current file will be
              overwritten without prompt.

        fprint
            Specifies whether or not to print infomation to the terminal.

        See Also
        --------

        :py:meth:`Estimator.from_pickle`
        """
        sanity_check(
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
            ("fprint", fprint, sfuncs.check_bool),
        )
        sanity_check(
            ("path", path, check_saveable_path, ("pkl", force_overwrite), True),
        )

        if path is None:
            x = 1
            while True:
                path = Path(f"estimator_{x}.pkl").resolve()
                if path.is_file():
                    x += 1
                else:
                    break

        path = configure_path(path, "pkl")
        save_file(self, path, binary=True, fprint=fprint)

    @classmethod
    def from_pickle(
        cls,
        path: Union[str, Path],
    ) -> Estimator:
        """Load a pickled estimator instance.

        Parameters
        ----------
        path
            The path to the pickle file.

        Returns
        -------
        estimator : :py:class:`Estimator`

        Notes
        -----
        .. warning::
           `From the Python docs:`

           *"The pickle module is not secure. Only unpickle data you trust.
           It is possible to construct malicious pickle data which will
           execute arbitrary code during unpickling. Never unpickle data
           that could have come from an untrusted source, or that could have
           been tampered with."*

           You should only use :py:meth:`from_pickle` on files that
           you are 100% certain were generated using
           :py:meth:`to_pickle`. If you load pickled data from a .pkl file,
           and the resulting output is not an instance of
           :py:class:`Estimator`, an error will be raised.

        See Also
        --------

        :py:meth:`Estimator.to_pickle`
        """
        sanity_check(("path", path, check_existent_path, ("pkl",)))
        path = configure_path(path, "pkl")
        obj = open_file(path, binary=True)

        if isinstance(obj, __class__):
            return obj
        else:
            raise TypeError(
                f"{RED}It is expected that the object loaded by"
                " `from_pickle` is an instance of"
                f" {__class__.__module__}.{__class__.__qualname__}."
                f" What was loaded didn't satisfy this!{END}"
            )

    @abstractmethod
    def estimate(*args, **kwargs):
        pass

    def get_results(self, indices: Optional[Iterable[int]] = None) -> Iterable[Result]:
        """Obtain a subset of the estimation results obtained.

        By default, all results are returned, in the order in which they are obtained.

        Parameters
        ----------
        indices
            The indices of results to return. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be returned.

        Returns
        -------
        List of results selected.
        """
        sanity_check(
            (
                "indices", indices, sfuncs.check_ints_less_than_n,
                (len(self._results),), True,
            ),
        )
        if indices is None:
            return self._results
        else:
            return [self._results[i] for i in indices]

    @abstractmethod
    def write_result(*args, **kwargs):
        pass

    @copydoc(plot_result)
    @abstractmethod
    def plot_results(*args, **kwargs):
        pass


class Result(ResultFetcher, metaclass=ABCMeta):

    def __init__(
        self,
        timestamp: datetime.datetime,
        signal: np.ndarray,
        expinfo: ExpInfo,
        region: Region,
        result: np.ndarray,
        errors: np.ndarray,
    ) -> None:
        self.__dict__.update(locals())
        self.expinfo.default_pts = self.signal.shape
        super().__init__(self.expinfo)

    @property
    def osc_number(self) -> int:
        """Return the number of oscillators in the result."""
        return self.result.shape[0]

    def get_region(self, unit: str = "hz") -> Iterable[Tuple[float, float]]:
        sanity_check(
            ("unit", unit, sfuncs.check_one_of, ("hz", "ppm"),),
        )
        return self.convert(self.region, f"hz->{unit}")

    def make_fid(
        self,
        pts: Optional[Iterable[int]] = None,
        oscillators: Optional[Iterable[int]] = None,
    ) -> Iterable[np.ndarray]:
        """Construct a synthetic FID using the estimation result.

        Parameters
        ----------
        pts
            The number of points to construct the FID with in each dimesnion.
            If ``None``, the number of points used will match the estimated signal.

        oscillators
            Which oscillators in the result to include. If ``None``, all
            oscillators will be included. If a list of ints, the subset of
            oscillators corresponding to these indices will be used.

        Returns
        -------
        fid
            The generated FID.
        """
        sanity_check(
            ("pts", pts, sfuncs.check_points, (self.dim,), True),
            (
                "oscillators", oscillators, sfuncs.check_ints_less_than_n,
                (self.osc_number,), True,
            ),
        )

        if pts is None:
            pts = self.signal.shape
        if oscillators is None:
            oscillators = list(range(self.osc_number))
        params = self.result[oscillators]
        return sig.make_fid(params, self.expinfo, pts)[0]
