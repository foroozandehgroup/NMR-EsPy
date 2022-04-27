# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 26 Apr 2022 10:45:47 BST

from __future__ import annotations
import abc
import datetime
import functools
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from nmrespy import ExpInfo
from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._files import (
    check_saveable_path,
    check_existent_path,
    configure_path,
    open_file,
    save_file,
)
from nmrespy._result_fetcher import ResultFetcher
from nmrespy._sanity import sanity_check, funcs as sfuncs

if USE_COLORAMA:
    import colorama
    colorama.init()


def logger(f: callable):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        class_instance = args[0]
        if "_log" in kwargs:
            if not kwargs["_log"]:
                return f(*args, **kwargs)
            else:
                del kwargs["_log"]
        class_instance._log += f"--> `{f.__name__}` {args[1:]} {kwargs}\n"
        return f(*args, **kwargs)
    return inner


class Estimator(ExpInfo, metaclass=abc.ABCMeta):
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
            default_pts=self._data.shape,
            fn_mode=expinfo.fn_mode,
        )

        self._results = []
        now = datetime.datetime.now().strftime('%d-%m-%y %H:%M:%S')
        self._log = (
            "=====================\n"
            "Logfile for Estimator\n"
            "=====================\n"
            f"--> Created @ {now}\n"
        )

    @property
    def data(self) -> np.ndarray:
        """Return the data assocaited with the estimator."""
        return self._data

    @abc.abstractmethod
    def phase_data(*args, **kwargs):
        pass

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

    @classmethod
    @abc.abstractmethod
    def new_bruker(*args, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
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
            ("path", path, check_saveable_path, ("pkl", force_overwrite), {}, True),
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

    @abc.abstractmethod
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
        """
        if not self._results:
            return None

        sanity_check(
            (
                "indices", indices, sfuncs.check_int_list, (),
                {
                    "must_be_positive": True,
                    "max_value": len(self._results) - 1,
                },
                True,
            ),
        )
        if indices is None:
            return self._results
        else:
            return [self._results[i] for i in indices]

    @abc.abstractmethod
    def write_result(*args, **kwargs):
        pass

    @abc.abstractmethod
    def plot_result(*args, **kwargs):
        pass


class Result(ResultFetcher):

    def __init__(
        self,
        result: np.ndarray,
        errors: np.ndarray,
        region: Iterable[Tuple[float, float]],
        noise_region: Iterable[Tuple[float, float]],
        sfo: Iterable[float],
    ) -> None:
        self.result = result
        self.errors = errors
        self.region = region
        self.noise_region = noise_region
        super().__init__(sfo)

    def get_region(self, unit: str = "hz"):
        sanity_check(
            ("unit", unit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
        )
        return self.convert(self.region, f"hz->{unit}")

    def get_noise_region(self, unit: str = "hz"):
        sanity_check(
            ("unit", unit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
        )
        return self.convert(self.noise_region, f"hz->{unit}")
