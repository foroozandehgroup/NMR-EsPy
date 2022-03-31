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
from nmrespy._result_fetcher import ResultFetcher
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy.freqfilter import Region
from nmrespy.plot import NmrespyPlot

if USE_COLORAMA:
    import colorama
    colorama.init()


def copydoc(fromfunc, sep="\n"):
    """Decorator: copy the docstring of `fromfunc`."""
    def _decorator(func):
        sourcedoc = fromfunc.__doc__
        if func.__doc__ is None:
            func.__doc__ = sourcedoc
        else:
            func.__doc__ = sep.join([sourcedoc, func.__doc__])
        return func
    return _decorator


class Estimator(metaclass=ABCMeta):
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
        self._expinfo = expinfo
        self._expinfo.default_pts = self._data.shape
        self._results = []
        now = datetime.datetime.now()
        self._log = (
            "=====================\n"
            "Logfile for Estimator\n"
            "=====================\n"
            f"--> Created @ {now.strftime('%d-%m-%y %H:%M:%S')}\n"
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

    @property
    def dim(self):
        return self._expinfo.dim

    def converter(self, shape: Optional[Iterable[int]] = None):
        if shape is None:
            shape = self._data.shape
        return FrequencyConverter(self._expinfo, shape)

    @classmethod
    @abstractmethod
    def new_bruker(
        cls, directory: Union[str, Path], ask_convdta: bool = True
    ) -> Estimator:
        """Generate an estimator instance from a Bruker data directory.

        Parameters
        ----------
        directory : str
            The path to the data containing the data of interest.

        ask_convdta : bool
            See :py:meth:`nmrespy.load_bruker`

        Returns
        -------
        estimator : :py:class:`Estimator`

        Notes
        -----
        For a more detailed specification of the directory requirements,
        see :py:meth:`nmrespy.load_bruker`.
        """
        pass

    @classmethod
    @abstractmethod
    def new_synthetic_from_simulation(
        cls,
        spin_system,
        expinfo,
        pts,
        channel,
        snr,
    ) -> Estimator:
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
    @logger
    def estimate(
        self,
        region: Optional[Region] = None,
        noise_region: Optional[Region] = None,
        region_unit: str = "ppm",
        initial_guess: Optional[Union[np.ndarray, int]] = None,
        hessian: str = "gauss-newton",
        phase_variance: bool = False,
        max_iterations: Optional[int] = None,
    ):
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
    def write_results(
        self,
        indices: Optional[Iterable[int]] = None,
        path: Union[Path, str] = "./nmrespy_result",
        fmt: str = "txt",
        description: Optional[str] = None,
        sig_figs: Optional[int] = 5,
        sci_lims: Optional[Tuple[int, int]] = (-2, 3),
        force_overwrite: bool = False,
        fprint: bool = True,
    ) -> None:
        """Write estimation results to text and PDF files.

        Multiple results, corresponding to different selected regions, can be
        included in a single file, and can be controlled by the ``indices``
        argument.

        Parameters
        ----------
        indices
           Indices specifying the estimation results to include in the result
           file. ``None`` indicates to include all results associated with the
           instance.

        path
            The path to save the file to. Note that if the appropriate file format
            suffix is not provided to the path, it will be appended.

        fmt
            File format. Should be one of ``'txt'``, ``'pdf'``. See the *Notes*
            section of :py:func:`nmrespy.write.write_file` for details on system
            requirements for PDF generation.

        description
            A descriptive statement.

        sig_figs
            The number of significant figures to give to parameter values. If
            ``None``, the full value will be used.

        sci_lims
            Given a value ``(x, y)``, for ``x < 0`` and ``y > 0``, any parameter
            ``p`` value which satisfies ``p < 10 ** x`` or ``p >= 10 ** y`` will
            be expressed in scientific notation, rather than explicit notation.
            If ``None``, all values will be expressed explicitely.

        force_overwrite
            Defines behaviour if the path specified already exists:

            * If set to ``False``, the user will be asked if they are happy
              overwriting the current file.
            * If set to ``True``, the current file will be overwritten without
              prompt.

        fprint
            Specifies whether or not to print information to the terminal.
        """
        return

    @abstractmethod
    def plot_results(
        self,
        indices: Optional[Iterable[int]] = None,
        *,
        plot_residual: bool = True,
        plot_model: bool = False,
        residual_shift: Optional[Iterable[float]] = None,
        model_shift: Optional[float] = None,
        shifts_unit: str = "ppm",
        data_color: Any = "#000000",
        residual_color: Any = "#808080",
        model_color: Any = "#808080",
        oscillator_colors: Optional[Any] = None,
        show_labels: bool = False,
        stylesheet: Optional[Union[str, Path]] = None,
    ) -> Iterable[NmrespyPlot]:
        """Produce figures of estimation results.

        Each figure consists of the spectral data, along
        with each oscillator, for each specified result. Optionally, a plot of the
        complete model, and the residual between the data amd the model can be plotted.

        .. note::

            Currently, only 1D data is supported.

        Parameters
        ----------
        indices
           Indices specifying the estimation results to generate plots of
           file. ``None`` indicates to include all results associated with the
           instance.

        plot_residual
            If ``True``, plot a difference between the FT of ``spectrum`` and the
            FT of the model generated using ``result``. NB the residual is plotted
            regardless of ``plot_residual``. ``plot_residual`` specifies the alpha
            transparency of the plot line (1 for ``True``, 0 for ``False``)

        residual_shift
            Specifies a translation of the residual plot along the y-axis. If
            ``None``, a default shift will be applied.

        plot_model
            If ``True``, plot the FT of the model generated using ``result``.
            NB the residual is plotted regardless of ``plot_model``. ``plot_model``
            specifies the alpha transparency of the plot line (1 for ``True``,
            0 for ``False``).

        model_shift
            Specifies a translation of the residual plot along the y-axis. If
            ``None``, a default shift will be applied.

        shifts_unit
            Units to display chemical shifts in. Can be either ``'ppm'`` or
            ``'hz'``.

        data_color
            The colour used to plot the original spectrum. Any value that is
            recognised by matplotlib as a color is permitted. See
            `here <https://matplotlib.org/stable/tutorials/colors/colors.html>`_
            for a full description of valid values.

        residual_color
            The colour used to plot the residual. See ``data_color`` for valid colors.

        model_color
            The colour used to plot the model. See ``data_color`` for valid colors.

        oscillator_colors
            Describes how to color individual oscillators. The following
            is a complete list of options:

            * If a valid matplotlib color is given, all oscillators will
              be given this color.
            * If a string corresponding to a matplotlib colormap is given,
              the oscillators will be consecutively shaded by linear increments
              of this colormap. For all valid colormaps, see
              `here <https://matplotlib.org/stable/tutorials/colors/\
              colormaps.html>`__
            * If an iterable object containing valid matplotlib colors is
              given, these colors will be cycled.
              For example, if ``oscillator_colors = ['r', 'g', 'b']``:

              + Oscillators 1, 4, 7, ... would be :red:`red (#FF0000)`
              + Oscillators 2, 5, 8, ... would be :green:`green (#008000)`
              + Oscillators 3, 6, 9, ... would be :blue:`blue (#0000FF)`

            * If ``None``, the default colouring method will be applied, which
              involves cycling through the following colors:

                - :oscblue:`#1063E0`
                - :oscorange:`#EB9310`
                - :oscgreen:`#2BB539`
                - :oscred:`#D4200C`

        show_labels
            If ``True``, each oscillator will be given a numerical label
            in the plot, if ``False``, the labels will be hidden.

        stylesheet
            The name of/path to a matplotlib stylesheet for further
            customaisation of the plot. See `<here https://matplotlib.org/\
            stable/tutorials/introductory/customizing.html>`__ for more
            information on stylesheets.
        """
        return


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
