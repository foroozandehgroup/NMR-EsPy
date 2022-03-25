# core.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 24 Mar 2022 12:10:56 GMT

from __future__ import annotations
import copy
from dataclasses import dataclass
import datetime
import functools
from pathlib import Path
import pickle
import re
import tempfile
from typing import Any, Iterable, Optional, Tuple, Union
import uuid

import numpy as np
from nmr_sims.experiments.pa import PulseAcquireSimulation
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem

from nmrespy import ExpInfo, GRE, ORA, RED, END, USE_COLORAMA, sig
import nmrespy._errors as errors
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy._freqconverter import FrequencyConverter
from nmrespy.freqfilter import Region, filter_spectrum
from nmrespy.load import load_bruker
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import NonlinearProgramming
from nmrespy.plot import NmrespyPlot, plot_result
from nmrespy.write import write_result, _configure_save_path
from nmrespy.write.textfile import _write_txt
from nmrespy.write.pdffile import _write_pdf

if USE_COLORAMA:
    import colorama
    colorama.init()


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
        self, data: np.ndarray, datapath: Optional[Path], expinfo: ExpInfo,
    ) -> None:
        self._data = data
        self._datapath = datapath
        self._expinfo = expinfo
        self._dim = self._expinfo.unpack("dim")
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
        print(self._log)

    def save_log(
        self, path: Union[str, Path] = "./espy_logfile", force_overwrite: bool = True
    ) -> None:
        sanity_check(
            ("path", path, sfuncs.check_path),
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
        )
        path = _configure_save_path(path, "txt", force_overwrite)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self._log)

    @property
    def dim(self):
        return self._dim

    @property
    def converter(self):
        conv = getattr(self, "_converter", None)
        if conv is None:
            self._converter = FrequencyConverter(self._expinfo, self._data.shape)

        return self._converter

    @classmethod
    def new_bruker(
        cls, directory: Union[str, Path], ask_convdta: bool = True
    ) -> Estimator:
        """Generate an instance of :py:class:`Estimator` from a Bruker data directory.

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
        sanity_check(
            ("directory", directory, sfuncs.check_path),
            ("ask_convdta", ask_convdta, sfuncs.check_bool),
        )
        directory = Path(directory).resolve()
        data, expinfo = load_bruker(directory, ask_convdta=ask_convdta)

        if directory.parent.name == "pdata":
            shape = tuple([slice(0, x // 2) for x in data.shape])
            data = (2 * data.ndim * sig.ift(data))[shape]

        return cls(data, directory, expinfo)

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
        estimator: :py:class:`Estimator`
        """
        sanity_check(("expinfo", expinfo, sfuncs.check_expinfo),)

        dim = expinfo.unpack("dim")
        sanity_check(
            ("parameters", parameters, sfuncs.check_parameter_array, (dim,)),
            ("pts", pts, sfuncs.check_points, (dim,)),
            ("snr", snr, sfuncs.check_positive_float, (), True),
        )

        data = sig.make_fid(parameters, expinfo, pts, snr=snr)[0]
        return cls(data, None, expinfo)

    @classmethod
    def new_synthetic_from_simulation(
        cls,
        spin_system: SpinSystem,
        expinfo: ExpInfo,
        pts: int,
        channel: Union[str, Nucleus] = "1H",
        snr: Optional[float] = 30.0,
    ) -> Estimator:
        """Generate an estimator with data derived from a pulse-aquire experiment
        simulation.

        Simulations are performed using the
        `nmr_sims.experiments.pa.PulseAcquireSimulation
        <https://foroozandehgroup.github.io/nmr_sims/content/references/experiments/
        pa.html#nmr_sims.experiments.pa.PulseAcquireSimulation>`_
        class.

        Parameters
        ----------
        spin_system
            Specification of the spin system to run simulations on.
            `See here <https://foroozandehgroup.github.io/nmr_sims/content/
            references/spin_system.html#nmr_sims.spin_system.SpinSystem.__init__>`_
            for more details.

        expinfo
            Experiment Information. **This this correspond to a 1D experiment.**

        pts
            The number of points sampled.

        channel
            Nucleus targeted in the experiment simulation. ¹H is set as the default.
            `See here <https://foroozandehgroup.github.io/nmr_sims/content/
            references/nuclei.html>`__ for more information.

        snr
            The signal-to-noise ratio of the resulting signal, in decibels. ``None``
            produces a noiseless signal.
        """
        sanity_check(
            ("spin_system", spin_system, sfuncs.check_spin_system),
            ("expinfo", expinfo, sfuncs.check_expinfo, (1,)),
            ("pts", pts, sfuncs.check_points, (1,)),
            ("channel", channel, sfuncs.check_nucleus),
            ("snr", snr, sfuncs.check_float, (), True),
        )
        sim = PulseAcquireSimulation(
            spin_system, pts[0], expinfo.sw[0], offset=expinfo.offset[0],
            channel=channel,
        )
        sim.simulate()
        _, data = sim.fid
        if snr is not None:
            data += sig._make_noise(data, snr)

        expinfo._nucleus = (channel,)
        expinfo._sfo = (spin_system.field)

        return cls(data, None, expinfo)

    @logger
    def to_pickle(
        self, path: Optional[Union[Path, str]] = None, force_overwrite: bool = False,
        fprint: bool = True
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
            ("path", path, sfuncs.check_path, (), True),
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
            ("fprint", fprint, sfuncs.check_bool),
        )

        if path is None:
            x = 1
            while True:
                path = Path(f"estimator_{x}.pkl").resolve()
                if path.is_file():
                    x += 1
                else:
                    break

        path = Path(path).resolve()
        # Append extension to file path
        if path.suffix != ".pkl":
            path = path.with_suffix(".pkl")
        if path.is_file() and not force_overwrite:
            print(
                f"{ORA}to_pickle: `path` {path} already exists, and you have not "
                f"given permission to overwrite with `force_overwrite`. Skipping{END}."
            )
            return

        with open(path, "wb") as fh:
            pickle.dump(self, fh, pickle.HIGHEST_PROTOCOL)

        if fprint:
            print(f"{GRE}Saved estimator to {path}{END}")

    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> Estimator:
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
        path = Path(path).resolve()
        if path.suffix != ".pkl":
            path = path.with_suffix(".pkl")
        if not path.is_file():
            raise ValueError(f"{RED}Invalid path specified.{END}")

        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, __class__):
            return obj
        else:
            raise TypeError(
                f"{RED}It is expected that the object loaded by"
                " `from_pickle` is an instance of"
                f" {__class__.__module__}.{__class__.__qualname__}."
                f" What was loaded didn't satisfy this!{END}"
            )

    def _extract_region(
        self, region: Region, noise_region: Region, region_unit: str,
    ) -> Tuple[Region, Region]:
        if region_unit == "hz":
            region_check = sfuncs.check_region_hz
        elif region_unit == "ppm":
            region_check = sfuncs.check_region_ppm

        sanity_check(
            ("region", region, region_check, (self._expinfo,)),
            ("noise_region", noise_region, region_check, (self._expinfo,)),
            ("region_unit"
        )

        region = self.converter.convert(region, f"{region_unit}->hz")
        noise_region = self.converter.convert(noise_region, f"{region_unit}->hz")

        return region, noise_region

    def _make_spectrum(self) -> np.ndarray:
        return sig.ft(sig.make_virtual_echo(self._data))

    @logger
    def estimate(
        self,
        region: Optional[Region] = None,
        noise_region: Optional[Region] = None,
        *,
        region_unit: str = "ppm",
        initial_guess: Optional[Union[np.ndarray, int]] = None,
        hessian: str = "gauss-newton",
        phase_variance: bool = False,
        max_iterations: Optional[int] = None,
    ):
        """Estimate a specified region of the signal.

        The basic steps that this method carries out are:

        * (Optional, but highly advised) Generate a frequency-filtered signal
          corresponding to the specified region.
        * (Optional) Generate an inital guess using the Matrix Pencil Method (MPM).
        * Apply numerical optimisation to determine a final estimate of the signal
          parameters

        Parameters
        ----------
        region
            The frequency range of interest.

        noise_region
            A frequency range where no noticeable signals reside, i.e. only noise
            exists.

        region_unit
            One of ``"hz"``, ``"ppm"``, ``"idx"``, corresponding to Hertz, parts per
            million, and array indices, respecitvely. Specifies the units that
            ``region`` and ``noise_region`` have been given as.

        initial_guess
            If ``None``, an initial guess will be generated using the MPM,
            with the Minimum Descritpion Length being used to estimate the
            number of oscilltors present. If and int, the MPM will be used to
            compute the initial guess with the value given being the number of
            oscillators. If a NumPy array, this array will be used as the initial
            guess.

        hessian
            Specifies how to compute the Hessian.

            * ``"exact"`` - the exact analytical Hessian will be computed.
            * ``"gauss-newton"`` - the Hessian will be approximated as per the
              Guass-Newton method.

        phase_variance
            Whether or not to include the variance of oscillator phases in the cost
            function. This should be included in cases where the signal being
            considered is derived from phased data.

        max_iterations
            The greatest number of iterations to allow the optimiser to run before
            terminating. If ``None``, this number will be set to a default, depending
            on the identity of ``hessian``.
        """
        sanity_check(
            (
                "initial_guess", initial_guess, sfuncs.check_initial_guess,
                (self.dim,), True
            ),
            ("hessian", hessian, sfuncs.check_one_of, ("gauss-newton", "exact")),
            ("phase_variance", phase_variance, sfuncs.check_bool),
            ("max_iterations", max_iterations, sfuncs.check_positive_int, (), True),
        )

        region, noise_region = self._extract_region(region, noise_region, region_unit)
        spectrum = self._make_spectrum()

        timestamp = datetime.datetime.now()
        filter_info = filter_spectrum(
            spectrum,
            self._expinfo,
            region,
            noise_region,
            region_unit="hz",
        )
        signal, expinfo = filter_info.get_filtered_fid()
        if isinstance(initial_guess, np.ndarray):
            x0 = initial_guess
        else:
            M = initial_guess if isinstance(initial_guess, int) else 0
            x0 = MatrixPencil(signal, expinfo, M=M).get_result()

        nlp_result = NonlinearProgramming(
            signal, x0, expinfo, phase_variance=phase_variance, hessian=hessian,
            max_iterations=max_iterations,
        )
        result, errors = nlp_result.get_result(), nlp_result.get_errors()
        self._results.append(
            Result(timestamp, signal, expinfo, region, result, errors)
        )

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
        sanity_check(
            (
                "indices", indices, sfuncs.check_ints_less_than_n,
                (len(self._results),), True,
            ),
            ("path", path, sfuncs.check_str),
            ("fmt", fmt, sfuncs.check_one_of, ("txt", "pdf")),
            ("description", description, sfuncs.check_str, (), True),
            ("sig_figs", sig_figs, sfuncs.check_positive_int, (), True),
            ("sci_lims", sci_lims, sfuncs.check_sci_lims, (), True),
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
            ("fprint", fprint, sfuncs.check_bool),
        )
        path = _configure_save_path(path, fmt, force_overwrite)

        results = self.get_results(indices)

        # Write each file to a random filename
        tmpdir = tempfile.TemporaryDirectory()
        files = [Path(tmpdir.name) / str(uuid.uuid4().hex)
                 for _ in results]
        contents = []
        for result, fname in zip(results, files):
            write_result(
                self._expinfo, result.get_result(), result.get_errors(),
                path=str(fname), fmt=fmt, description=description, sig_figs=sig_figs,
                sci_lims=sci_lims, force_overwrite=force_overwrite, fprint=False,
            )
            if fmt == "pdf":
                fname = fname.with_suffix(".tex")
            else:
                fname = fname.with_suffix(f".{fmt}")

            with open(fname, "r") as fh:
                contents.append((fh.read(), result.get_region(unit="ppm")))

        if fmt == "txt":
            txt = self._process_txt(contents)
            _write_txt(path, txt, fprint)
        elif fmt == "pdf":
            txt = self._process_pdf(contents)
            _write_pdf(path, txt, None, fprint)

    @staticmethod
    def _process_txt(
        contents: Iterable[Tuple[str, Iterable[Tuple[float, float]]]]
    ) -> str:
        n = len(contents)
        fulltxt = ""
        for i, (txt, region) in enumerate(contents):
            table_start = re.search("Osc. │", txt).span()[0]
            table_end = re.search("\n\n\nEstimation performed", txt).span()[0]

            # Add title
            if i == 0:
                fulltxt += txt[:table_start]

            region_list = ", ".join(
                [f"{r[0]} - {r[1]} ppm (F{i})" for i, r in enumerate(region, start=1)]
            )
            if len(region) == 1:
                region_list = region_list[:-5]

            fulltxt += f"{region_list}:\n"
            fulltxt += f"{txt[table_start : table_end]}\n\n"

            # Add blurb
            if i == n - 1:
                fulltxt += txt[table_end + 2:]

        return fulltxt

    @staticmethod
    def _process_pdf(
        contents: Iterable[Tuple[str, Iterable[Tuple[float, float]]]]
    ) -> str:
        n = len(contents)
        fulltxt = ""
        for i, (txt, region) in enumerate(contents):
            table_start = re.search(
                r"\\begin\{longtable\}\[l\]\{c c c c", txt
            ).span()[0]
            table_end = re.search("\n\n\n\n% blurb", txt).span()[0]

            # Add title
            if i == 0:
                fulltxt += txt[:table_start - 21]

            region_list = ", ".join(
                [f"{r[0]} -- {r[1]} ppm (F{i})" for i, r in enumerate(region, start=1)]
            )
            if len(region) == 1:
                region_list = region_list.replace(" (F1)", "")

            fulltxt += f"\\subsection*{{{region_list}}}\n"
            fulltxt += f"{txt[table_start : table_end]}\n\n"

            # Add blurb
            if i == n - 1:
                fulltxt += txt[table_end + 4:]

        return fulltxt

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

        Returns
        -------
        plot: :py:class:`NmrespyPlot`
            The result plot.
        """
        sanity_check(
            ("indices", indices, sfuncs.check_ints_less_than_n, (), True),
            ("plot_residual", plot_residual, sfuncs.check_bool),
            ("plot_model", plot_model, sfuncs.check_bool),
            ("residual_shift", residual_shift, sfuncs.check_float, (), True),
            ("model_shift", model_shift, sfuncs.check_float, (), True),
            ("shifts_unit", shifts_unit, sfuncs.check_one_of, ("hz", "ppm")),
            ("data_color", data_color, sfuncs.check_mpl_color),
            ("residual_color", residual_color, sfuncs.check_mpl_color),
            ("model_color", model_color, sfuncs.check_mpl_color),
            (
                "oscillator_colors", oscillator_colors, sfuncs.check_oscillator_colors,
                (), True,
            ),
            ("show_labels", show_labels, sfuncs.check_bool),
            ("stylesheet", stylesheet, sfuncs.check_str, (), True),
        )
        results = self.get_results(indices)

        if self.dim == 2:
            raise errors.TwoDimUnsupportedError()

        return [
            plot_result(
                sig.ft(self._data),
                result.get_result(),
                self._expinfo,
                region=result.get_region(unit="ppm"),
            )
            for result in results
        ]


@dataclass
class Result:
    timestamp: datetime.datetime
    signal: np.ndarray
    expinfo: ExpInfo
    region: Region
    result: np.ndarray
    errors: np.ndarray

    @property
    def dim(self) -> int:
        return self.expinfo.unpack("dim")

    @property
    def osc_number(self) -> int:
        """Return the number of oscillators in the result."""
        return self.result.shape[0]

    @property
    def converter(self):
        conv = getattr(self, "_converter", None)
        if conv is None:
            self._converter = FrequencyConverter(self.expinfo, self.signal.shape)

        return self._converter

    def get_region(self, unit: str = "hz") -> Iterable[Tuple[float, float]]:
        sanity_check(
            ("unit", unit, sfuncs.check_one_of, ("hz", "ppm"),),
        )
        return self.converter.convert(self.region, f"hz->{unit}")

    def get_result(self, funit: str = "hz"):
        """Returns the estimation result

        Parameters
        ----------
        funit
            The unit to express the frequencies in. Should be ``"hz"`` or ``"ppm"``.
        """
        return self._get_array("result", funit)

    def get_errors(self, funit: str = "hz"):
        """Returns the errors of the estimation result.

        Parameters
        ----------
        funit
            The unit to express the frequencies in. Should be ``"hz"`` or ``"ppm"``.
        """
        return self._get_array("errors", funit)

    def _get_array(self, name, funit):
        array = copy.deepcopy(getattr(self, name))
        if funit == "hz":
            return array

        elif funit == "ppm":
            if self.expinfo.unpack("sfo") is None:
                raise TypeError(
                    f"{RED}Error in trying to convert frequencies to ppm. sfo hasn't"
                    f" been specified!{END}"
                )

            f_idx = [2 + i for i in range(self.signal.ndim)]
            for i in range(2, 2 + self.expinfo.unpack("dim")):
                array[:, f_idx] = np.array(
                    self.converter.convert(
                        [a for a in array[:, f_idx].T], conversion="hz->ppm",
                    )
                ).T
                return array

        else:
            raise errors.InvalidUnitError("hz", "ppm")

    def get_timepoints(
        self,
        pts: Optional[Iterable[int]] = None,
        meshgrid_2d: bool = True,
    ) -> Iterable[np.ndarray]:
        """Construct time-points which reflect the experiment parameters.

        Parameters
        ----------
        pts
            The number of points to construct the time-points with in each dimesnion.
            If ``None``, the number of points used will match the estimated signal.

        meshgrid_2d
            If time-points are being derived for a two-dimensional signal, setting
            this argument to ``True`` will return two two-dimensional arrays
            corresponding to all pairs of x and y values to construct a 3D
            plot/contour plot.

        Returns
        -------
        shifts
            The sampled chemical shifts.
        """
        sanity_check(
            ("pts", pts, sfuncs.check_points, (self.dim,), True),
            ("meshgrid_2d", meshgrid_2d, sfuncs.check_bool),
        )
        return sig.get_timepoints(self.expinfo, pts, meshgrid_2d=meshgrid_2d)

    def get_shifts(
        self,
        pts: Optional[Iterable[int]] = None,
        unit: str = "hz",
        meshgrid_2d: bool = True,
    ) -> Iterable[np.ndarray]:
        """Construct chemical shifts which reflect the experiment parameters.

        Parameters
        ----------
        pts
            The number of points to construct the shifts with in each dimesnion.
            If ``None``, the number of points used will match the estimated signal.

        unit
            The unit of the chemical shifts. One of ``"hz"``, ``"ppm"``.

        meshgrid_2d
            If shifts are being derived for a two-dimensional signal, setting
            this argument to ``True`` will return two two-dimensional arrays
            corresponding to all pairs of x and y values to construct a 3D
            plot/contour plot.

        Returns
        -------
        shifts
            The sampled chemical shifts.
        """
        sanity_check(
            ("pts", pts, sfuncs.check_points, (self.dim,), True),
            ("unit", unit, sfuncs.check_frequency_unit),
            ("meshgrid_2d", meshgrid_2d, sfuncs.check_bool),
        )
        if pts is None:
            pts = self.signal.shape
        return sig.get_shifts(self.expinfo, pts, unit=unit, meshgrid_2d=meshgrid_2d)

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
