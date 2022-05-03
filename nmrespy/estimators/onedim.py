# onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 03 May 2022 13:40:34 BST

from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from nmr_sims.experiments.pa import PulseAcquireSimulation
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem

from nmrespy import ExpInfo, sig
from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._files import (
    check_existent_dir,
    check_saveable_path,
)

from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)
from nmrespy.freqfilter import Filter
from nmrespy.load import load_bruker
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import NonlinearProgramming
from nmrespy.plot import ResultPlotter
from nmrespy.write import ResultWriter

from . import logger, Estimator, Result


if USE_COLORAMA:
    import colorama
    colorama.init()


class Estimator1D(Estimator):
    """Estimator class for 1D data.

    .. note::

        To create an instance of ``Estimator1D``, you should use one of the following
        methods:

        * :py:meth:`new_bruker`
        * :py:meth:`new_synthetic_from_parameters`
        * :py:meth:`new_synthetic_from_simulation`
        * :py:meth:`from_pickle`
    """

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ExpInfo,
        datapath: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        data
            Time-domain data to estimate.

        expinfo
            Experiment information.

        datapath
            If applicable, the path that the data was derived from.
        """
        super().__init__(data, expinfo, datapath)

    @classmethod
    def new_bruker(
        cls,
        directory: Union[str, Path],
        ask_convdta: bool = True,
    ) -> Estimator1D:
        """Create a new instance from Bruker-formatted data.

        Parameters
        ----------
        directory
            Absolute path to data directory.

        ask_convdta
            If ``True``, the user will be warned that the data should have its
            digitial filter removed prior to importing if the data to be impoprted
            is from an ``fid`` or ``ser`` file. If ``False``, the user is not
            warned.

        Notes
        -----
        **Directory Requirements**

        There are certain file paths expected to be found relative to ``directory``
        which contain the data and parameter files. Here is an extensive list of
        the paths expected to exist, for different data types:

        * Raw FID

          + ``directory/fid``
          + ``directory/acqus``

        * Processed data

          + ``directory/1r``
          + ``directory/../../acqus``
          + ``directory/procs``

        **Digital Filters**

        If you are importing raw FID data, make sure the path specified
        corresponds to an ``fid`` file which has had its group delay artefact
        removed. To do this, open the data you wish to analyse in TopSpin, and
        enter ``convdta`` in the bottom-left command line. You will be prompted
        to enter a value for the new data directory. It is this value you
        should use in ``directory``, not the one corresponding to the original
        (uncorrected) signal.
        """
        sanity_check(
            ("directory", directory, check_existent_dir),
            ("ask_convdta", ask_convdta, sfuncs.check_bool),
        )

        directory = Path(directory).expanduser()
        data, expinfo = load_bruker(directory, ask_convdta=ask_convdta)

        if data.ndim != 1:
            raise ValueError(f"{RED}Data dimension should be 1.{END}")

        if directory.parent.name == "pdata":
            slice_ = slice(0, data.shape[0] // 2)
            data = (2 * sig.ift(data))[slice_]

        return cls(data, expinfo, directory)

    @classmethod
    def new_synthetic_from_parameters(
        cls,
        params: np.ndarray,
        pts: int,
        sw: float,
        offset: float = 0.0,
        sfo: Optional[float] = None,
        snr: float = 30.0,
    ) -> Estimator1D:
        """Generate an estimator instance from an array of oscillator parameters.

        Parameters
        ----------
        params
            Parameter array with the following structure:

              .. code:: python

                 params = numpy.array([
                    [a_1, φ_1, f_1, η_1],
                    [a_2, φ_2, f_2, η_2],
                    ...,
                    [a_m, φ_m, f_m, η_m],
                 ])

        pts
            The number of points the signal comprises.

        sw
            The sweep width of the signal (Hz).

        offset
            The transmitter offset (Hz).

        sfo
            The transmitter frequency (MHz).

        snr
            The signal-to-noise ratio. If ``None`` then no noise will be added
            to the FID.
        """
        sanity_check(
            ("params", params, sfuncs.check_ndarray, (), {"dim": 2, "shape": [(1, 4)]}),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}),
            ("sw", sw, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("offset", offset, sfuncs.check_float, (), {}, True),
            ("sfo", sfo, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            ("snr", snr, sfuncs.check_float, (), {"greater_than_zero": True}, True),
        )

        expinfo = ExpInfo(
            dim=1,
            sw=sw,
            offset=offset,
            sfo=sfo,
            default_pts=pts,
        )

        data = expinfo.make_fid(params, snr=snr)
        return cls(data, expinfo)

    @classmethod
    def new_synthetic_from_simulation(
        cls,
        spin_system: SpinSystem,
        sw: float,
        offset: float,
        pts: int,
        freq_unit: str = "hz",
        channel: Union[str, Nucleus] = "1H",
        snr: Optional[float] = 30.0,
    ) -> Estimator1D:
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
            for more details. **N.B. the transmitter frequency (sfo) will
            be determined by** ``spin_system.field``.

        sw
            The sweep width in Hz.

        offset
            The transmitter offset frequency in Hz.

        pts
            The number of points sampled.

        freq_unit
            The unit that ``sw`` and ``offset`` are expressed in. Should
            be either ``"hz"`` or ``"ppm"``.

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
            ("sw", sw, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("offset", offset, sfuncs.check_float),
            ("pts", pts, sfuncs.check_positive_int),
            ("freq_unit", freq_unit, sfuncs.check_one_of, ("hz", "ppm")),
            ("channel", channel, sfuncs.check_nmrsims_nucleus),
            ("snr", snr, sfuncs.check_float, (), {}, True),
        )

        sw = f"{sw}{freq_unit}"
        offset = f"{offset}{freq_unit}"
        sim = PulseAcquireSimulation(
            spin_system, pts, sw, offset=offset, channel=channel,
        )
        sim.simulate()
        _, data = sim.fid
        if snr is not None:
            data += sig._make_noise(data, snr)

        expinfo = ExpInfo(
            dim=1,
            sw=sim.sweep_widths[0],
            offset=sim.offsets[0],
            sfo=sim.sfo[0],
            nuclei=channel,
            default_pts=data.shape,
        )

        return cls(data, expinfo)

    def phase_data(
        self,
        p0: float = 0.0,
        p1: float = 0.0,
        pivot: int = 0,
    ) -> None:
        """Apply first-order phae correction to the estimator's data.

        Parameters
        ----------
        p0
            Zero-order phase correction, in radians.

        p1
            First-order phase correction, in radians.

        pivot
            Index of the pivot.
        """
        sanity_check(
            ("p0", p0, sfuncs.check_float),
            ("p1", p1, sfuncs.check_float),
            ("pivot", pivot, sfuncs.check_index, (self._data.size,)),
        )
        self._data = sig.phase(self._data, [p0], [p1], [pivot])

    def view_data(
        self,
        domain: str = "freq",
        components: str = "real",
        freq_unit: str = "hz",
    ) -> None:
        """View the data.

        Parameters
        ----------
        domain
            Must be ``"freq"`` or ``"time"``.

        components
            Must be ``"real"``, ``"imag"``, or ``"both"``.

        freq_unit
            Must be ``"hz"`` or ``"ppm"``.
        """
        sanity_check(
            ("domain", domain, sfuncs.check_one_of, ("freq", "time")),
            ("components", components, sfuncs.check_one_of, ("real", "imag", "both")),
            ("freq_unit", freq_unit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
        )

        fig = plt.figure()
        ax = fig.add_subplot()
        y = copy.deepcopy(self._data)

        if domain == "freq":
            x = self.get_shifts(unit=freq_unit)[0]
            y[0] /= 2
            y = sig.ft(y)
            label = f"$\\omega$ ({freq_unit.replace('h', 'H')})"
        elif domain == "time":
            x = self.get_timepoints()[0]
            label = "$t$ (s)"

        if components in ["real", "both"]:
            ax.plot(x, y.real, color="k")
        if components in ["imag", "both"]:
            ax.plot(x, y.imag, color="#808080")

        ax.set_xlabel(label)
        ax.set_xlim((x[0], x[-1]))

        plt.show()

    @logger
    def estimate(
        self,
        region: Optional[Tuple[float, float]] = None,
        noise_region: Optional[Tuple[float, float]] = None,
        region_unit: str = "hz",
        initial_guess: Optional[Union[np.ndarray, int]] = None,
        method: str = "gauss-newton",
        phase_variance: bool = True,
        max_iterations: Optional[int] = None,
        cut_ratio: Optional[float] = 1.1,
        mpm_trim: Optional[int] = 4096,
        nlp_trim: Optional[int] = None,
        fprint: bool = True,
        _log: bool = True,
    ) -> None:
        r"""Estimate a specified region of the signal.

        The basic steps that this method carries out are:

        * (Optional, but highly advised) Generate a frequency-filtered signal
          corresponding to the specified region.
        * (Optional) Generate an inital guess using the Matrix Pencil Method (MPM).
        * Apply numerical optimisation to determine a final estimate of the signal
          parameters

        Parameters
        ----------
        region
            The frequency range of interest. Should be of the form ``[left, right]``
            where ``left`` and ``right`` are the left and right bounds of the region
            of interest. If ``None``, the full signal will be considered, though
            for sufficently large and complex signals it is probable that poor and
            slow performance will be achieved.

        noise_region
            If ``region`` is not ``None``, this must be of the form ``[left, right]``
            too. This should specify a frequency range where no noticeable signals
            reside, i.e. only noise exists.

        region_unit
            One of ``"hz"`` or ``"ppm"`` Specifies the units that ``region``
            and ``noise_region`` have been given as.

        initial_guess
            If ``None``, an initial guess will be generated using the MPM,
            with the Minimum Descritpion Length being used to estimate the
            number of oscilltors present. If and int, the MPM will be used to
            compute the initial guess with the value given being the number of
            oscillators. If a NumPy array, this array will be used as the initial
            guess.

        method
            Specifies the optimisation method.

            * ``"exact"`` Uses SciPy's
              `trust-constr routine <https://docs.scipy.org/doc/scipy/reference/
              optimize.minimize-trustconstr.html\#optimize-minimize-trustconstr>`_
              The Hessian will be exact.
            * ``"gauss-newton"`` Uses SciPy's
              `trust-constr routine <https://docs.scipy.org/doc/scipy/reference/
              optimize.minimize-trustconstr.html\#optimize-minimize-trustconstr>`_
              The Hessian will be approximated based on the
              `Gauss-Newton method <https://en.wikipedia.org/wiki/
              Gauss%E2%80%93Newton_algorithm>`_
            * ``"lbfgs"`` Uses SciPy's
              `L-BFGS-B routine <https://docs.scipy.org/doc/scipy/reference/
              optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb>`_.

        phase_variance
            Whether or not to include the variance of oscillator phases in the cost
            function. This should be set to ``True`` in cases where the signal being
            considered is derived from well-phased data.

        max_iterations
            A value specifiying the number of iterations the routine may run
            through before it is terminated. If ``None``, the default number
            of maximum iterations is set (``100`` if ``method`` is
            ``"exact"`` or ``"gauss-newton"``, and ``500`` if ``"method"`` is
            ``"lbfgs"``).

        mpm_trim
            Specifies the maximal size allowed for the filtered signal when
            undergoing the Matrix Pencil. If ``None``, no trimming is applied
            to the signal. If an int, and the filtered signal has a size
            greater than ``mpm_trim``, this signal will be set as
            ``signal[:mpm_trim]``.

        nlp_trim
            Specifies the maximal size allowed for the filtered signal when undergoing
            nonlinear programming. By default (``None``), no trimming is applied to
            the signal. If an int, and the filtered signal has a size greater than
            ``nlp_trim``, this signal will be set as ``signal[:nlp_trim]``.

        fprint
            Whether of not to output information to the terminal.

        _log
            Ignore this!
        """
        sanity_check(
            (
                "region_unit", region_unit, sfuncs.check_frequency_unit,
                (self.sfo is not None,),
            ),
            (
                "initial_guess", initial_guess, sfuncs.check_initial_guess,
                (self.dim,), {}, True
            ),
            ("method", method, sfuncs.check_one_of, ("lbfgs", "gauss-newton", "exact")),
            ("phase_variance", phase_variance, sfuncs.check_bool),
            (
                "max_iterations", max_iterations, sfuncs.check_int, (),
                {"min_value": 1}, True,
            ),
            ("fprint", fprint, sfuncs.check_bool),
            ("mpm_trim", mpm_trim, sfuncs.check_int, (), {"min_value": 1}, True),
            ("nlp_trim", nlp_trim, sfuncs.check_int, (), {"min_value": 1}, True),
            (
                "cut_ratio", cut_ratio, sfuncs.check_float, (),
                {"greater_than_one": True}, True,
            ),
        )

        sanity_check(
            (
                "region", region, sfuncs.check_region,
                (self.sw(region_unit), self.offset(region_unit)), {}, True,
            ),
            (
                "noise_region", noise_region, sfuncs.check_region,
                (self.sw(region_unit), self.offset(region_unit)), {}, True,
            ),
        )

        # The plan of action:
        # --> Derive filtered signals (both cut and uncut)
        # --> Run the MDL followed by MPM for an initial guess on cut signal
        # --> Run Optimiser on cut signal
        # --> Run Optimiser on uncut signal

        filt = Filter(
            self._data,
            ExpInfo(1, self.sw(), self.offset(), self.sfo),
            region,
            noise_region,
            region_unit=region_unit,
        )

        cut_signal, cut_expinfo = filt.get_filtered_fid()
        uncut_signal, uncut_expinfo = filt.get_filtered_fid(cut_ratio=None)
        region = filt.get_region()

        cut_size = cut_signal.size
        uncut_size = uncut_signal.size
        if (mpm_trim is None) or (mpm_trim > cut_size):
            mpm_trim = cut_size
        if (nlp_trim is None) or (nlp_trim > uncut_size):
            nlp_trim = uncut_size

        if isinstance(initial_guess, np.ndarray):
            x0 = initial_guess
        else:
            oscillators = initial_guess if isinstance(initial_guess, int) else 0
            x0 = MatrixPencil(
                cut_expinfo,
                cut_signal[:mpm_trim],
                oscillators=oscillators,
                fprint=fprint,
            ).get_result()

        cut_result = NonlinearProgramming(
            cut_expinfo,
            cut_signal[:mpm_trim],
            x0,
            phase_variance=phase_variance,
            method=method,
            max_iterations=max_iterations,
            fprint=fprint,
        ).get_result()

        final_result = NonlinearProgramming(
            uncut_expinfo,
            uncut_signal[:nlp_trim],
            cut_result,
            phase_variance=phase_variance,
            method=method,
            max_iterations=max_iterations,
            fprint=fprint,
        )

        self._results.append(
            Result(
                final_result.get_result(),
                final_result.get_errors(),
                filt.get_region(),
                filt.get_noise_region(),
                self.sfo,
            )
        )

    @logger
    def write_result(
        self,
        indices: Optional[Iterable[int]] = None,
        path: Union[Path, str] = "./nmrespy_result",
        fmt: str = "txt",
        description: Optional[str] = None,
        sig_figs: Optional[int] = 5,
        sci_lims: Optional[Tuple[int, int]] = (-2, 3),
        force_overwrite: bool = False,
        fprint: bool = True,
        pdflatex_exe: Optional[Union[str, Path]] = None,
    ) -> None:
        """Write estimation results to text and PDF files.

        Parameters
        ----------
        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        path
            Path to save the result file to.

        fmt
            Must be one of ``"txt"`` or ``"pdf"``.

        description
            A description to add to the result file.

        sig_figs
            The number of significant figures to give to parameters. If
            ``None``, the full value will be used.

        sci_lims
            Given a value ``(-x, y)``, for ints ``x`` and ``y``, any parameter ``p``
            with a value which satisfies ``p < 10 ** -x`` or ``p >= 10 ** y`` will be
            expressed in scientific notation, rather than explicit notation.
            If ``None``, all values will be expressed explicitely.

        force_overwrite
            If the file specified already exists, and this is set to ``False``, the
            user will be prompted to specify that they are happy overwriting the
            current file.

        fprint
            Specifies whether or not to print information to the terminal.

        pdflatex_exe
            The path to the system's ``pdflatex`` executable.

            .. note::

               You are unlikely to need to set this manually. It is primarily
               present to specify the path to ``pdflatex.exe`` on Windows when
               the NMR-EsPy GUI has been loaded from TopSpin.
        """
        sanity_check(
            (
                "indices", indices, sfuncs.check_int_list, (),
                {
                    "must_be_positive": True,
                    "max_value": len(self._results) - 1,
                },
                True,
            ),
            ("fmt", fmt, sfuncs.check_one_of, ("txt", "pdf")),
            ("description", description, sfuncs.check_str, (), {}, True),
            ("sig_figs", sig_figs, sfuncs.check_int, (), {"min_value": 1}, True),
            ("sci_lims", sci_lims, sfuncs.check_sci_lims, (), {}, True),
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
            ("fprint", fprint, sfuncs.check_bool),
        )
        sanity_check(("path", path, check_saveable_path, (fmt, force_overwrite)))

        expinfo = ExpInfo(
            1,
            self.sw(),
            self.offset(),
            self.sfo,
            self.nuclei,
            self.default_pts,
        )

        indices = range(len(self._results)) if indices is None else indices
        results = [self._results[i] for i in indices]
        writer = ResultWriter(
            expinfo,
            [result.get_result() for result in results],
            [result.get_errors() for result in results],
            description,
        )
        region_unit = "ppm" if self.hz_ppm_valid else "hz"
        titles = [
            f"{left:.3f} - {right:.3f} {region_unit}".replace("h", "H")
            for left, right in [
                result.get_region(region_unit)[0]
                for result in results
            ]
        ]

        writer.write(
            path=path,
            fmt=fmt,
            titles=titles,
            parameters_sig_figs=sig_figs,
            parameters_sci_lims=sci_lims,
            force_overwrite=True,
            fprint=fprint,
            pdflatex_exe=pdflatex_exe,
        )

    @logger
    def plot_result(
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
    ) -> Iterable[ResultPlotter]:
        """Write estimation results to text and PDF files.

        Parameters
        ----------
        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        plot_model
            If ``True``, plot the model generated using ``result``. This model is
            a summation of all oscillator present in the result.

        plot_residual
            If ``True``, plot the difference between the data and the model
            generated using ``result``.

        residual_shift
            Specifies a translation of the residual plot along the y-axis. If
            ``None``, a default shift will be applied.

        model_shift
            Specifies a translation of the residual plot along the y-axis. If
            ``None``, a default shift will be applied.

        shifts_unit
            Units to display chemical shifts in. Must be either ``'ppm'`` or
            ``'hz'``.

        data_color
            The colour used to plot the data. Any value that is recognised by
            matplotlib as a color is permitted. See `here
            <https://matplotlib.org/stable/tutorials/colors/colors.html>`_ for
            a full description of valid values.

        residual_color
            The colour used to plot the residual. See ``data_color`` for a
            description of valid colors.

        model_color
            The colour used to plot the model. See ``data_color`` for a
            description of valid colors.

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
            customaisation of the plot. See `here <https://matplotlib.org/\
            stable/tutorials/introductory/customizing.html>`__ for more
            information on stylesheets.
        """
        sanity_check(
            (
                "indices", indices, sfuncs.check_int_list, (),
                {
                    "must_be_positive": True,
                    "max_value": len(self._results) - 1,
                },
                True,
            ),
            ("plot_residual", plot_residual, sfuncs.check_bool),
            ("plot_model", plot_model, sfuncs.check_bool),
            ("residual_shift", residual_shift, sfuncs.check_float, (), {}, True),
            ("model_shift", model_shift, sfuncs.check_float, (), {}, True),
            (
                "shifts_unit", shifts_unit, sfuncs.check_frequency_unit,
                (self.hz_ppm_valid,),
            ),
            ("data_color", data_color, sfuncs.check_mpl_color),
            ("residual_color", residual_color, sfuncs.check_mpl_color),
            ("model_color", model_color, sfuncs.check_mpl_color),
            (
                "oscillator_colors", oscillator_colors, sfuncs.check_oscillator_colors,
                (), {}, True,
            ),
            ("show_labels", show_labels, sfuncs.check_bool),
            ("stylesheet", stylesheet, sfuncs.check_str, (), {}, True),
        )
        results = self.get_results(indices)

        expinfo = ExpInfo(
            1,
            sw=self.sw(),
            offset=self.offset(),
            sfo=self.sfo,
            nuclei=self.nuclei,
            default_pts=self.default_pts,
        )

        return [
            ResultPlotter(
                self._data,
                result.get_result(funit="hz"),
                expinfo,
                region=result.get_region(unit=shifts_unit),
                shifts_unit=shifts_unit,
                plot_residual=plot_residual,
                plot_model=plot_model,
                residual_shift=residual_shift,
                model_shift=model_shift,
                data_color=data_color,
                residual_color=residual_color,
                model_color=model_color,
                oscillator_colors=oscillator_colors,
                show_labels=show_labels,
                stylesheet=stylesheet,
            )
            for result in results
        ]

    def _positive_index(self, index: int) -> int:
        return index % len(self._results)

    @logger
    def merge_oscillators(
        self,
        oscillators: Iterable[int],
        index: int = -1,
        **estimate_kwargs,
    ) -> None:
        """Merge oscillators in an estimation result.

        Removes the osccilators specified, and constructs a single new
        oscillator with a cumulative amplitude, and averaged phase,
        frequency and damping. Then runs optimisation on the updated set of
        oscillators.

        Parameters
        ----------
        oscillators
            A list of indices corresponding to the oscillators to be merged.

        index
            The index of the result to edit. Index ``0`` corresponds to the
            first result obtained using the estimator, ``1`` corresponds to the
            next, etc. By default, the most recently obtained result will be
            edited.

        estimate_kwargs
            Keyword arguments to provide to the call to :py:meth:`estimate`. Note
            that ``"initial_guess"`` and ``"region_unit"`` are set internally and
            will be ignored if given.

        Notes
        -----
        Assuming that an estimation result contains a subset of oscillators
        denoted by indices :math:`\\{m_1, m_2, \\cdots, m_J\\}`, where :math:`J
        \\leq M`, the new oscillator formed by the merging of the oscillator
        subset will possess the following parameters prior to re-running estimation:

            * :math:`a_{\\mathrm{new}} = \\sum_{i=1}^J a_{m_i}`
            * :math:`\\phi_{\\mathrm{new}} = \\frac{1}{J} \\sum_{i=1}^J
              \\phi_{m_i}`
            * :math:`f_{\\mathrm{new}} = \\frac{1}{J} \\sum_{i=1}^J f_{m_i}`
            * :math:`\\eta_{\\mathrm{new}} = \\frac{1}{J} \\sum_{i=1}^J
              \\eta_{m_i}`
        """
        sanity_check(
            ("index", index, sfuncs.check_index, (len(self._results),)),
        )
        index = self._positive_index(index)
        result = self._results[index]
        x0 = result.get_result()
        sanity_check(
            (
                "oscillators", oscillators, sfuncs.check_int_list,
                (), {"min_value": 0, "max_value": x0.shape[0] - 1},
            )
        )

        to_merge = x0[oscillators]
        # Sum amps, phases, freqs and damping over the oscillators
        # to be merged.
        # keepdims ensures that the final array is [[a, φ, f, η]]
        # rather than [a, φ, f, η]
        new_osc = np.sum(to_merge, axis=0, keepdims=True)

        # Get mean for phase, frequency and damping
        new_osc[:, 1:] = new_osc[:, 1:] / float(len(oscillators))
        # wrap phase
        new_osc[:, 1] = (new_osc[:, 1] + np.pi) % (2 * np.pi) - np.pi

        x0 = np.delete(x0, oscillators, axis=0)
        x0 = np.vstack((x0, new_osc))

        self._optimise_after_edit(x0, result, index)

    @logger
    def split_oscillator(
        self,
        oscillator: int,
        index: int = -1,
        separation_frequency: Optional[float] = None,
        unit: str = "hz",
        split_number: int = 2,
        amp_ratio: Optional[Iterable[float]] = None,
        **estimate_kwargs,
    ) -> None:
        """Splits an oscillator in an estimation result into multiple oscillators.

        Removes an oscillator, and incorporates two or more oscillators whose
        cumulative amplitudes match that of the removed oscillator. Then runs
        optimisation on the updated set of oscillators.

        Parameters
        ----------
        oscillator
            The index of the oscillator to be split.

        index
            The index of the result to edit. Index ``0`` corresponds to the
            first result obtained using the estimator, ``1`` corresponds to the
            next, etc. By default, the most recently obtained result will be
            edited.

        separation_frequency
            The frequency separation given to adjacent oscillators formed
            from the splitting. If ``None``, the splitting will be set to
            ``sw / n`` where ``sw`` is the sweep width and ``n`` is the number
            of points in the data.

        unit
            The unit that ``separation_frequency`` is expressed in.

        split_number
            The number of peaks to split the oscillator into.

        amp_ratio
            The ratio of amplitudes to be fulfilled by the newly formed
            peaks. If a list, ``len(amp_ratio) == split_number`` must be
            satisfied. The first element will relate to the highest
            frequency oscillator constructed, and the last element will
            relate to the lowest frequency oscillator constructed. If `None`,
            all oscillators will be given equal amplitudes.

        estimate_kwargs
            Keyword arguments to provide to the call to :py:meth:`estimate`. Note
            that ``"initial_guess"`` and ``"region_unit"`` are set internally and
            will be ignored if given.
        """
        sanity_check(
            ("index", index, sfuncs.check_index, (len(self._results),)),
            (
                "separation_frequency", separation_frequency, sfuncs.check_float,
                (), {}, True,
            ),
            ("unit", unit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
            ("split_number", split_number, sfuncs.check_int, (), {"min_value": 2}),
        )
        index = self._positive_index(index)
        result = self._results[index]
        x0 = result.get_result()
        sanity_check(
            (
                "amp_ratio", amp_ratio, sfuncs.check_float_list, (),
                {
                    "length": split_number,
                    "must_be_positive": True,
                },
                True,
            ),
            (
                "oscillator", oscillator, sfuncs.check_int, (),
                {"min_value": 0, "max_value": x0.shape[0] - 1},
            ),
        )

        if separation_frequency is None:
            separation_frequency = self.sw("hz")[0] / self.default_pts[0]
        else:
            separation_frequency = (
                self.convert([separation_frequency], f"{unit}->hz")[0]
            )

        if amp_ratio is None:
            amp_ratio = np.ones((split_number,))
        else:
            amp_ratio = np.array(amp_ratio)

        # Of form: [a, φ, f, η] (i.e. 1D array)
        osc = x0[oscillator]
        amps = osc[0] * amp_ratio / amp_ratio.sum()
        # Highest frequency of all the new oscillators
        max_freq = osc[2] + ((split_number - 1) * separation_frequency / 2)
        # Array of all frequencies (lowest to highest)
        freqs = [max_freq - i * separation_frequency for i in range(split_number)]

        new_oscs = np.zeros((split_number, 4), dtype="float64")
        new_oscs[:, 0] = amps
        new_oscs[:, 1] = osc[1]
        new_oscs[:, 2] = freqs
        new_oscs[:, 3] = osc[3]

        x0 = np.delete(x0, oscillator, axis=0)
        x0 = np.vstack((x0, new_oscs))

        self._optimise_after_edit(x0, result, index, **estimate_kwargs)

    @logger
    def add_oscillators(
        self,
        params: np.ndarray,
        index: int = -1,
        **estimate_kwargs,
    ) -> None:
        """Add oscillators to an estimation result.

        Optimisation is carried out afterwards, on the updated set of oscillators.

        Parameters
        ----------
        params
            The parameters of new oscillators to be added. Should be of shape
            ``(n, 4)``, where ``n`` is the number of new oscillators to add. Even
            when one oscillator is being added this should be a 2D array, i.e.:

            .. code:: python3

                params = oscillators = np.array([[a, φ, f, η]])

        index
            The index of the result to edit. Index ``0`` corresponds to the
            first result obtained using the estimator, ``1`` corresponds to the
            next, etc. By default, the most recently obtained result will be
            edited.

        estimate_kwargs
            Keyword arguments to provide to the call to :py:meth:`estimate`. Note
            that ``"initial_guess"`` and ``"region_unit"`` are set internally and
            will be ignored if given.
        """
        sanity_check(
            (
                "params", params, sfuncs.check_ndarray, (),
                {"dim": 2, "shape": ((1, 4),)},
            ),
            ("index", index, sfuncs.check_index, (len(self._results),)),
        )
        index = self._positive_index(index)
        result = self._results[index]
        x0 = np.vstack((result.get_result(), params))
        self._optimise_after_edit(x0, result, index, **estimate_kwargs)

    @logger
    def remove_oscillators(
        self,
        oscillators: Iterable[int],
        index: int = -1,
        **estimate_kwargs,
    ) -> None:
        """Remove oscillators from an estimation result.

        Optimisation is carried out afterwards, on the updated set of oscillators.

        Parameters
        ----------
        oscillators
            A list of indices corresponding to the oscillators to be removed.

        index
            The index of the result to edit. Index ``0`` corresponds to the
            first result obtained using the estimator, ``1`` corresponds to the
            next, etc. By default, the most recently obtained result will be
            edited.

        estimate_kwargs
            Keyword arguments to provide to the call to :py:meth:`estimate`. Note
            that ``"initial_guess"`` and ``"region_unit"`` are set internally and
            will be ignored if given.
        """
        sanity_check(("index", index, sfuncs.check_index, (len(self._results),)))
        index = self._positive_index(index)
        result = self._results[index]
        x0 = result.get_result()
        sanity_check(
            (
                "oscillators", oscillators, sfuncs.check_int_list, (),
                {"min_value": 0, "max_value": x0.shape[0] - 1},
            ),
        )
        x0 = np.delete(x0, oscillators, axis=0)
        self._optimise_after_edit(x0, result, index, **estimate_kwargs)

    def _optimise_after_edit(
        self,
        x0: np.ndarray,
        result: Result,
        index: int,
        **estimate_kwargs,
    ) -> None:
        for key in estimate_kwargs.keys():
            if key in ("region_unit", "initial_guess", "fprint"):
                del estimate_kwargs[key]

        if getattr(estimate_kwargs, "fprint", None) is None:
            estimate_kwargs["fprint"] = True

        self.estimate(
            result.get_region(),
            result.get_noise_region(),
            region_unit="hz",
            initial_guess=x0,
            _log=False,
            **estimate_kwargs,
        )

        del self._results[index]
        self._results.insert(index, self._results.pop(-1))
        print([r.result for r in self._results])
