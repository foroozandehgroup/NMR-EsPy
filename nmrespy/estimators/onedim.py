# onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 25 Mar 2022 10:37:48 GMT

from __future__ import annotations
import datetime
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Optional, Tuple, Union
import uuid

import numpy as np

from nmr_sims.experiments.pa import PulseAcquireSimulation
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem

from nmrespy import ExpInfo, sig
from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._errors import TwoDimUnsupportedError
from nmrespy._files import (
    check_existent_dir,
    check_saveable_path,
    configure_path,
    save_file,
)
from nmrespy._misc import copydoc

from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)
from nmrespy.freqfilter import Filter
from nmrespy.load import load_bruker
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import NonlinearProgramming
from nmrespy.plot import (
    NmrespyPlot,
    plot_result,
)
from nmrespy.write import ResultWriter

from . import logger, Estimator, Result


if USE_COLORAMA:
    import colorama
    colorama.init()


class Estimator1D(Estimator):
    """Estimator class for 1D data."""

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
    @copydoc(Estimator.new_bruker)
    def new_bruker(
        cls,
        directory: Union[str, Path],
        ask_convdta: bool = True,
    ) -> Estimator1D:
        sanity_check(
            ("directory", directory, check_existent_dir),
            ("ask_convdta", ask_convdta, sfuncs.check_bool),
        )

        directory = Path(directory).resolve()
        data, expinfo = load_bruker(directory, ask_convdta=ask_convdta)

        if expinfo.dim != 1:
            raise ValueError(f"{RED}Data dimension should be 1.{END}")

        if directory.parent.name == "pdata":
            shape = tuple([slice(0, x // 2) for x in data.shape])
            data = (2 * data.ndim * sig.ift(data))[shape]

        return cls(data, expinfo, directory)

    @classmethod
    def new_synthetic_from_parameters(
        cls,
        parameters: np.ndarray,
        pts: int,
        sw: float,
        offset: float = 0.0,
        sfo: Optional[float] = None,
        snr: float = 30.0,
    ) -> Estimator1D:
        """Generate an estimator instance from an array of oscillator parameters.

        Parameters
        ----------
        parameters
            Parameter array with the following structure:

              .. code:: python

                 parameters = numpy.array([
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
            ("parameters", parameters, sfuncs.check_parameter_array, (expinfo.dim,)),
            ("pts", pts, sfuncs.check_positive_int),
            ("snr", snr, sfuncs.check_positive_float, (), True),
        )

        if expinfo.dim != 1:
            raise ValueError(f"{RED}Should be specifying as 1D signal.{END}")

        data = sig.make_fid(parameters, expinfo, [pts], snr=snr)[0]
        return cls(data, expinfo)

    @classmethod
    def new_synthetic_from_simulation(
        cls,
        spin_system: SpinSystem,
        sweep_width: float,
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

        sweep_width
            The sweep width.

        offset
            The transmitter offset frequency.

        pts
            The number of points sampled.

        freq_unit
            The unit that ``sweep_width`` and ``offset`` are expressed in. Should
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
            ("sweep_width", sweep_width, sfuncs.check_positive_float),
            ("offset", offset, sfuncs.check_float),
            ("pts", pts, sfuncs.check_positive_int),
            ("freq_unit", freq_unit, sfuncs.check_one_of, ("hz", "ppm")),
            ("channel", channel, sfuncs.check_nmrsims_nucleus),
            ("snr", snr, sfuncs.check_float, (), {}, True),
        )
        sweep_width = f"{sweep_width}{freq_unit}"
        offset = f"{offset}{freq_unit}"
        sim = PulseAcquireSimulation(
            spin_system, pts, sweep_width, offset=offset, channel=channel,
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

    def view_data(
        self,
        domain: str = "freq",
        components = "real",
    ) -> None:
        """View the data.

        Parameters
        ----------
        domain
            Must be ``"freq"`` or ``"time"``.

        components
            Must be ``"real"``, ``"imag"``, or ``"both"``.
        """
        sanity_check(
            ("domain", domain, sfuncs.check_one_of, ("freq", "time")),
            ("components", components, sfuncs.check_one_of, ("real", "imag", "both")),
        )



    @logger
    def estimate(
        self,
        region: Optional[Tuple[float, float]] = None,
        noise_region: Optional[Tuple[float, float]] = None,
        region_unit: str = "hz",
        initial_guess: Optional[Union[np.ndarray, int]] = None,
        method: str = "gauss-newton",
        phase_variance: bool = False,
        max_iterations: Optional[int] = None,
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

        timestamp = datetime.datetime.now()

        # The plan of action:
        # --> Derive filtered signals (both cut and uncut)
        # --> Run the MDL followed by MPM for an initial guess
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

        if isinstance(initial_guess, np.ndarray):
            x0 = initial_guess
        else:
            oscillators = initial_guess if isinstance(initial_guess, int) else 0
            x0 = MatrixPencil(
                cut_expinfo, cut_signal, oscillators=oscillators
            ).get_result()

        cut_result = NonlinearProgramming(
            cut_expinfo, cut_signal, x0, phase_variance=phase_variance, method=method,
            max_iterations=max_iterations,
        ).get_result()

        final_result = NonlinearProgramming(
            uncut_expinfo, uncut_signal, cut_result, phase_variance=phase_variance,
            method=method, max_iterations=max_iterations,
        )

        self._results.append(
            {
                "timestamp": timestamp,
                "region": region,
                "result": final_result.get_result(),
                "errors": final_result.get_errors(),
            }
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
            [result["result"] for result in results],
            [result["errors"] for result in results],
            description,
        )
        region_unit = "ppm" if self.hz_ppm_valid else "hz"
        print([
            self.convert(result["region"], f"hz->{region_unit}")
            for result in results
        ])

        region_unit = "ppm" if self.hz_ppm_valid else "hz"
        titles = [
            f"{left:.3f} - {right:.3f} {region_unit}".replace("h", "H")
            for left, right in [
                self.convert(result["region"], f"hz->{region_unit}")[0]
                for result in results
            ]
        ]
        print(titles)

        writer.write(
            path=path,
            fmt=fmt,
            titles=titles,
            parameters_sig_figs=sig_figs,
            parameters_sci_lims=sci_lims,
            force_overwrite=True,
            fprint=fprint,
        )

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
                [f"{r[1]:.4f} - {r[0]:.4f} ppm (F{i})"
                 for i, r in enumerate(region, start=1)]
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

    @copydoc(Estimator.plot_results)
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
        sanity_check(
            ("indices", indices, sfuncs.check_ints_less_than_n, (), True),
            ("plot_residual", plot_residual, sfuncs.check_bool),
            ("plot_model", plot_model, sfuncs.check_bool),
            ("residual_shift", residual_shift, sfuncs.check_float, (), True),
            ("model_shift", model_shift, sfuncs.check_float, (), True),
            ("shifts_unit", shifts_unit, sfuncs.check_frequency_unit),
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
            raise TwoDimUnsupportedError()
        return [
            plot_result(
                self._data,
                result.get_result(funit="hz"),
                self._expinfo,
                region=result.get_region(unit=shifts_unit),
                shifts_unit=shifts_unit,
            )
            for result in results
        ]
