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
from nmrespy.write import write_result
from nmrespy.write.pdffile import _write_pdf
from nmrespy.write.textfile import _write_txt

from . import copydoc, Estimator, Result


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
        super().__init__(data, expinfo, datapath)

    @classmethod
    @copydoc(Estimator.new_bruker)
    def new_bruker(
        cls, directory: Union[str, Path], ask_convdta: bool = True
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
        expinfo: ExpInfo,
        pts: int,
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

        expinfo
            Experiment information

        pts
            The number of points the signal comprises.

        snr
            The signal-to-noise ratio. If ``None`` then no noise will be added
            to the FID.
        """
        sanity_check(("expinfo", expinfo, sfuncs.check_expinfo),)

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
            for more details. **N.B. that the transmitter frequency (sfo) will
            be determined by ``spin_system.field``.

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
            ("snr", snr, sfuncs.check_float, (), True),
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

    # Estimator.@logger
    def estimate(
        self,
        region: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
        noise_region: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
        region_unit: str = "hz",
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
            One of ``"hz"`` or ``"ppm"``, corresponding to Hertz, parts per
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
            Specifies how to compute the Hessian at each iteration of the optimisation.

            * ``"exact"`` - the exact analytical Hessian will be computed.
            * ``"gauss-newton"`` - the Hessian will be approximated as per the
              Guass-Newton method.

        phase_variance
            Whether or not to include the variance of oscillator phases in the cost
            function. This should be set to ``True`` in cases where the signal being
            considered is derived from well-phased data.

        max_iterations
            The number of iterations to allow the optimiser to run before
            terminatingi if convergence has yet to be achieved. If ``None``, this
            number will be set to a default, depending on the identity of ``hessian``.
        """
        sanity_check(
            ("region_unit", region_unit, sfuncs.check_frequency_unit),
            (
                "initial_guess", initial_guess, sfuncs.check_initial_guess,
                (self.dim,), True
            ),
            ("hessian", hessian, sfuncs.check_one_of, ("gauss-newton", "exact")),
            ("phase_variance", phase_variance, sfuncs.check_bool),
            ("max_iterations", max_iterations, sfuncs.check_positive_int, (), True),
        )

        if region_unit == "hz":
            region_check = sfuncs.check_region_hz
        elif region_unit == "ppm":
            region_check = sfuncs.check_region_ppm

        sanity_check(
            ("region", region, region_check, (self._expinfo,), True),
            ("noise_region", noise_region, region_check, (self._expinfo,), True),
        )

        timestamp = datetime.datetime.now()
        filt = Filter(
            self._data,
            self._expinfo,
            region,
            noise_region,
            region_unit=region_unit,
        )

        signal, expinfo = filt.get_filtered_fid()
        region = filt.get_region()

        if isinstance(initial_guess, np.ndarray):
            x0 = initial_guess
        else:
            oscillators = initial_guess if isinstance(initial_guess, int) else 0
            x0 = MatrixPencil(signal, expinfo, oscillators=oscillators).get_result()

        nlp_result = NonlinearProgramming(
            signal, x0, expinfo, phase_variance=phase_variance, hessian=hessian,
            max_iterations=max_iterations,
        )
        result, errors = nlp_result.get_result(), nlp_result.get_errors()

        self._results.append(
            Result(timestamp, signal, expinfo, region, result, errors)
        )

    @copydoc(Estimator.write_results)
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
        sanity_check(
            (
                "indices", indices, sfuncs.check_ints_less_than_n,
                (len(self._results),), True,
            ),
            ("fmt", fmt, sfuncs.check_one_of, ("txt", "pdf")),
            ("description", description, sfuncs.check_str, (), True),
            ("sig_figs", sig_figs, sfuncs.check_positive_int, (), True),
            ("sci_lims", sci_lims, sfuncs.check_sci_lims, (), True),
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
            ("fprint", fprint, sfuncs.check_bool),
        )
        sanity_check(("path", path, check_saveable_path, (fmt, force_overwrite)))
        path = configure_path(path, fmt)

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
                sci_lims=sci_lims, force_overwrite=True, fprint=False,
            )
            if fmt == "pdf":
                fname = fname.with_suffix(".tex")
            else:
                fname = fname.with_suffix(f".{fmt}")

            with open(fname, "r") as fh:
                contents.append((fh.read(), result.get_region(unit="ppm")))

        if fmt == "txt":
            txt = self._process_txt(contents)
            save_file(txt, path, fprint=fprint)
        elif fmt == "pdf":
            txt = self._process_pdf(contents)
            save_file(txt, path, fprint=fprint)

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
