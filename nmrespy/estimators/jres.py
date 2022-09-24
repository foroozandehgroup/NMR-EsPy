# jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 23 Sep 2022 15:52:09 BST

from __future__ import annotations
import copy
import io
import itertools
from pathlib import Path
import re
import sys
import tkinter as tk
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)

from nmr_sims.experiments.jres import JresSimulation
from nmr_sims.spin_system import SpinSystem

from nmrespy import MATLAB_AVAILABLE, ExpInfo, sig
from nmrespy.plot import make_color_cycle
from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._files import cd, check_existent_dir
from nmrespy._paths_and_links import SPINACHPATH
from nmrespy.app.custom_widgets import MyEntry
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)
from nmrespy.estimators import logger, Estimator, Result
from nmrespy.freqfilter import Filter
from nmrespy.load import load_bruker
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import NonlinearProgramming


if USE_COLORAMA:
    import colorama
    colorama.init()

if MATLAB_AVAILABLE:
    import matlab
    import matlab.engine


class Estimator2DJ(Estimator):
    def __init__(
        self, data: np.ndarray, expinfo: ExpInfo, datapath: Optional[Path] = None,
    ) -> None:
        super().__init__(data, expinfo, datapath)

    @classmethod
    def new_bruker(
        cls,
        directory: Union[str, Path],
        convdta: bool = True,
    ) -> Estimator2DJ:
        """Create a new instance from Bruker-formatted data.

        Parameters
        ----------
        directory
            Absolute path to data directory.

        convdta
            If ``True``, removal of the FID's digital filter will be carried out.

        Notes
        -----
        **Directory Requirements**

        There are certain file paths expected to be found relative to ``directory``
        which contain the data and parameter files:

        * ``directory/ser``
        * ``directory/acqus``
        * ``directory/acqu2s``
        """
        sanity_check(
            ("directory", directory, check_existent_dir),
            ("convdta", convdta, sfuncs.check_bool),
        )

        directory = Path(directory).expanduser()
        data, expinfo = load_bruker(directory)

        if data.ndim != 2:
            raise ValueError(f"{RED}Data dimension should be 2.{END}")

        if directory.parent.name == "pdata":
            raise ValueError(f"{RED}Importing pdata is not permitted.{END}")

        if convdta:
            grpdly = expinfo.parameters["acqus"]["GRPDLY"]
            data = sig.convdta(data, grpdly)

        expinfo._offset = (0., expinfo.offset()[1])
        expinfo._sfo = (None, expinfo.sfo[1])
        expinfo._default_pts = data.shape

        return cls(data, expinfo, directory)

    @classmethod
    def new_spinach(
        cls,
        shifts: Iterable[float],
        pts: Tuple[int, int],
        sw: Tuple[float, float],
        offset: float,
        field: float = 11.74,
        field_unit: str = "tesla",
        couplings: Optional[Iterable[Tuple(int, int, float)]] = None,
        channel: str = "1H",
        nuclei: Optional[List[str]] = None,
        snr: Optional[float] = 20.,
        lb: Optional[Tuple[float, float]] = (6.91, 6.91),
    ) -> None:
        r"""Create a new instance from a 2DJ Spinach simulation.

        Parameters
        ----------
        shifts
            A list or tuple of chemical shift values for each spin.

        pts
            The number of points the signal comprises.

        sw
            The sweep width of the signal (Hz).

        offset
            The transmitter offset (Hz).

        field
            The magnetic field stength, in either Tesla or MHz (see ``field_unit``).

        field_unit
            ``MHz`` or ``Tesla``. The unit that ``field`` is given as. If ``MHz``,
            this will be taken as the Larmor frequency of proton.

        couplings
            The scalar couplings present in the spin system. Given ``shifts`` is of
            length ``n``, couplings should be an iterable with entries of the form
            ``(i1, i2, coupling)``, where ``1 <= i1, i2 <= n`` are the indices of
            the two spins involved in the coupling, and ``coupling`` is the value
            of the scalar coupling in Hz.

        channel
            The identity of the nucleus targeted in the pulse sequence.

        nuclei
            The type of nucleus for each spin. Can be either:

            * ``None``, in which case each spin will be set as the identity of
              ``channel``.
            * A list of length ``n``, where ``n`` is the number of spins. Each
              entry should be a string satisfying the regular expression
              ``"\d+[A-Z][a-z]*"``, and recognised by Spinach as a real nucleus
              e.g. ``"1H"``, ``"13C"``, ``"195Pt"``.

        snr
            The signal-to-noise ratio of the resulting signal, in decibels. ``None``
            produces a noiseless signal.

        lb
            Line broadening (exponential damping) to apply to the signal. If a tuple
            of two floats, damping in T1 will be dictated by ``lb[0]`` and damping
            in T2 will be dictated by ``lb[1]``. Note that the first point will be
            unaffected by damping, and the final point will be multiplied by
            ``np.exp(-lb[i])`` for each dimension. The default results in the final
            point being decreased in value by a factor of roughly 1000.
        """
        if not MATLAB_AVAILABLE:
            raise NotImplementedError(
                f"{RED}MATLAB isn't accessible to Python. To get up and running, "
                "take at look here:\n"
                "https://www.mathworks.com/help/matlab/matlab_external/"
                f"install-the-matlab-engine-for-python.html{END}"
            )

        sanity_check(
            ("shifts", shifts, sfuncs.check_float_list),
            ("pts", pts, sfuncs.check_int_list, (), {"length": 2}),
            (
                "sw", sw, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True},
            ),
            ("offset", offset, sfuncs.check_float),
            ("channel", channel, sfuncs.check_nucleus),
            ("field", field, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("field_unit", field_unit, sfuncs.check_one_of, ("tesla", "MHz")),
            ("snr", snr, sfuncs.check_float, (), {}, True),
            (
                "lb", lb, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True},
            ),
        )

        nspins = len(shifts)
        sanity_check(
            ("nuclei", nuclei, sfuncs.check_nucleus_list, (), {"length": nspins}, True),
            (
                "couplings", couplings, sfuncs.check_spinach_couplings, (nspins,),
                {}, True,
            ),
        )

        if couplings is None:
            couplings = []

        if nuclei is None:
            nuclei = nspins * [channel]

        if field_unit == "MHz":
            field = (2e6 * np.pi * field) / 2.6752218744e8

        with cd(SPINACHPATH):
            # devnull = io.StringIO(str(os.devnull))
            try:
                eng = matlab.engine.start_matlab()
                fid, sfo = eng.jres_sim(
                    field, nuclei, shifts, couplings, offset,
                    matlab.double(sw), matlab.double(pts), channel, nargout=2,
                    # stdout=devnull, stderr=devnull,
                )
            except matlab.engine.MatlabExecutionError:
                raise ValueError(
                    f"{RED}Something went wrong in trying to run Spinach. This "
                    "is likely due to one of two things:\n"
                    "1. An inappropriate argument was given which was not noticed by "
                    "sanity checks. For example, you provided an isotope of the "
                    "correct format but which is unknown\n"
                    "2. You have not correctly configured Spinach.\n"
                    "Read what is stated below the line "
                    "\"matlab.engine.MatlabExecutionError:\" "
                    f"for more details on the error raised.{END}"
                )

        fid = sig.phase(np.array(fid), (0., np.pi / 2), (0., 0.))

        # Apply exponential damping
        for i, k in enumerate(lb):
            fid = sig.exp_apodisation(fid, k, axes=[i])

        if snr is not None:
            fid = sig.add_noise(fid, snr)

        expinfo = ExpInfo(
            dim=2,
            sw=sw,
            offset=(0., offset),
            sfo=(None, sfo),
            nuclei=(None, channel),
            default_pts=fid.shape,
        )

        return cls(fid, expinfo)

    @classmethod
    def new_nmrsims(
        cls,
        shifts: Iterable[float],
        pts: Tuple[int, int],
        sw: Tuple[float, float],
        offset: float,
        field: float = 11.74,
        field_unit: str = "tesla",
        couplings: Optional[Iterable[Tuple(int, int, float)]] = None,
        channel: str = "1H",
        nuclei: Optional[List[str]] = None,
        snr: Optional[float] = 20.,
        lb: Optional[Tuple[float, float]] = (6.91, 6.91),
    ) -> Estimator2DJ:
        r"""Create a new instance from a 2DJ NMR Sims simulation.

        Parameters
        ----------
        shifts
            A list or tuple of chemical shift values for each spin.

        pts
            The number of points the signal comprises.

        sw
            The sweep width of the signal (Hz).

        offset
            The transmitter offset (Hz).

        field
            The magnetic field stength, in either Tesla or MHz (see ``field_unit``).

        field_unit
            ``MHz`` or ``Tesla``. The unit that ``field`` is given as. If ``MHz``,
            this will be taken as the Larmor frequency of the nucleus specified by
            ``channel``.

        couplings
            The scalar couplings present in the spin system. Given ``shifts`` is of
            length ``n``, couplings should be an iterable with entries of the form
            ``(i1, i2, coupling)``, where ``1 <= i1, i2 <= n`` are the indices of
            the two spins involved in the coupling, and ``coupling`` is the value
            of the scalar coupling in Hz.

        channel
            The identity of the nucleus targeted in the pulse sequence.

        nuclei
            The type of nucleus for each spin. Can be either:

            * ``None``, in which case each spin will be set as the identity of
              ``channel``.
            * A list of length ``n``, where ``n`` is the number of spins. Each
              entry should be a string satisfying the regular expression
              ``"\d+[A-Z][a-z]*"``, and recognised as a real nucleus e.g. ``"1H"``,
              ``"13C"``.

        snr
            The signal-to-noise ratio of the resulting signal, in decibels. ``None``
            produces a noiseless signal.

        lb
            Line broadening (exponential damping) to apply to the signal. If a tuple
            of two floats, damping in T1 will be dictated by ``lb[0]`` and damping
            in T2 will be dictated by ``lb[1]``. Note that the first point will be
            unaffected by damping, and the final point will be multiplied by
            ``np.exp(-lb[i])`` for each dimension. The default results in the final
            point being decreased in value by a factor of roughly 1000.
        """
        sanity_check(
            ("shifts", shifts, sfuncs.check_float_list),
            ("pts", pts, sfuncs.check_int_list, (), {"length": 2}),
            (
                "sw", sw, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True},
            ),
            ("offset", offset, sfuncs.check_float),
            ("field", field, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("field_unit", field_unit, sfuncs.check_one_of, ("tesla", "MHz")),
            ("channel", channel, sfuncs.check_nmrsims_nucleus),
            ("snr", snr, sfuncs.check_float),
            (
                "lb", lb, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True}, True,
            ),
        )

        nspins = len(shifts)
        sanity_check(
            ("nuclei", nuclei, sfuncs.check_nucleus_list, (), {"length": nspins}, True),
            (
                "couplings", couplings, sfuncs.check_spinach_couplings, (nspins,),
                {}, True,
            ),
        )

        if nuclei is None:
            nuclei = nspins * [channel]

        # Construct SpinSystem object
        spins = {
            i: {"shift": shift, "nucleus": nucleus}
            for i, (shift, nucleus) in enumerate(zip(shifts, nuclei), start=1)
        }
        for i1, i2, coupling in couplings:
            i1, i2 = min(i1, i2), max(i1, i2)
            if "couplings" not in spins[i1]:
                spins[i1]["couplings"] = {}
            spins[i1]["couplings"][i2] = coupling

        if field_unit == "tesla":
            field = f"{field}T"
        elif field_unit == "MHz":
            field = f"{field}MHz"

        spin_system = SpinSystem(spins, field=field)

        # Prevent normal nmr_sims output from appearing
        text_trap = io.StringIO()
        sys.stdout = text_trap

        sim = JresSimulation(spin_system, pts, sw, offset, channel)
        sim.simulate()

        sys.stdout = sys.__stdout__

        if lb is None:
            # Determine factor to ensure that final point dampened by a factor
            # of 1/1000
            lb = tuple([-(1 / x) * np.log(0.001) for x in pts])

        _, fid, _ = sim.fid(lb=(0., 0.))

        if snr is not None:
            fid = sig.add_noise(fid, snr)

        # Apply exponential damping
        for i, k in enumerate(lb):
            fid = sig.exp_apodisation(fid, k, axes=[i])

        expinfo = ExpInfo(
            dim=2,
            sw=sim.sweep_widths,
            offset=(0., sim.offsets[0]),
            sfo=(None, sim.sfo[0]),
            nuclei=(None, sim.channels[0].name),
            default_pts=fid.shape,
            fn_mode="QF",
        )

        return cls(fid, expinfo)

    def view_data(
        self,
        domain: str = "freq",
        abs_: bool = False,
    ) -> None:
        """View the data.

        Parameters
        ----------
        domain
            Must be ``"freq"`` or ``"time"``.

        abs_
            Whether or not to display frequency-domain data in absolute-value mode.
        """
        sanity_check(
            ("domain", domain, sfuncs.check_one_of, ("freq", "time")),
            ("abs_", abs_, sfuncs.check_bool),
        )

        if domain == "freq":
            spectrum = np.abs(self.spectrum) if abs_ else self.spectrum
            app = ContourApp(spectrum, self.expinfo)
            app.mainloop()

        elif domain == "time":
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            x, y = self.get_timepoints()
            xlabel, ylabel = [f"$t_{i}$ (s)" for i in range(1, 3)]

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(reversed(ax.get_xlim()))
            ax.set_ylim(reversed(ax.get_ylim()))
            ax.set_zticks([])

            plt.show()

    @property
    def direct_expinfo(self) -> ExpInfo:
        """Return :py:meth:`~nmrespy.ExpInfo` for the direct dimension."""
        return ExpInfo(
            dim=1,
            sw=self.sw()[1],
            offset=self.offset()[1],
            sfo=self.sfo[1],
            nuclei=self.nuclei[1],
            default_pts=self.default_pts[1],
        )

    @property
    def spectrum_zero_t1(self) -> np.ndarray:
        """Generate a 1D spectrum of the first time-slice in the indirect dimension."""
        data = copy.deepcopy(self.data[0])
        data[0] *= 0.5
        return sig.ft(data)

    @property
    def spectrum(self) -> np.ndarray:
        data = copy.deepcopy(self.data)
        data[0, 0] *= 0.5
        return sig.ft(data)

    @logger
    def estimate(
        self,
        region: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
        noise_region: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
        region_unit: str = "ppm",
        initial_guess: Optional[Union[np.ndarray, int]] = None,
        method: str = "gauss-newton",
        phase_variance: bool = True,
        max_iterations: Optional[int] = None,
        cut_ratio: Optional[float] = 1.1,
        mpm_trim: Optional[int] = 256,
        nlp_trim: Optional[int] = 1024,
        fprint: bool = True,
        _log: bool = True,
    ):
        r"""Estimate a specified region in F2.

        The basic steps that this method carries out are:

        * (Optional, but highly advised) Generate a frequency-filtered signal
          corresponding to the specified region.
        * (Optional) Generate an inital guess using the Matrix Pencil Method (MPM).
        * Apply numerical optimisation to determine a final estimate of the signal
          parameters

        Parameters
        ----------
        region
            The frequency range of interest in F2. Should be of the form
            ``(left, right)`` where ``left`` and ``right`` are the left and right
            bounds of the region of interest. If ``None``, the full signal will be
            considered, though for sufficently large and complex signals it is
            probable that poor and slow performance will be realised.

        noise_region
            If ``region`` is not ``None``, this must be of the form ``(left, right)``
            too. This should specify a frequency range in F2 where no noticeable
            signals reside, i.e. only noise exists.

        region_unit
            One of ``"hz"`` or ``"ppm"``. Specifies the units that ``region`` and
            ``noise_region`` have been given as.

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
            Specifies the maximal size in the direct dimension allowed for the
            filtered signal when undergoing the Matrix Pencil. If ``None``, no
            trimming is applied to the signal. If an int, and the direct
            dimension filtered signal has a size greater than ``mpm_trim``,
            this signal will be set as ``signal[:, :mpm_trim]``.

        nlp_trim
            Specifies the maximal size allowed in the direct dimension for the
            filtered signal when undergoing nonlinear programming. By default
            (``None``), no trimming is applied to the signal. If an int, and
            the direct dimension filtered signal has a size greater than
            ``nlp_trim``, this signal will be set as ``signal[:, :nlp_trim]``.

        fprint
            Whether of not to output information to the terminal.

        _log
            Ignore this!
        """
        sanity_check(
            (
                "region_unit", region_unit, sfuncs.check_frequency_unit,
                (self.hz_ppm_valid,)
            ),
            (
                "initial_guess", initial_guess, sfuncs.check_initial_guess,
                (self.dim,), {}, True
            ),
            ("method", method, sfuncs.check_one_of, ("gauss-newton", "exact", "lbfgs")),
            ("phase_variance", phase_variance, sfuncs.check_bool),
            (
                "max_iterations", max_iterations, sfuncs.check_int, (),
                {"min_value": 1}, True,
            ),
            (
                "cut_ratio", cut_ratio, sfuncs.check_float, (),
                {"greater_than_one": True}, True,
            ),
            ("mpm_trim", mpm_trim, sfuncs.check_int, (), {"min_value": 1}, True),
            ("nlp_trim", nlp_trim, sfuncs.check_int, (), {"min_value": 1}, True),
            ("fprint", fprint, sfuncs.check_bool),
        )

        sanity_check(
            (
                "region", region, sfuncs.check_region,
                (
                    (self.sw(region_unit)[1],),
                    (self.offset(region_unit)[1],),
                ), {}, True,
            ),
            (
                "noise_region", noise_region, sfuncs.check_region,
                (
                    (self.sw(region_unit)[1],),
                    (self.offset(region_unit)[1],),
                ), {}, True,
            ),
        )

        if region is None:
            region = self.convert(
                ((0, self._data.shape[0] - 1), (0, self._data.shape[1] - 1)),
                "idx->hz",
            )
            noise_region = None
            mpm_signal = nlp_signal = self._data
            mpm_expinfo = nlp_expinfo = self.expinfo

        else:
            region = (None, region)
            noise_region = (None, noise_region)

            filt = Filter(
                self._data,
                self.expinfo,
                region,
                noise_region,
                region_unit=region_unit,
                twodim_dtype="jres",
            )

            mpm_signal, mpm_expinfo = filt.get_filtered_fid(cut_ratio=cut_ratio)
            nlp_signal, nlp_expinfo = filt.get_filtered_fid(cut_ratio=None)
            region = filt.get_region()
            noise_region = filt.get_noise_region()

        if (mpm_trim is None) or (mpm_trim > mpm_signal.shape[1]):
            mpm_trim = mpm_signal.shape[1]
        if (nlp_trim is None) or (nlp_trim > nlp_signal.shape[1]):
            nlp_trim = nlp_signal.shape[1]

        if isinstance(initial_guess, np.ndarray):
            x0 = initial_guess
        else:
            oscillators = initial_guess if isinstance(initial_guess, int) else 0
            x0 = MatrixPencil(
                mpm_expinfo,
                mpm_signal[:, :mpm_trim],
                oscillators=oscillators,
                fprint=fprint,
            ).get_params()

            if x0 is None:
                return self._results.append(
                    Result(
                        np.array([[]]),
                        np.array([[]]),
                        region,
                        noise_region,
                        self.sfo,
                    )
                )

        result = NonlinearProgramming(
            nlp_expinfo,
            nlp_signal[:, :nlp_trim],
            x0,
            phase_variance=phase_variance,
            method=method,
            max_iterations=max_iterations,
            fprint=fprint,
        )

        self._results.append(
            Result(
                result.get_params(),
                result.get_errors(),
                region,
                noise_region,
                self.sfo,
            )
        )

    def subband_estimate(
        self,
        noise_region: Tuple[float, float],
        noise_region_unit: str = "hz",
        nsubbands: Optional[int] = None,
        method: str = "gauss-newton",
        phase_variance: bool = True,
        max_iterations: Optional[int] = None,
        cut_ratio: Optional[float] = 1.1,
        mpm_trim: Optional[int] = 128,
        nlp_trim: Optional[int] = 256,
        fprint: bool = True,
        _log: bool = True,
    ) -> None:
        r"""Perform estiamtion on the entire signal via estimation of
        frequency-filtered sub-bands.

        This method splits the signal up into ``nsubbands`` equally-sized regions
        in the direct dimension and extracts parameters from each region before
        finally concatenating all the results together.

        Parameters
        ----------
        noise_region
            Specifies a direct dimension frequency range where no noticeable
            signals reside, i.e. only noise exists.

        noise_region_unit
            One of ``"hz"`` or ``"ppm"``. Specifies the units that ``noise_region``
            have been given in.

        nsubbands
            The number of sub-bands to break the signal into. If ``None``, the number
            will be set as the nearest integer to the data size divided by 500.

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
            Specifies the maximal size in the direct dimension allowed for the
            filtered signal when undergoing the Matrix Pencil. If ``None``, no
            trimming is applied to the signal. If an int, and the filtered
            signal has a direct dimension size greater than ``mpm_trim``, this
            signal will be set as ``signal[:, :mpm_trim]``.

        nlp_trim
            Specifies the maximal size allowed in the direct dimension for the
            filtered signal when undergoing nonlinear programming. If ``None``,
            no trimming is applied to the signal. If an int, and the filtered
            signal has a direct dimension size greater than ``nlp_trim``, this
            signal will be set as ``signal[:, :nlp_trim]``.

        fprint
            Whether of not to output information to the terminal.

        _log
            Ignore this!
        """
        sanity_check(
            (
                "noise_region_unit", noise_region_unit, sfuncs.check_frequency_unit,
                (self.hz_ppm_valid,),
            ),
            ("nsubbands", nsubbands, sfuncs.check_int, (), {"min_value": 1}, True),
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
                "noise_region", noise_region, sfuncs.check_region,
                (
                    (self.sw(noise_region_unit)[1],),
                    (self.offset(noise_region_unit)[1],),
                ), {}, True,
            ),
        )

        kwargs = {
            "method": method,
            "phase_variance": phase_variance,
            "max_iterations": max_iterations,
            "cut_ratio": cut_ratio,
            "mpm_trim": mpm_trim,
            "nlp_trim": nlp_trim,
            "fprint": fprint,
        }

        self._subband_estimate(nsubbands, noise_region, noise_region_unit, **kwargs)

    def diagonal_signal(
        self,
        positive_t1: bool = False,
        positive_t2: bool = True,
        indices: Optional[Iterable[int]] = None,
        pts: Optional[int] = None,
    ) -> np.ndarray:
        r"""Generate the synthetic signal :math:`y_{-45^{\circ}}(t)`, where
        :math:`t \geq 0`:

        .. math::

            y_{-45^{\circ}}(t) = \sum_{m=1}^M a_m \exp\left( \mathrm{i} \phi_m \right)
            \exp\left( 2 \mathrm{i} \pi f_{1,m} t \right)
            \exp\left( -t \left[2 \mathrm{i} \pi f_{2,m} + \eta_{2,m} \right] \right)

        .. image:: https://raw.githubusercontent.com/foroozandehgroup/NMR-EsPy/2dj/nmrespy/images/neg_45.png  # noqa: E501

        Producing this signal from parameters derived from estimation of a 2DJ dataset
        should generate a 1D homodecoupled spectrum.

        Parameters
        ----------
        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        pts
            The number of points to construct the signal from. If ``None``,
            ``self.default_pts`` will be used.
        """
        self._check_results_exist()
        sanity_check(
            ("positive_t1", positive_t1, sfuncs.check_bool),
            ("positive_t2", positive_t2, sfuncs.check_bool),
            (
                "indices", indices, sfuncs.check_int_list, (),
                {
                    "len_one_can_be_listless": True,
                    "min_value": -len(self._results),
                    "max_value": len(self._results) - 1,
                },
                True,

            ),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}, True),
        )

        params = self.get_params(indices)
        offset = self.offset()[1]
        if pts is None:
            pts = self.default_pts[1]
        tp = self.get_timepoints(pts=(1, pts), meshgrid=False)[1]
        f1 = params[:, 2] if positive_t1 else -params[:, 2]
        f2 = params[:, 3] if positive_t2 else -params[:, 3]
        signal = np.einsum(
            "ij,j->i",
            np.exp(
                np.outer(
                    tp,
                    2j * np.pi * (f1 + f2 - offset) - params[:, 5],
                )
            ),
            params[:, 0] * np.exp(1j * params[:, 1])
        )

        return signal

    def predict_multiplets(
        self,
        indices: Optional[Iterable[int]] = None,
        thold: Optional[float] = None,
    ) -> Iterable[Iterable[int]]:
        """Predict the estimated oscillators which correspond to each multiplet
        in the signal.

        Parameters
        ----------
        thold
            Frequency threshold. All oscillators that make up a multiplet are assumed
            to obey the following expression:

            .. math::
                f_c - f_t < f_{2,m} - f_{1,m} < f_c + f_t

            where :math:`f_c` is the central frequency of the multiplet, and `f_t` is
            ``thold``
        """
        self._check_results_exist()
        length = len(self._results)
        sanity_check(
            (
                "indices", indices, sfuncs.check_int_list, (),
                {"min_value": 0, "max_value": length - 1}, True,
            ),
            ("thold", thold, sfuncs.check_float, (), {"greater_than_zero": True}, True),
        )
        if thold is None:
            thold = 0.5 * (self.sw()[0] / self.default_pts[0])

        params = self.get_params(indices)
        groups = {}
        in_range = lambda f, g: (g - thold < f < g + thold)
        for i, osc in enumerate(params):
            centre_freq = osc[3] - osc[2]
            assigned = False
            for freq in groups:
                if in_range(centre_freq, freq):
                    groups[freq].append(i)
                    assigned = True
                    break
            if not assigned:
                groups[centre_freq] = [i]

        multiplets = []
        for freq in sorted(groups):
            multiplets.append(groups[freq])

        return multiplets

    def find_spurious_oscillators(
        self,
        thold: Optional[float] = None,
    ) -> Dict[int, Iterable[int]]:
        r"""Predict which oscillators are spurious.

        This predicts the multiplet structures in the estimationm result, and then
        purges all oscillators which fall into the following criteria:

        * The oscillator is the only one in the multiplet.
        * The frequency in F1 is greater than ``thold``.

        Parameters
        ----------
        thold
            Frequency threshold within which :math:`f_2 - f_1` of the oscillators
            in a multiplet should agree. If ``None``, this is set to be
            :math:`N_1 / 2 f_{\mathrm{sw}, 1}``

        Returns
        -------
        A dictionary with int keys corresponding to result indices, and list
        values corresponding to oscillators which are deemed spurious.
        """
        self._check_results_exist()
        sanity_check(
            ("thold", thold, sfuncs.check_float, (), {"greater_than_zero": True}, True),
        )
        if thold is None:
            thold = 0.5 * (self.default_pts[0] / self.sw()[0])

        params = self.get_params()
        multiplets = self.predict_multiplets(thold=thold)
        spurious = {}
        for multiplet in multiplets:
            if len(multiplet) == 1 and abs(params[multiplet[0], 2]) > thold:
                osc_loc = self.find_osc(params[multiplet[0]])
                if osc_loc[0] in spurious:
                    spurious[osc_loc[0]].append(osc_loc[1])
                else:
                    spurious[osc_loc[0]] = [osc_loc[1]]

        return spurious

    def remove_spurious_oscillators(
        self,
        thold: Optional[float] = None,
        **estimate_kwargs,
    ) -> None:
        r"""Attempt to remove spurious oscillators from the estimation result.

        See :py:meth:`find_spurious_oscillators` for information on how spurious
        oscillators are predicted.

        Oscillators deemed spurious are removed using :py:meth:`remove_oscillators`.

        Parameters
        ----------
        thold
            Frequency threshold within which :math:`f_2 - f_1` of the oscillators
            in a multiplet should agree. If ``None``, this is set to be
            :math:`N_1 / 2 f_{\mathrm{sw}, 1}``

        estimate_kwargs
            Keyword arguments to provide to :py:meth:`remove_oscillators`. Note
            that ``"initial_guess"`` and ``"region_unit"`` are set internally and
            will be ignored if given.
        """
        self._check_results_exist()
        sanity_check(
            ("thold", thold, sfuncs.check_float, (), {"greater_than_zero": True}, True),
        )
        spurious = self.find_spurious_oscillators(thold)
        for res_idx, osc_idx in spurious.items():
            self.remove_oscillators(osc_idx, res_idx, **estimate_kwargs)

    def sheared_signal(
        self,
        indices: Optional[Iterable[int]] = None,
        pts: Optional[Tuple[int, int]] = None,
        indirect_modulation: Optional[str] = None,
    ) -> np.ndarray:
        r"""Return an FID where direct dimension frequencies are perturbed such that:

        .. math::

            f_{2, m} = f_{2, m} - f_{1, m}\ \forall\ m \in \{1, \cdots, M\}

        This should yeild a signal where all components in a multiplet are centered
        at the spin's chemical shift in the direct dimenion, akin to "shearing" 2DJ
        data.

        Parameters
        ----------
        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        pts
            The number of points to construct the signal from. If ``None``,
            ``self.default_pts`` will be used.

        indirect_modulation
            Acquisition mode in indirect dimension of a 2D experiment. If the
            data is not 1-dimensional, this should be one of:

            * ``None`` - :math:`y \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              e^{\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`
            * ``"amp"`` - amplitude modulated pair:
              :math:`y_{\mathrm{cos}} \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              \cos\left(\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1\right)
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`
              :math:`y_{\mathrm{sin}} \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              \sin\left(\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1\right)
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`
            * ``"phase"`` - phase-modulated pair:
              :math:`y_{\mathrm{P}} \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              e^{\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`
              :math:`y_{\mathrm{N}} \left(t_1, t_2\right) = \sum_{m} a_m
              e^{\mathrm{i} \phi_m}
              e^{\left(-2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
              e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}`

            ``None`` will lead to an array of shape ``(*pts)``. ``amp`` and ``phase``
            will lead to an array of shape ``(2, *pts)``.
        """
        self._check_results_exist()
        sanity_check(
            (
                "indices", indices, sfuncs.check_index,
                (len(self._results),), {}, True,
            ),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}, True),
        )

        edited_params = copy.deepcopy(self.get_params(indices))
        edited_params[:, 3] -= edited_params[:, 2]

        return super(Estimator, self).make_fid(
            edited_params, pts=pts, indirect_modulation=indirect_modulation,
        )

    def exp_apodisation(self, k: float) -> None:
        """Apply an exponential window function to the direct dimnsion of the data.

        The window function is computed as ``np.exp(-k * np.linspace(0, 1, n2))``,
        where ``n2`` is the number of points in the direct dimension.
        """
        sanity_check(("k", k, sfuncs.check_float, (), {"greater_than_zero": True}))
        self._data = sig.exp_apodisation(self._data, k, axes=[1])

    def baseline_correction(
        self,
        min_length: int = 50,
    ) -> None:
        sanity_check(
            (
                "min_length", min_length, sfuncs.check_int, (),
                {"min_value": 1, "max_value": self.data.shape[1]},
            ),
        )
        shape = self.data.shape
        t2_size = (2 * shape[1] - 1) // 2
        new_data = np.zeros((shape[0], t2_size), dtype="complex128")
        for i, t2_fid in enumerate(self.data):
            spectrum = sig.ft(sig.make_virtual_echo(t2_fid)).real
            spectrum, _ = sig.baseline_correction(spectrum, min_length=min_length)
            new_data[i] = sig.ift(spectrum)[:t2_size]

        self._data = new_data
        self._default_pts = (shape[0], t2_size)

    def phase_data(
        self,
        p0: float = 0.0,
        p1: float = 0.0,
        pivot: int = 0,
    ) -> None:
        """Apply a first-order phase correction in the direct dimension.

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
        self._data = sig.ift(
            sig.phase(
                sig.ft(
                    self._data,
                    axes=[1],
                ),
                p0=[0., p0],
                p1=[0., p1],
                pivot=[0, pivot],
            ),
            axes=[1],
        )

    def manual_phase_data(
        self,
        max_p1: float = 10 * np.pi,
    ) -> Tuple[float, float]:
        sanity_check(
            ("max_p1", max_p1, sfuncs.check_float, (), {"greater_than_zero": True}),
        )
        p0, p1 = sig.manual_phase_data(self.spectrum_zero_t1, max_p1=[max_p1])
        p0, p1 = p0[0], p1[0]
        self.phase_data(p0=p0, p1=p1)
        return p0, p1

    def plot_result(
        self,
        indices: Optional[Iterable[int]] = None,
        multiplet_thold: Optional[float] = None,
        high_resolution_pts: Optional[int] = None,
        ratio_1d_2d: Tuple[float, float] = (2., 1.),
        figure_size: Tuple[float, float] = (8., 6.),
        axes_left: float = 0.07,
        axes_right: float = 0.96,
        axes_bottom: float = 0.08,
        axes_top: float = 0.96,
        axes_region_separation: float = 0.05,
        xaxis_label_height: float = 0.02,
        xaxis_ticks: Optional[Iterable[Tuple[int, Iterable[float]]]] = None,
        contour_base: Optional[float] = None,
        contour_nlevels: Optional[int] = None,
        contour_factor: Optional[float] = None,
        contour_lw: float = 0.5,
        contour_color: Any = "k",
        multiplet_colors: Any = "rainbow",
        multiplet_lw: float = 1.,
        multiplet_vertical_shift: float = 0.7,
        multiplet_show_center_freq: bool = True,
        multiplet_show_45: bool = True,
        marker_size: float = 3.,
        marker_shape: str = "o",
        label_peaks: bool = False,
        denote_regions: bool = False,
    ) -> Tuple[mpl.figure.Figure, np.ndarray[mpl.axes.Axes]]:
        """Generate a figure of the estimation result.

        The figure includes a contour plot of the 2DJ spectrum, a 1D plot of the
        first slice through the indirect dimension, plots of estimated multiplets,
        and a plot of the spectrum generated from :py:meth:`diagonal_signal`.

        Parameters
        ----------
        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        multiplet_thold
            Frequency threshold for multiplet prediction. All oscillators that make
            up a multiplet are assumed to obey the following expression:

            .. math::
                f_c - f_t < f^{(2)} - f^{(1)} < f_c + f_t

            where :math:`f_c` is the central frequency of the multiplet, and `f_t` is
            ``multiplet_thold``

        high_resolution_pts
            Indicates the number of points used to generate the multiplet structures
            and :py:meth:`diagonal_signal` spectrum. Should be greater than or
            equal to ``self.default_pts[1]``.

        ratio_1d_2d
            The relative heights of the regions containing the 1D spectra and the
            2DJ spectrum.

        figure_size
            The size of the figure in inches.

        axes_left
            The position of the left edge of the axes, in figure coordinates. Should
            be between ``0.`` and ``1.``.

        axes_right
            The position of the right edge of the axes, in figure coordinates. Should
            be between ``0.`` and ``1.``.

        axes_top
            The position of the top edge of the axes, in figure coordinates. Should
            be between ``0.`` and ``1.``.

        axes_bottom
            The position of the bottom edge of the axes, in figure coordinates. Should
            be between ``0.`` and ``1.``.

        axes_region_separation
            The extent by which adjacent regions are separated in the figure.

        xaxis_label_height
            The vertical location of the x-axis label, in figure coordinates. Should
            be between ``0.`` and ``1.``, though you are likely to want this to be
            slightly larger than ``0.`` typically.

        xaxis_ticks
            Specifies custom x-axis ticks for each region, overwriting the default
            ticks. Should be of the form: ``[(i, (a, b, ...)), (j, (c, d, ...)), ...]``
            where ``i`` and ``j`` are ints indicating the region under consideration,
            and ``a``-``d`` are floats indicating the tick values.

        contour_base
            The lowest level for the contour levels in the 2DJ spectrum plot.

        contour_nlevels
            The number of contour levels in the 2DJ spectrum plot.

        contour_factor
            The geometric scaling factor for adjacent contours in the 2DJ spectrum
            plot.

        contour_lw
            The linewidth of contours in the 2DJ spectrum plot.

        contour_color
            The color of the 2DJ spectrum plot.

        multiplet_colors
            **TODO**

        multiplet_lw
            Line width of multiplet plots

        multiplet_vertical_shift
            The vertical displacement of adjacent mutliplets, as a multiple of
            ``mutliplet_lw``. Set to ``0.`` if you want all mutliplets to lie on the
            same line.

        multiplet_show_center_freq
            If ``True``, lines are plotted on the 2DJ spectrum indicating the central
            frequency of each mutliplet.

        multiplet_show_45
            If ``True``, lines are plotted on the 2DJ spectrum indicating the 45° line
            along which peaks lie in ech multiplet.

        marker_size
            The size of markers indicating positions of peaks on the 2DJ contour plot.

        marker_shape
            The shape of markers indicating positions of peaks on the 2DJ contour plot.

        Returns
        -------
        fig
            The result figure

        axs
            A ``(2, N)`` NumPy array of the axes used for plotting.

        Notes
        -----
        **Figure coordinates** are a system in which ``0.`` indicates the left/bottom
        edge of the figure, and ``1.`` indicates the right/top.
        """
        sanity_check(
            (
                "indices", indices, sfuncs.check_int_list, (),
                {
                    "len_one_can_be_listless": True,
                    "min_value": -len(self._results),
                    "max_value": len(self._results) - 1,
                },
                True,
            ),
            (
                "multiplet_thold", multiplet_thold, sfuncs.check_float, (),
                {"greater_than_zero": True}, True,
            ),
            (
                "high_resolution_pts", high_resolution_pts, sfuncs.check_int, (),
                {"min_value": self.default_pts[1]}, True,
            ),
            (
                "ratio_1d_2d", ratio_1d_2d, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True},
            ),
            (
                "figure_size", figure_size, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True},
            ),
            (
                "axes_left", axes_left, sfuncs.check_float, (),
                {"min_value": 0., "max_value": 1.},
            ),
            (
                "axes_right", axes_right, sfuncs.check_float, (),
                {"min_value": 0., "max_value": 1.},
            ),
            (
                "axes_bottom", axes_bottom, sfuncs.check_float, (),
                {"min_value": 0., "max_value": 1.},
            ),
            (
                "axes_top", axes_top, sfuncs.check_float, (),
                {"min_value": 0., "max_value": 1.},
            ),
            (
                "axes_region_separation", axes_region_separation, sfuncs.check_float,
                (), {"min_value": 0., "max_value": 1.},
            ),
            (
                "xaxis_label_height", xaxis_label_height, sfuncs.check_float, (),
                {"min_value": 0., "max_value": 1.},
            ),
            (
                "contour_base", contour_base, sfuncs.check_float, (),
                {"min_value": 0.}, True,
            ),
            (
                "contour_nlevels", contour_nlevels, sfuncs.check_int, (),
                {"min_value": 1}, True,
            ),
            (
                "contour_factor", contour_factor, sfuncs.check_float, (),
                {"min_value": 1.}, True,
            ),
            ("contour_lw", contour_lw, sfuncs.check_float, (), {"min_value": 0.}),
            ("marker_size", marker_size, sfuncs.check_float, (), {"min_value": 0.}),
            (
                "multiplet_colors", multiplet_colors, sfuncs.check_oscillator_colors,
                (), {}, True,
            ),
            ("multiplet_lw", multiplet_lw, sfuncs.check_float, (), {"min_value": 0.}),
            (
                "multiplet_vertical_shift", multiplet_vertical_shift,
                sfuncs.check_float, (), {"min_value": 0.},
            ),
            (
                "multiplet_show_center_freq", multiplet_show_center_freq,
                sfuncs.check_bool,
            ),
            ("multiplet_show_45", multiplet_show_45, sfuncs.check_bool),
            ("denote_regions", denote_regions, sfuncs.check_bool),
        )
        # TODO
        # contour_color
        # linewidth
        # marker_shape: str = "o",

        # TODO use self._process_indices(indices)
        indices = (
            [i % len(self._results) for i in indices]
            if indices is not None
            else list(range(len(self._results)))
        )
        regions = sorted(
            [
                (i, result.get_region(unit="ppm")[1])
                for i, result in enumerate(self.get_results())
                if i in indices
            ],
            key=lambda x: x[1][0],
            reverse=True,
        )

        # Megre overlapping/bordering regions
        merge_indices = []
        merge_regions = []
        for idx, region in regions:
            assigned = False
            for i, reg in enumerate(merge_regions):
                if max(region) >= min(reg):
                    merge_regions[i] = (max(reg), min(region))
                    assigned = True
                elif min(region) >= max(reg):
                    merge_regions[i] = (max(region), min(reg))
                    assigned = True

                if assigned:
                    merge_indices[i].append(idx)
                    break

            if not assigned:
                merge_indices.append([idx])
                merge_regions.append(region)

        n_regions = len(merge_regions)

        fig, axs = plt.subplots(
            nrows=2,
            ncols=n_regions,
            gridspec_kw={
                "left": axes_left,
                "right": axes_right,
                "bottom": axes_bottom,
                "top": axes_top,
                "wspace": axes_region_separation,
                "hspace": 0.,
                "width_ratios": [r[0] - r[1] for r in merge_regions],
                "height_ratios": ratio_1d_2d,
            },
            figsize=figure_size,
        )
        if n_regions == 1:
            axs = axs.reshape(2, 1)

        if all(
            [isinstance(x, (float, int))
             for x in (contour_base, contour_nlevels, contour_factor)]
        ):
            contour_levels = [
                contour_base * contour_factor ** i
                for i in range(contour_nlevels)
            ]
        else:
            contour_levels = None

        if high_resolution_pts is None:
            high_resolution_pts = self.default_pts[1]

        expinfo_1d = self.direct_expinfo
        expinfo_1d_highres = copy.deepcopy(expinfo_1d)
        expinfo_1d_highres.default_pts = (high_resolution_pts,)
        full_shifts_1d, = expinfo_1d.get_shifts(unit="ppm")
        full_shifts_1d_highres, = expinfo_1d_highres.get_shifts(unit="ppm")
        full_shifts_2d_y, full_shifts_2d_x = self.get_shifts(unit="ppm")
        sfo = self.sfo[1]

        shifts_2d = []
        shifts_1d = []
        shifts_1d_highres = []
        spectra_2d = []
        spectra_1d = []
        neg_45_spectra = []
        f1_f2 = []
        center_freqs = []
        multiplet_spectra = []
        multiplet_indices = []

        for idx, region in zip(merge_indices, merge_regions):
            slice_ = slice(*expinfo_1d.convert([region], "ppm->idx")[0])
            highres_slice = slice(*expinfo_1d_highres.convert([region], "ppm->idx")[0])

            shifts_2d.append(
                (full_shifts_2d_x[:, slice_], full_shifts_2d_y[:, slice_])
            )
            shifts_1d.append(full_shifts_1d[slice_])
            shifts_1d_highres.append(full_shifts_1d_highres[highres_slice])

            spectra_2d.append(np.abs(self.spectrum).real[:, slice_])
            spectra_1d.append(self.spectrum_zero_t1.real[slice_])
            neg_45_spectra.append(
                sig.ft(
                    self.diagonal_signal(indices=idx, pts=high_resolution_pts)
                ).real[highres_slice]
            )

            params = self.get_params(indices=idx)
            multiplet_indices.append(
                list(
                    reversed(
                        self.predict_multiplets(indices=idx, thold=multiplet_thold)
                    )
                )
            )
            multiplet_params = [params[i] for i in multiplet_indices[-1]]
            f1_f2_region = []
            center_freq = []
            for multiplet_param in multiplet_params:
                f1, f2 = multiplet_param[:, [2, 3]].T
                center_freq.append(np.mean(f2 - f1) / sfo)
                f2 /= sfo
                f1_f2_region.append((f1, f2))

                multiplet = expinfo_1d.make_fid(
                    multiplet_param[:, [0, 1, 3, 5]],
                    pts=high_resolution_pts,
                )
                multiplet[0] *= 0.5
                multiplet_spectra.append(sig.ft(multiplet).real)

            f1_f2.append(f1_f2_region)
            center_freqs.append(center_freq)

        n_multiplets = len(multiplet_spectra)

        # Plot individual mutliplets
        for ax in axs[0]:
            colors = make_color_cycle(multiplet_colors, n_multiplets)
            ymax = -np.inf
            for i, mp_spectrum in enumerate(multiplet_spectra):
                color = next(colors)
                x = n_multiplets - 1 - i
                line = ax.plot(
                    full_shifts_1d_highres,
                    mp_spectrum + multiplet_vertical_shift * x,
                    color=color,
                    lw=multiplet_lw,
                    zorder=i,
                )[0]
                line_max = np.amax(line.get_ydata())
                if line_max > ymax:
                    ymax = line_max
                i += 1

        # Plot 1D spectrum
        spec_1d_low_pt = min([np.amin(spec) for spec in spectra_1d])
        shift = 1.03 * (ymax - spec_1d_low_pt)
        ymax = -np.inf
        for ax, shifts, spectrum in zip(axs[0], shifts_1d, spectra_1d):
            line = ax.plot(shifts, spectrum + shift, color="k")[0]
            line_max = np.amax(line.get_ydata())
            if line_max > ymax:
                ymax = line_max

        # Plot homodecoupled spectrum
        homo_spec_low_pt = min([np.amin(spec) for spec in neg_45_spectra])
        shift = 1.03 * (ymax - homo_spec_low_pt)
        for ax, shifts, spectrum in zip(axs[0], shifts_1d_highres, neg_45_spectra):
            ax.plot(shifts, spectrum + shift, color="k")

        # Plot 2DJ contour
        for ax, shifts, spectrum in zip(axs[1], shifts_2d, spectra_2d):
            ax.contour(
                *shifts,
                spectrum,
                colors=contour_color,
                linewidths=contour_lw,
                levels=contour_levels,
                zorder=0,
            )

        # Plot peak positions onto 2DJ
        colors = make_color_cycle(multiplet_colors, n_multiplets)
        for ax, f1f2, mp_idxs in zip(axs[1], f1_f2, multiplet_indices):
            for mp_f1f2, mp_idx in zip(f1f2, mp_idxs):
                color = next(colors)
                f1, f2 = mp_f1f2
                ax.scatter(
                    x=f2,
                    y=f1,
                    s=marker_size,
                    marker=marker_shape,
                    color=color,
                    zorder=100,
                )
                if label_peaks:
                    for f1_, f2_, idx in zip(f1, f2, mp_idx):
                        ax.text(
                            x=f2_,
                            y=f1_,
                            s=str(idx),
                            color=color,
                            fontsize=8,
                        )

        ylim1 = (shifts_2d[0][1][0, 0], shifts_2d[0][1][-1, 0])
        # Plot multiplet central frequencies
        if multiplet_show_center_freq:
            colors = make_color_cycle(multiplet_colors, n_multiplets)
            for ax, center_freq in zip(axs[1], center_freqs):
                for cf in center_freq:
                    color = next(colors)
                    ax.plot(
                        [cf, cf],
                        ylim1,
                        color=color,
                        lw=0.8,
                        zorder=2,
                    )

        # Plot 45 lines that multiplets lie along
        if multiplet_show_45:
            colors = make_color_cycle(multiplet_colors, n_multiplets)
            for ax, center_freq in zip(axs[1], center_freqs):
                for cf in center_freq:
                    color = next(colors)
                    ax.plot(
                        [cf + lim / sfo for lim in ylim1],
                        ylim1,
                        color=color,
                        lw=0.8,
                        zorder=2,
                        ls=":",
                    )

        # Configure axis appearance
        ylim0 = (
            min([ax.get_ylim()[0] for ax in axs[0]]),
            max([ax.get_ylim()[1] for ax in axs[0]]),
        )

        if denote_regions:
            for i, mi in enumerate(merge_indices):
                if len(mi) > 1:
                    locs_to_plot = [reg[1][0] for reg in regions if reg[0] in mi[1:]]
                    for loc in locs_to_plot:
                        for j, y in enumerate((ylim0, ylim1)):
                            axs[j, i].plot(
                                [loc, loc],
                                y,
                                color="#808080",
                                ls=":",
                            )

        axs[0, 0].spines["left"].set_zorder(1000)
        axs[0, -1].spines["right"].set_zorder(1000)
        for ax in axs[0]:
            ax.spines["bottom"].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylim(ylim0)

        for ax in axs[1]:
            ax.spines["top"].set_visible(False)
            ax.set_ylim(ylim1)

        for region, ax_col in zip(merge_regions, axs.T):
            for ax in ax_col:
                ax.set_xlim(*region)

        if n_regions > 1:
            for axs_col in axs[:, :-1]:
                for ax in axs_col:
                    ax.spines["right"].set_visible(False)
            for axs_col in axs[:, 1:]:
                for ax in axs_col:
                    ax.spines["left"].set_visible(False)

            break_kwargs = {
                "marker": [(-1, -3), (1, 3)],
                "markersize": 10,
                "linestyle": "none",
                "color": "k",
                "mec": "k",
                "mew": 1,
                "clip_on": False,
            }
            for ax in axs[0, :-1]:
                ax.plot([1], [1], transform=ax.transAxes, **break_kwargs)
            for ax in axs[0, 1:]:
                ax.plot([0], [1], transform=ax.transAxes, **break_kwargs)
            for ax in axs[1, :-1]:
                ax.plot([1], [0], transform=ax.transAxes, **break_kwargs)
            for ax in axs[1, 1:]:
                ax.plot([0], [0], transform=ax.transAxes, **break_kwargs)
                ax.set_yticks([])

        if xaxis_ticks is not None:
            for i, ticks in xaxis_ticks:
                axs[1, i].set_xticks(ticks)

        axs[1, 0].set_ylabel("Hz")
        fig.text(
            x=(axes_left + axes_right) / 2,
            y=xaxis_label_height,
            s=f"{self.latex_nuclei[1]} (ppm)",
            horizontalalignment="center",
        )

        return fig, axs

    def new_synthetic_from_simulation(self):
        pass

    def plot_multiplets(
        self,
        shifts_unit: str = "hz",
        merge_multiplet_oscillators: bool = True,
        thold: Optional[float] = None,
        pts: Optional[int] = None,
    ) -> mpl.figure.Figure:
        """Display a 1D spectrum of the multiplets predicted by the estimation routine.

        A figure of the first slice in T1 is plotted, along with oscillators produced
        in the estimation. Oscillators predicted to belong to the same multiplet
        structure are assigned the same colour.

        Parameters
        ----------
        shifts_unit
            ``"hz"`` or ``"ppm"``. The unit of the chemical shifts.

        merge_multiplet_oscillators
            If ``False``, each oscillator will be plotted separately. If ``True``,
            all oscillators corresponding to one multiplet will be summed prior to
            plotting.
        """
        sanity_check(
            (
                "shifts_unit", shifts_unit, sfuncs.check_frequency_unit,
                (self.hz_ppm_valid,),
            ),
            (
                "merge_multiplet_oscillators", merge_multiplet_oscillators,
                sfuncs.check_bool,
            ),
            ("thold", thold, sfuncs.check_float, (), {"greater_than_zero": True}, True),
        )

        fig = plt.figure()
        ax = fig.add_subplot()
        expinfo_1d = self.direct_expinfo
        shifts, = expinfo_1d.get_shifts(unit=shifts_unit)
        ax.plot(shifts, self.spectrum_zero_t1.real, color="k")

        params = self.get_params()
        multiplets = self.predict_multiplets(thold=thold)
        rainbow = itertools.cycle(
            ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
        )
        expinfo_1d = self.direct_expinfo
        for multiplet in multiplets:
            color = next(rainbow)
            if merge_multiplet_oscillators:
                mult = params[multiplet][:, [0, 1, 3, 5]]
                fid = expinfo_1d.make_fid(mult)
                fid[0] *= 0.5
                ax.plot(shifts, sig.ft(fid).real, color=color)

            else:
                for i in multiplet:
                    osc = np.expand_dims(params[i][[0, 1, 3, 5]], axis=0)
                    fid = expinfo_1d.make_fid(osc)
                    fid[0] *= 0.5
                    ax.plot(shifts, sig.ft(fid).real, color=color)

        ax.set_xlim(reversed(ax.get_xlim()))
        ax.set_xlabel(f"{self.latex_nuclei[1]} ({shifts_unit.replace('h', 'H')})")
        ax.set_yticks([])

        return fig

    def plot_contour(
        self,
        thold: Optional[float] = None,
        nlevels: Optional[int] = None,
        base: Optional[float] = None,
        factor: Optional[float] = None,
        shifts_unit: str = "hz",
    ) -> mpl.figure.Figure:
        sanity_check(
            ("thold", thold, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            ("nlevels", nlevels, sfuncs.check_int, (), {"min_value": 1}, True),
            ("base", base, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            (
                "factor", factor, sfuncs.check_float, (), {"greater_than_one": True},
                True,
            ),
            (
                "shifts_unit", shifts_unit, sfuncs.check_frequency_unit,
                (self.hz_ppm_valid,),
            ),
        )

        fig = plt.figure()
        ax = fig.add_subplot()
        shifts = self.get_shifts(unit="ppm")

        if any([x is None for x in (nlevels, base, factor)]):
            levels = None
        else:
            levels = [base * factor ** i for i in range(nlevels)]
            levels = [-x for x in reversed(levels)] + levels

        ax.contour(
            shifts[1].T, shifts[0].T, np.abs(self.spectrum).T, levels=levels,
            cmap="coolwarm",
        )

        params = self.get_params(funit=shifts_unit)
        peaks_x = params[:, 3]
        peaks_y = params[:, 2]
        multiplets = self.predict_multiplets(thold)
        rainbow = itertools.cycle(
            ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
        )
        for multiplet in multiplets:
            color = next(rainbow)
            for i in multiplet:
                ax.scatter(
                    peaks_x[i], peaks_y[i], marker="x", color=color, zorder=100,
                )

        ax.set_xlim(reversed(ax.get_xlim()))
        ax.set_xlabel(f"{self.latex_nuclei[1]} ({shifts_unit.replace('h', 'H')})")
        ax.set_ylim(reversed(ax.get_ylim()))
        ax.set_ylabel("Hz")

        return fig


class ContourApp(tk.Tk):
    """Tk app for viewing 2D spectra as contour plots."""

    def __init__(self, data: np.ndarray, expinfo) -> None:
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.quit)
        self.shifts = list(reversed(
            [s.T for s in expinfo.get_shifts(data.shape, unit="ppm")]
        ))
        nuclei = expinfo.nuclei
        units = ["ppm" if sfo is not None else "Hz" for sfo in expinfo.sfo]
        self.f1_label, self.f2_label = [
            f"{nuc} ({unit})" if nuc is not None
            else unit
            for nuc, unit in zip(nuclei, units)
        ]

        self.data = data.T.real

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.fig = plt.figure(dpi=160, frameon=True)
        self._color_fig_frame()

        self.ax = self.fig.add_axes([0.1, 0.1, 0.87, 0.87])
        self.ax.set_xlim(self.shifts[0][0][0], self.shifts[0][-1][0])
        self.ax.set_ylim(self.shifts[1][0][0], self.shifts[1][0][-1])

        self.cmap = tk.StringVar(self, "bwr")
        self.nlevels = 10
        self.factor = 1.3
        self.base = np.amax(np.abs(self.data)) / 10
        self.update_plot()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(
            row=0,
            column=0,
            padx=10,
            pady=10,
            sticky="nsew",
        )

        self.toolbar = NavigationToolbar2Tk(
            self.canvas,
            self,
            pack_toolbar=False,
        )
        self.toolbar.grid(row=1, column=0, pady=(0, 10), sticky="w")

        self.widget_frame = tk.Frame(self)
        self._add_widgets()
        self.widget_frame.grid(
            row=2,
            column=0,
            padx=10,
            pady=(0, 10),
            sticky="nsew",
        )
        self.close_button = tk.Button(
            self, text="Close", command=self.quit,
        )
        self.close_button.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="w")

    def _color_fig_frame(self) -> None:
        r, g, b = [x >> 8 for x in self.winfo_rgb(self.cget("bg"))]
        color = f"#{r:02x}{g:02x}{b:02x}"
        if not re.match(r"^#[0-9a-f]{6}$", color):
            color = "#d9d9d9"

        self.fig.patch.set_facecolor(color)

    def _add_widgets(self) -> None:
        # Colormap selection
        self.cmap_label = tk.Label(self.widget_frame, text="Colormap:")
        self.cmap_label.grid(row=0, column=0, padx=(0, 10))
        self.cmap_widget = tk.OptionMenu(
            self.widget_frame,
            self.cmap,
            self.cmap.get(),
            "PiYG",
            "PRGn",
            "BrBG",
            "PuOr",
            "RdGy",
            "RdBu",
            "RdYlBu",
            "RdYlGn",
            "Spectral",
            "coolwarm",
            "bwr",
            "seismic",
            command=lambda x: self.update_plot(),
        )
        self.cmap_widget.grid(row=0, column=1)

        # Number of contour levels
        self.nlevels_label = tk.Label(self.widget_frame, text="levels")
        self.nlevels_label.grid(row=0, column=2, padx=(0, 10))
        self.nlevels_box = MyEntry(
            self.widget_frame,
            return_command=self.change_levels,
            return_args=("nlevels",),
        )
        self.nlevels_box.insert(0, str(self.nlevels))
        self.nlevels_box.grid(row=0, column=3)

        # Base contour level
        self.base_label = tk.Label(self.widget_frame, text="base")
        self.base_label.grid(row=0, column=4, padx=(0, 10))
        self.base_box = MyEntry(
            self.widget_frame,
            return_command=self.change_levels,
            return_args=("base",),
        )
        self.base_box.insert(0, f"{self.base:.2f}")
        self.base_box.grid(row=0, column=5)

        # Contour level scaling factor
        self.factor_label = tk.Label(self.widget_frame, text="factor")
        self.factor_label.grid(row=0, column=6, padx=(0, 10))
        self.factor_box = MyEntry(
            self.widget_frame,
            return_command=self.change_levels,
            return_args=("factor",),
        )
        self.factor_box.insert(0, f"{self.factor:.2f}")
        self.factor_box.grid(row=0, column=7)

    def change_levels(self, var: str) -> None:
        input_ = self.__dict__[f"{var}_box"].get()
        try:
            if var == "nlevels":
                value = int(input_)
                if value <= 0.:
                    raise ValueError
            else:
                value = float(input_)
                if (
                    value <= 1. and var == "factor" or
                    value <= 0. and var == "base"
                ):
                    raise ValueError

            self.__dict__[var] = value
            self.update_plot()

        except ValueError:
            box = self.__dict__[f"{var}_box"]
            box.delete(0, "end")
            box.insert(0, str(self.__dict__[var]))

    def make_levels(self) -> Iterable[float]:
        levels = [self.base * self.factor ** i
                  for i in range(self.nlevels)]
        return [-x for x in reversed(levels)] + levels

    def update_plot(self) -> None:
        levels = self.make_levels()
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()
        self.ax.contour(
            *self.shifts, self.data, cmap=self.cmap.get(), levels=levels,
        )
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel(self.f2_label)
        self.ax.set_ylabel(self.f1_label)
        self.fig.canvas.draw_idle()
