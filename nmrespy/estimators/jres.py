# jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 11 May 2022 16:16:19 BST

from __future__ import annotations
import copy
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from nmr_sims.experiments.jres import JresSimulation
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem

from nmrespy import ExpInfo, sig
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)
from nmrespy.estimators import logger, Estimator, Result
from nmrespy.freqfilter import Filter
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import NonlinearProgramming


class Estimator2DJ(Estimator):
    def __init__(
        self, data: np.ndarray, expinfo: ExpInfo, datapath: Optional[Path] = None,
    ) -> None:
        super().__init__(data, expinfo, datapath)

    @classmethod
    def new_bruker(cls):
        pass

    @classmethod
    def new_synthetic_from_simulation(
        cls,
        spin_system: SpinSystem,
        sweep_widths: Tuple[float, float],
        offset: float,
        pts: Tuple[int, int],
        channel: Union[str, Nucleus] = "1H",
        f2_unit: str = "ppm",
        snr: Optional[float] = 30.0,
        lb: Optional[Tuple[float, float]] = None,
    ) -> Estimator2DJ:
        """Generate an estimator with data derived from a J-resolved experiment
        simulation.

        Simulations are performed using
        `nmr_sims.experiments.jres.JresEstimator
        <https://foroozandehgroup.github.io/nmr_sims/content/references/experiments/
        pa.html#nmr_sims.experiments.jres.JresEstimator>`_.

        Parameters
        ----------
        spin_system
            Specification of the spin system to run simulations on. `See here
            <https://foroozandehgroup.github.io/nmr_sims/content/references/
            spin_system.html#nmr_sims.spin_system.SpinSystem.__init__>`_
            for more details.

        sweep_widths
            The sweep width in each dimension. The first element, corresponding
            to F1, should be in Hz. The second element, corresponding to F2,
            should be expressed in the unit which corresponds to ``f2_unit``.

        offset
            The transmitter offset. The value's unit should correspond with
            ``f2_unit``.

        pts
            The number of points sampled in each dimension.

        channel
            Nucleus targeted in the experiment simulation. ¹H is set as the default.
            `See here <https://foroozandehgroup.github.io/nmr_sims/content/
            references/nuclei.html>`__ for more information.

        f2_unit
            The unit that the sweep width and transmitter offset in F2 are given in.
            Should be either ``"ppm"`` (default) or ``"hz"``.

        snr
            The signal-to-noise ratio of the resulting signal, in decibels. ``None``
            produces a noiseless signal.

        lb
            The damping (line-broadening) factor applied to the simulated FID.
            By default, this will be set to ensure that the final point in each
            dimension in scaled to be 1/1000 of it's un-damped value.
        """
        sanity_check(
            ("spin_system", spin_system, sfuncs.check_spin_system),
            (
                "sweep_widths", sweep_widths, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True},
            ),
            ("offset", offset, sfuncs.check_float),
            (
                "pts", pts, sfuncs.check_int_list, (),
                {"length": 2, "must_be_positive": True},
            ),
            ("channel", channel, sfuncs.check_nmrsims_nucleus),
            ("f2_unit", f2_unit, sfuncs.check_frequency_unit, (True,)),
            ("snr", snr, sfuncs.check_float, (), {}, True),
            (
                "lb", lb, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True}, True,
            ),
        )

        sweep_widths = [f"{sweep_widths[0]}hz", f"{sweep_widths[1]}{f2_unit}"]
        offset = f"{offset}{f2_unit}"

        sim = JresSimulation(spin_system, pts, sweep_widths, offset, channel)
        sim.simulate()
        _, data, _ = sim.fid(lb=lb)

        if snr is not None:
            data += sig._make_noise(data, snr)

        expinfo = ExpInfo(
            dim=2,
            sw=sim.sweep_widths,
            offset=[0.0, sim.offsets[0]],
            sfo=[None, sim.sfo[0]],
            nuclei=[None, sim.channels[0].name],
            default_pts=data.shape,
            fn_mode="QF",
        )
        return cls(data, expinfo, None)

    def view_data(
        self,
        domain: str = "freq",
        components: str = "real",
        freq_unit: str = "hz",
        abs_: bool = False,
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

        abs_
            Whether or not to display frequency-domain data in absolute-value mode.
        """
        sanity_check(
            ("domain", domain, sfuncs.check_one_of, ("freq", "time")),
            ("components", components, sfuncs.check_one_of, ("real", "imag", "both")),
            ("freq_unit", freq_unit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
            ("abs_", abs_, sfuncs.check_bool),
        )

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        z = copy.deepcopy(self._data)

        if domain == "freq":
            x, y = self.get_shifts(unit=freq_unit)
            z[0, 0] /= 2
            z = sig.ft(z)

            if abs_:
                z = np.abs(z)

            if self.nuclei is None:
                pass
            else:
                freq_units = ("Hz", freq_unit.replace("h", "H"))
                nuclei = [
                    nuc if nuc is not None else "$\\omega$"
                    for nuc in self.latex_nuclei
                ]
                xlabel, ylabel = [
                    f"{nuc} ({fu})" for nuc, fu in zip(nuclei, freq_units)

                ]
        elif domain == "time":
            x, y = self.get_timepoints()
            xlabel, ylabel = [f"$t_{i}$ (s)" for i in range(1, 3)]

        if components in ("real", "both"):
            ax.plot_wireframe(
                x, y, z.real, color="k", lw=0.2, rstride=1, cstride=1,
            )
        if components in ("imag", "both"):
            ax.plot_wireframe(
                x, y, z.imag, color="#808080", lw=0.2, rstride=1, cstride=1,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(reversed(ax.get_xlim()))
        ax.set_ylim(reversed(ax.get_ylim()))
        ax.set_zticks([])

        plt.show()

    @logger
    def estimate(
        self,
        region: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
        noise_region: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
        region_unit: str = "ppm",
        initial_guess: Optional[Union[np.ndarray, int]] = None,
        method: str = "gauss-newton",
        phase_variance: bool = False,
        max_iterations: Optional[int] = None,
        fprint: bool = True,
        _log: bool = True,
    ):
        r"""Estimate a specified region in F2 of the signal.

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

        # The plan of action:
        # --> Derive filtered signals (both cut and uncut)
        # --> Run the MDL followed by MPM for an initial guess on cut signal
        # --> Run Optimiser on cut signal
        # --> Run Optimiser on uncut signal
        filt = Filter(
            self._data,
            ExpInfo(2, self.sw(), self.offset(), self.sfo),
            (None, region),
            (None, noise_region),
            region_unit=region_unit,
            twodim_dtype="jres",
        )

        cut_signal, cut_expinfo = filt.get_filtered_fid()
        uncut_signal, uncut_expinfo = filt.get_filtered_fid(cut_ratio=None)
        region = filt.get_region()
        noise_region = filt.get_noise_region()

        if isinstance(initial_guess, np.ndarray):
            x0 = initial_guess
        else:
            oscillators = initial_guess if isinstance(initial_guess, int) else 0
            x0 = MatrixPencil(
                cut_expinfo,
                cut_signal,
                oscillators=oscillators,
                fprint=fprint,
            ).get_params()

        cut_result = NonlinearProgramming(
            cut_expinfo,
            cut_signal,
            x0,
            phase_variance=phase_variance,
            method=method,
            max_iterations=max_iterations,
            fprint=fprint,
        ).get_params()

        final_result = NonlinearProgramming(
            uncut_expinfo,
            uncut_signal,
            cut_result,
            phase_variance=phase_variance,
            method=method,
            max_iterations=max_iterations,
            fprint=fprint,
        )

        self._results.append(
            Result(
                final_result.get_params(),
                final_result.get_errors(),
                region,
                noise_region,
                self.sfo,
            )
        )

    def write_results(self):
        pass

    def plot_results(self):
        pass

    def negative_45_signal(
        self,
        indices: Optional[Iterable[int]] = None,
        pts: Optional[int] = None,
    ) -> np.ndarray:
        sanity_check(
            (
                "indices", indices, sfuncs.check_ints_less_than_n,
                (len(self._results),), True,
            ),
            ("pts", pts, sfuncs.check_positive_int, (), True),
        )

        params = self.get_results(indices)
        offset = self._expinfo.offset[1]
        tp = self._expinfo.get_timepoints(meshgrid=False)[1]

        signal = (
            np.exp(  # Z1
                np.outer(
                    tp,
                    -2j * np.pi * params[:, 2] - params[:, 4],
                )
            ) *
            np.exp(  # Z2
                np.outer(
                    tp,
                    2j * np.pi * (params[:, 3] - offset) - params[:, 5],
                )
            )
        ) @ (params[:, 0] * np.exp(1j * params[:, 1]))  # alpha

        return signal

    def phase_data(self):
        pass

    def plot_result(self):
        pass

    def write_result(self):
        pass
