# jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 30 Mar 2022 14:32:52 BST

from __future__ import annotations
import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from nmr_sims.experiments.jres import JresSimulation
from nmr_sims.nuclei import Nucleus
from nmr_sims.spin_system import SpinSystem

from nmrespy import ExpInfo, sig
from nmrespy.estimators import Estimator
from nmrespy.freqfilter import Filter
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import NonlinearProgramming
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)


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
            Specification of the spin system to run simulations on.
            `See here <https://foroozandehgroup.github.io/nmr_sims/content/
            references/spin_system.html#nmr_sims.spin_system.SpinSystem.__init__>`_
            for more details.

        sweep_widths
            The sweep width in each dimension. The first element, corresponding to
            F1, should be in Hz. The second element, corresponding to F2, should have
            be expressed in the unit which corresponding to ``f2_unit``.

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
            )
            ("channel", channel, sfuncs.check_nmrsims_nucleus),
            ("f2_unit", f2_unit, sfuncs.check_frequency_unit, (True,)),
            ("snr", snr, sfuncs.check_float, (), {}, True),
        )
        sweep_widths = [f"{sweep_widths[0]}hz", f"{sweep_widths[1]}{f2_unit}"]
        offset = f"{offset}{f2_unit}"
        sim = JresSimulation(
            spin_system, pts, sweep_widths, offset, channel,
        )
        sim.simulate()
        _, data = sim.fid
        data = data.T
        if snr is not None:
            data += sig._make_noise(data, snr)

        expinfo = ExpInfo(
            dim=2,
            sw=sim.sweep_widths,
            offset=[0.0, sim.offsets[0]],
            sfo=[None, sim.sfo[0]],
            nuclei=[None, sim.channels[0].name],
            default_pts=data.shape,
        )
        return cls(data, expinfo, None)

    def estimate(
        self,
        region: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
        noise_region: Optional[Tuple[Union[int, float], Union[int, float]]] = None,
        region_unit: str = "ppm",
        initial_guess: Optional[Union[np.ndarray, int]] = None,
        hessian: str = "gauss-newton",
        phase_variance: bool = False,
        max_iterations: Optional[int] = None,
    ):
        """Estimate a specified region in F2 of the signal.

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
            ``[left, right]`` where ``left`` and ``right`` are the left and right
            bounds of the region of interest. If ``None``, the full signal will be
            considered, though for sufficently large and complex signals it is
            probable that poor and slow performance will be achieved.

        noise_region
            If ``region`` is not ``None``, this must be of the form ``[left, right]``
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
            region_check = sfuncs.check_jres_region_hz
        elif region_unit == "ppm":
            region_check = sfuncs.check_jres_region_ppm

        sanity_check(
            ("region", region, region_check, (self._expinfo,), True),
            ("noise_region", noise_region, region_check, (self._expinfo,), True),
        )

        filt = Filter(
            self._data,
            self._expinfo,
            [None, region],
            [None, noise_region],
            region_unit=region_unit,
            twodim_dtype="jres",
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
        result, errors = nlp_result.get_result("hz"), nlp_result.get_errors()

        self._results.append(
            {
                "timestamp": datetime.datetime.now(),
                "filter": filt,
                "result": result,
                "errors": errors,
            }
        )

    def make_fid(
        self,
        pts: Optional[Iterable[int]] = None,
        indices: Optional[Iterable[int]] = None,
    ) -> np.ndarray:
        """Construct a synthetic FID using estimation results.

        Parameters
        ----------
        pts
            The number of points to construct the FID with in each dimesnion.
            If ``None``, the number of points used will match the data.

        indices
            The indices of results to return. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be returned.
        """
        sanity_check(
            ("pts", pts, sfuncs.check_points, (self.dim,), True),
            (
                "indices", indices, sfuncs.check_ints_less_than_n,
                (len(self._results),), True,
            ),
        )

        if pts is None:
            pts = self._expinfo.default_pts

        return sig.make_fid(self.get_results(indices), self._expinfo, pts)[0]

    def get_results(
        self,
        indices: Optional[Iterable[int]],
        merge: bool = True,
    ) -> np.ndarray:
        sanity_check(
            (
                "indices", indices, sfuncs.check_ints_less_than_n,
                (len(self._results),), True,
            ),
            ("merge", merge, sfuncs.check_bool),
        )

        if indices is None:
            indices = range(len(self._results))

        results = [self._results[i]["result"] for i in indices]

        if merge:
            sizes = [result.shape[0] for result in results]
            params = np.zeros((sum(sizes), 2 * self.dim + 2))

            idx = 0
            for size, result in zip(sizes, results):
                params[idx : idx + size] = result
                idx += size

        else:
            params = results

        return params

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
        ) @ (params[:, 0] * np.exp(1j * params[:, 1])) # alpha

        return signal


if __name__ == "__main__":
    spin_system = SpinSystem({
        1: {
            "shift": 5.5,
            "couplings": {
                2: 4.6,
                3: 4.6,
            }
        },
        2: {
            "shift": 8.2,
        },
        3: {
            "shift": 8.2,
        },
    })

    # Experiment parameters
    channel = "1H"
    sweep_widths = [50., 10.]
    points = [32, 512]
    offset = 5.

    estimator = Estimator2DJ.new_synthetic_from_simulation(
        spin_system, sweep_widths, offset, points, channel=channel, f2_unit="ppm",
        snr=10.
    )
    estimator.estimate([6.0, 5.0], [1.0, 0.5], initial_guess=3)
    estimator.estimate([8.7, 7.7], [1.0, 0.5], initial_guess=2)
    fid = estimator.make_fid()
    fid[0, 0] /= 2
    model_spectrum = np.abs(sig.ft(fid)).real
    real_spectrum = np.abs(sig.ft(estimator._data)).real
    shiftsf1, shiftsf2 = estimator._expinfo.get_shifts(unit="ppm")

    import matplotlib as mpl
    mpl.use("tkAgg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(shiftsf1 + 5, shiftsf2 + 0.2, model_spectrum / 2, lw=0.2, cstride=1, rstride=1)
    ax.plot_wireframe(shiftsf1, shiftsf2, real_spectrum, lw=0.2, color="b", cstride=1, rstride=1)
    ax.set_xlim(reversed(ax.get_xlim()))
    ax.set_ylim(reversed(ax.get_ylim()))
    plt.show()

    neg45 = estimator.negative_45_signal(pts=32000)
    neg45[0] /= 2
    shifts = estimator._expinfo.get_shifts(unit="ppm", meshgrid=False)[1]
    spectrum = sig.ft(neg45)

    import matplotlib as mpl
    mpl.use("tkAgg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(shifts, spectrum)
    ax.set_xlim(reversed(ax.get_xlim()))
    plt.show()
