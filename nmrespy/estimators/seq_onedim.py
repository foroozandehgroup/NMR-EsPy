# seq_onedim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Thu 18 May 2023 10:56:55 BST

from __future__ import annotations
import copy
import io
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axis3d import Axis
import numpy as np

import nmrespy as ne
from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._files import check_existent_dir
from nmrespy._misc import proc_kwargs_dict
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy.estimators import logger
from nmrespy.estimators.onedim import Estimator1D
from nmrespy.load import load_bruker
from nmrespy.nlp.optimisers import trust_ncg
from nmrespy.plot import make_color_cycle


if USE_COLORAMA:
    import colorama
    colorama.init()


# Patch Axes3D to prevent annoying padding
# https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
if not hasattr(Axis, "_get_coord_info_old"):
    def _get_coord_info_new(self, renderer):
        mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
        mins += deltas / 4
        maxs -= deltas / 4
        return mins, maxs, centers, deltas, tc, highs
    Axis._get_coord_info_old = Axis._get_coord_info
    Axis._get_coord_info = _get_coord_info_new


class EstimatorSeq1D(Estimator1D):

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        increments: Optional[np.ndarray],
        datapath: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        data
            The data associated with the binary file in `path`.

        expinfo
            Experiment information.

        increments
            The values of increments used to acquire the FIDs. Examples would
            include:

            * Delay times in an inversion recovery experiment.
            * Gradient strengths in a diffusion experiment.

        datapath
            The path to the directory containing the NMR data.
        """
        sanity_check(
            (
                "increments", increments, sfuncs.check_ndarray, (),
                {"dim": 1, "shape": [(0, data.shape[0])]}, True,
            ),
        )
        super().__init__(data[0], expinfo, datapath=datapath)
        self._estimators = [ne.Estimator1D(fid, expinfo) for fid in data]
        self.increments = increments

    # TODO: More useful!
    def __str__(self):
        return self.__class__

    @property
    def data(self) -> np.ndarray:
        data = np.zeros((self.n_increments, self.default_pts[0]), dtype="complex")
        for i, estimator in enumerate(self._estimators):
            data[i] = estimator.data

        return data

    @property
    def n_increments(self) -> int:
        return len(self.increments)

    @property
    def increment_label(self) -> str:
        return getattr(self, "_increment_label", "")

    @property
    def fit_labels(self) -> Optional[str]:
        return getattr(self, "_fit_labels", None)

    @property
    def fit_units(self) -> Optional[str]:
        return getattr(self, "_fit_units", None)

    @classmethod
    def _new_bruker_pre(
        cls,
        directory: Union[str, Path],
        increment_file: Optional[str],
        convdta: bool,
    ) -> Tuple[np.ndarray, ne.ExpInfo, np.ndarray, Path]:
        sanity_check(
            ("directory", directory, check_existent_dir),
            ("convdta", convdta, sfuncs.check_bool),
            ("increment_file", increment_file, sfuncs.check_str, (), {}, True),
        )

        directory = Path(directory).expanduser()
        data, expinfo_2d = load_bruker(directory)

        if data.ndim != 2:
            raise ValueError(f"{RED}Data dimension should be 2.{END}")

        float_list_files = cls._check_float_list_file(directory)

        if not float_list_files:
            raise ValueError(
                f"{RED}Could not find a suitable delay file in directory: "
                f"{directory}.{END}"
            )

        elif len(float_list_files) == 1:
            name, floats = next(iter(float_list_files.items()))
            if increment_file is None or name == increment_file:
                increments = floats
            else:
                raise ValueError(
                    f"{RED}Increment file ({increment_file}) either doesn't exist or "
                    f"is of the incorrect format. Note that \"{name}\" is of the "
                    f"correct format to be a delay file.{END}"
                )

        elif len(float_list_files) > 1:
            if increment_file is None:
                files = ", ".join(float_list_files.keys())
                raise ValueError(
                    f"{RED}Multiple files were found that are of the correct format "
                    f"to be a delay file:\n{files}\nPlease specifiy the correct file "
                    f"with the `gradient_file` argument{END}."
                )
            elif increment_file in float_list_files.keys():
                increments = float_list_files[increment_file]
            else:
                raise ValueError(
                    f"{RED}Increment file ({increment_file}) either doesn't exist, or "
                    f"it does not correspond to a correctly formatted file.{END}"
                )

        if convdta:
            grpdly = expinfo_2d.parameters["acqus"]["GRPDLY"]
            data = ne.sig.convdta(data, grpdly)

        expinfo = ne.ExpInfo(
            dim=1,
            sw=expinfo_2d.sw()[-1],
            offset=expinfo_2d.offset()[-1],
            sfo=expinfo_2d.sfo[-1],
            nuclei=expinfo_2d.nuclei[-1],
            default_pts=data.shape[-1],
            parameters=expinfo_2d.parameters
        )

        return data, expinfo, increments, directory

    @staticmethod
    def _check_float_list_file(
        directory: Path,
    ) -> Optional[Dict[str, np.ndarray]]:
        ignore = ["acqus", "acqu2s", "ser", "pulseprogram"]
        files = [directory / fname for fname in os.listdir(directory)]
        files = [f for f in files if f.is_file() and f.name not in ignore]

        float_list_files = {}

        for f in files:
            try:
                with open(f, "r") as fh:
                    vals = np.array([float(x) for x in fh.readlines()])
                float_list_files[f.name] = vals
            except ValueError:
                pass

        return float_list_files if float_list_files else None

    def exp_apodisation(self, k: float) -> None:
        for estimator in self._estimators:
            estimator.exp_apodisation(k)

    def phase_data(self, p0: float = 0., p1: float = 0., pivot: int = 0.) -> None:
        for estimator in self._estimators:
            estimator.phase_data(p0, p1, pivot)

    def manual_phase_data(
        self,
        max_p1: float = 10 * np.pi,
    ) -> Tuple[float, float]:
        p0, p1 = self._estimators[0].manual_phase_data(max_p1)
        for estimator in self._estimators[1:]:
            estimator.phase_data(p0=p0, p1=p1)

    def baseline_correction(self, min_length: int = 50) -> None:
        for estimator in self._estimators:
            estimator.baseline_correction(min_length)

    def view_data(
        self,
        domain: str = "freq",
        components: str = "real",
        freq_unit: str = "hz",
    ) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        zs = self.data

        if domain == "freq":
            x, = self.get_shifts(unit=freq_unit)
            xlabel, = self._axis_freq_labels(freq_unit)
            zs[:, 0] *= 0.5
            zs = ne.sig.ft(zs, axes=-1)

        elif domain == "time":
            x, = self.get_timepoints()
            xlabel = "$t$ (s)"

        ys = self.increments

        for (y, z) in zip(ys, zs):
            y_arr = np.full(z.shape, y)
            if components in ("real", "both"):
                ax.plot(x, y_arr, z.real, color="k")
            if components in ("imag", "both"):
                ax.plot(x, y_arr, z.imag, color="#808080")

        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.increment_label)
        ax.set_zticks([])

        plt.show()

    def view_data_increment(
        self,
        increment: int = 0,
        domain: str = "freq",
        components: str = "real",
        freq_unit: str = "hz",
    ) -> None:
        """View the data (FID or spectrum) of a paricular increment in the dataset
        with an interacitve matplotlib plot.

        Parameters
        ----------
        increment
            The increment to view. The data displayed is ``self.data[increment]``.

        domain
            Must be ``"freq"`` or ``"time"``.

        components
            Must be ``"real"``, ``"imag"``, or ``"both"``.

        freq_unit
            Must be ``"hz"`` or ``"ppm"``. If ``domain`` is ``"freq"``, this
            determines which unit to set chemical shifts to.
        """
        sanity_check(
            (
                "increment", increment, sfuncs.check_int, (),
                {"min_value": 0, "max_value": self.increments - 1},
            ),
        )
        self._estimators[increment].view_data(
            domain=domain,
            components=components,
            freq_unit=freq_unit,
        )

    def estimate(self, **kwargs) -> None:
        if type(self).__name__ == "EstimatorInvRec":
            old_data0 = self._estimators[0].data
            new_data0 = ne.sig.ift(-1 * ne.sig.ft(old_data0))
            self._estimators[0]._data = new_data0

        self._estimators[0].estimate(**kwargs)
        x0 = self._estimators[0].get_params(indices=[-1])

        if type(self).__name__ == "EstimatorInvRec":
            self._estimators[0]._data = old_data0
            x0[:, 0] *= -1
            self._estimators[0]._results[-1].params = x0

        self._optimise_amps(x0, **kwargs)

    def _optimise_amps(self, x0: np.ndarray, **kwargs) -> None:
        kwargs["mode"] = "a"
        kwargs["negative_amps"] = "ignore"
        for estimator in self._estimators[1:]:
            sys.stdout = io.StringIO(str(os.devnull))
            estimator.estimate(
                initial_guess=x0,
                **kwargs,
            )
            x0 = estimator.get_params(indices=[-1])
            sys.stdout = sys.__stdout__

    def edit_result(self, index=-1, **kwargs) -> None:
        kwargs["negative_amps"] = "ignore"
        sanity_check(self._index_check(index))
        index, = self._process_indices([index])
        self._estimators[0].edit_result(index=index, **kwargs)

        result = self._estimators[0]._results[index]
        x0 = result.params
        region = result.region
        noise_region = result.noise_region
        kwargs["region"] = region
        kwargs["noise_region"] = noise_region
        self._optimise_amps(x0, **kwargs)
        for estimator in self._estimators[1:]:
            del estimator._results[index]
            estimator._results.insert(index, estimator._results.pop(-1))

    def _fit(
        self,
        indices: Optional[Iterable[int]] = None,
        oscs: Optional[Iterable[int]] = None,
    ) -> Iterable[np.ndarray]:
        sanity_check(self._indices_check(indices))
        indices = self._process_indices(indices)

        n_oscs = self.get_params(indices=indices)[0].shape[0]
        sanity_check(self._oscs_check(oscs, n_oscs))
        oscs = self._process_oscs(oscs, n_oscs)

        results = []
        errors = []
        for amps in self.amplitudes(indices, oscs):
            norm = np.linalg.norm(amps)
            amps /= norm
            args = (amps, self.increments)
            # get_x0 is defined within child classes.
            x0 = self.get_x0(*args)

            # Suppress output from trust_ncg function
            sys.stdout = io.StringIO(str(os.devnull))

            nlp_result = trust_ncg(
                x0=x0,
                args=args,
                function_factory=self.function_factory,
                output_mode=None,
            )

            # Switch output back on
            sys.stdout = sys.__stdout__

            x = nlp_result.x
            x[0] *= norm
            errs = nlp_result.errors
            errs[0] *= norm

            results.append(x)
            errors.append(errs / np.sqrt(len(self.increments) - 1))

        return results, errors

    def amplitudes(
        self,
        indices: Optional[Iterable[int]] = None,
        oscs: Optional[Iterable[int]] = None,
    ) -> Iterable[np.ndarray]:
        """Get the amplitudes associated with specified oscillators.

        Parameters
        ----------
        indices
            See :ref:`INDICES`.

        oscs
            Oscillators to get amplitudes for. By default (``None``) all oscillators
            are considered.
        """
        sanity_check(
            self._indices_check(indices),
        )
        params = self.get_params(indices)
        n_oscs = params[0].shape[0]
        sanity_check(
            self._oscs_check(oscs, n_oscs),
        )
        oscs = self._process_oscs(oscs, n_oscs)
        amps = np.array([p[oscs, 0] for p in params]).T
        return amps

    def plot_fit_single_oscillator(
        self,
        osc: int = 0,
        index: int = -1,
        fit_increments: int = 100,
        fit_line_kwargs: Dict = None,
        scatter_kwargs: Dict = None,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Plot the fit across increments for a particular oscillator.

        Parameters
        ----------
        osc
            Index for the oscillator of interest.

        index
            See :ref:`INDEX`. By default (``-1``), the last acquired result is used.

        fit_increments
            The number of points in the fit line.

        fit_line_kwargs
            Keyword arguments for the fit line. Use this to specify features such
            as color, linewidth etc. All keys should be valid arguments for
            `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.

        scatter_kwargs
            Keyword arguments for the scatter plot. All keys should be valid arguments
            for
            `matplotlib.axes.Axes.scatter
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html>`_.

        kwargs
            Keyword arguments for
            `matplotlib.pyplot.figure
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_

        Returns
        -------
        fig
            The figure generated

        ax
            The axes assigned to the figure.
        """
        self._check_results_exist()
        sanity_check(
            self._index_check(index),
            (
                "fit_increments", fit_increments, sfuncs.check_int, (),
                {"min_value": len(self.increments)},
            ),
        )
        n_oscs = self.get_params(indices=[index])[0].shape[0]
        sanity_check(
            (
                "osc", osc, sfuncs.check_int, (),
                {"min_value": 0, "max_value": n_oscs - 1},
            ),
        )

        fit_line_kwargs = proc_kwargs_dict(
            fit_line_kwargs,
            default={"color": "#808080"},
        )
        scatter_kwargs = proc_kwargs_dict(
            scatter_kwargs,
            default={"color": "k", "marker": "x"},
        )

        amplitudes = self.amplitudes([index], [osc])
        params, errors = self.fit([index], [osc])
        params = params[0]
        errors = errors[0]

        fig, ax = plt.subplots(ncols=1, nrows=1, **kwargs)
        x = np.linspace(
            np.amin(self.increments),
            np.amax(self.increments),
            fit_increments,
        )
        ax.plot(x, self.model(*params, x), **fit_line_kwargs)
        ax.scatter(self.increments, amplitudes, **scatter_kwargs)
        ax.set_xlabel(self.increment_label)
        ax.set_ylabel("$a$")

        text = "\n".join(
            [
                f"$p_{{{i}}}$ $= {para:.4g} \\pm {err:.4g}$U{i}"
                for i, (para, err) in enumerate(zip(params, errors), start=1)
            ]
        )
        if self.fit_labels is not None:
            for i, (flab, ulab) in enumerate(
                zip(self.fit_labels, self.fit_units),
                start=1,
            ):
                text = text.replace(f"$p_{{{i}}}$", flab)
                text = text.replace(f"U{i}", f" {ulab}" if ulab != "" else "")
        else:
            i = 0
            while True:
                if (to_rm := f"U{i}") in text:
                    text = text.replace(to_rm, "")
                else:
                    break

        ax.text(0.02, 0.98, text, ha="left", va="top", transform=ax.transAxes)

        return fig, ax

    def plot_fit_multi_oscillators(
        self,
        indices: Optional[Iterable[int]] = None,
        oscs: Optional[Iterable[int]] = None,
        fit_increments: int = 100,
        xaxis_unit: str = "ppm",
        xaxis_ticks: Optional[Union[Iterable[int], Iterable[Iterable[int, Iterable[float]]]]] = None,  # noqa: E501
        region_separation: float = 0.02,
        labels: bool = True,
        colors: Any = None,
        azim: float = 45.,
        elev: float = 45.,
        label_zshift: float = 0.,
        fit_line_kwargs: Optional[Dict] = None,
        scatter_kwargs: Optional[Dict] = None,
        label_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Create a 3D figure showing fits for multiple oscillators.

        The x-, y-, and z-axes comprise chemical shifts/oscillator indices,
        increment values, and amplitudes, respectively.

        Parameters
        ----------
        indices
            See :ref:`INDICES`.

        oscs
            Oscillators to consider. By default (``None``) all oscillators are
            considered.

        fit_increments
            The number of points in the fit lines.

        xaxis_unit
            The unit of the x-axis. Should be one of:

            * ``"hz"`` or ``"ppm"``: Fits are plotted at the oscillators'
              frequecnies, either in Hz or ppm.
            * ``"osc_idx"``: Fits are evenly spaced about the x-axis, featuring
              at the oscillators' respective indices.

        xaxis_ticks
            Specifies custom x-axis ticks for each region, overwriting the default
            ticks. Keeping this ``None`` (default) will use default ticks.

            * If ``xaxis_unit`` is ``"hz"`` or ``"ppm"``, see :ref:`XAXIS_TICKS`.
            * If ``xaxis_unit`` is ``osc_idx``, this should be a list of ints,
              all of which correspond to a valid oscillator index.

        region_separation
            The extent by which adjacent regions are separated in the figure,
            in axes coordinates.

        labels
            If ``True``, the values extracted from the fits are written by each
            fit.

        colors
            Colors to give to the fits. See :ref:`COLOR_CYCLE`.

        elev
            Elevation angle for the plot view.

        azim
            Azimuthal angle for the plot view.

        label_zshift
            Vertical shift for the labels of T1 values (if the exist, see
            ``labels``). Given in axes coordinates.

        fit_line_kwargs
            Keyword arguments for the fit line. Use this to specify features such
            as color, linewidth etc. All keys should be valid arguments for
            `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.
            If ``"color"`` is included, it is ignored (colors are processed
            based on the ``colors`` argument.

        scatter_kwargs
            Keyword arguments for the scatter plot. All keys should be valid arguments
            for
            `matplotlib.axes.Axes.scatter
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html>`_.
            If ``"color"`` is included, it is ignored (colors are procecessed
            based on the ``colors`` argument.

        label_kwargs
            Keyword arguments for fit labels. All keys should be valid arguments for
            `matplotlib.text.Text
            <https://matplotlib.org/stable/api/text_api.html#matplotlib.text.Text>`_
            If ``"color"`` is included, it is ignored (colors are procecessed
            based on the ``colors`` argument. ``"horizontalalignment"``, ``"ha"``,
            ``"verticalalignment"``, and ``"va"`` are also ignored.

        kwargs
            Keyword arguments for
            `matplotlib.pyplot.figure
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_

        Notes
        -----
        This function can take a bit of time to run (tens of seconds
        sometimes), and timings will significantly vary depending on the
        backend used (PGF, which I typically use, is quite time-consuming, for
        example). You can't rush art.
        """
        sanity_check(
            self._indices_check(indices),
            (
                "fit_increments", fit_increments, sfuncs.check_int, (),
                {"min_value": len(self.increments)},
            ),
            ("xaxis_unit", xaxis_unit, sfuncs.check_one_of, ("osc_idx", "hz", "ppm")),
            ("labels", labels, sfuncs.check_bool),
            ("colors", colors, sfuncs.check_oscillator_colors, (), {}, True),
            ("elev", elev, sfuncs.check_float),
            ("azim", azim, sfuncs.check_float),
            ("label_zshift", label_zshift, sfuncs.check_float),
        )
        if xaxis_unit != "osc_idx":
            sanity_check(self._funit_check(xaxis_unit, "xaxis_unit"))

        scatter_kwargs = proc_kwargs_dict(
            scatter_kwargs,
            default={"lw": 0., "depthshade": False},
            to_pop=("color", "c", "zorder"),
        )
        fit_line_kwargs = proc_kwargs_dict(
            fit_line_kwargs,
            to_pop=("color", "zorder"),
        )
        label_kwargs = proc_kwargs_dict(
            label_kwargs,
            to_pop=("ha", "horizontalalignment", "va", "verticalalignment", "color", "zdir"),  # noqa: E501
        )

        indices = self._process_indices(indices)
        params = self.get_params(
            indices=indices,
            funit="hz" if xaxis_unit != "ppm" else "ppm",
        )
        n_oscs = params[0].shape[0]
        sanity_check(self._oscs_check(oscs, n_oscs))
        oscs = self._process_oscs(oscs, n_oscs)
        params = [p[oscs] for p in params]

        amplitudes = self.amplitudes(indices, oscs)
        fit, errors = self.fit(indices, oscs)

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(projection="3d")

        colors = make_color_cycle(colors, n_oscs)
        zorder = n_oscs

        if xaxis_unit == "osc_idx":
            sanity_check(
                (
                    "xaxis_ticks", xaxis_ticks, sfuncs.check_list_with_elements_in,
                    (oscs,), {}, True,
                )
            )
            xs = oscs
            xlabel = "oscillator index"

        else:
            merge_indices, merge_regions = self._plot_regions(indices, xaxis_unit)
            sanity_check(
                (
                    "xaxis_ticks", xaxis_ticks, sfuncs.check_xaxis_ticks,
                    (merge_regions,), {}, True,
                ),
            )
            if (n_regions := len(merge_regions)) > 1:
                max_rs = 1 / (n_regions - 1)
                sanity_check(
                    (
                        "region_separation", region_separation, sfuncs.check_float,
                        (), {"min_value": 0., "max_value": max_rs},
                    ),
                )
            else:
                region_separation = 0.

            merge_region_spans = self._get_3d_xaxis_spans(
                merge_regions,
                region_separation,
            )

            xaxis_ticks, xaxis_ticklabels = self._get_3d_xticks_and_labels(
                xaxis_ticks,
                indices,
                xaxis_unit,
                merge_regions,
                merge_region_spans,
            )
            xs = self._transform_freq_to_xaxis(
                params[0][:, 2],
                merge_regions,
                merge_region_spans,
            )

            xlabel, = self._axis_freq_labels(xaxis_unit)

        y_min, y_max = np.amin(self.increments), np.amax(self.increments)
        for x, params, amps in zip(xs, fit, amplitudes):
            color = next(colors)
            x_scatter = np.full(self.n_increments, x)
            y_scatter = self.increments
            z_scatter = amps
            ax.scatter(
                x_scatter,
                y_scatter,
                z_scatter,
                color=color,
                zorder=zorder,
                **scatter_kwargs,
            )

            x_fit = np.full(fit_increments, x)
            y_fit = np.linspace(y_min, y_max, fit_increments)
            z_fit = self.model(*params, y_fit)
            ax.plot(x_fit, y_fit, z_fit, color=color, zorder=zorder, **fit_line_kwargs)

            ax.text(
                x_fit[-1],
                y_fit[-1],
                z_fit[-1] + label_zshift,
                f"{params[-1]:.3g}",
                color=color,
                zdir="z",
                ha="center",
                **label_kwargs,
            )

            zorder -= 1

        # azim at 270 provies a face-on view of the spectra.
        ax.view_init(elev=elev, azim=270. + azim)

        ax.set_xlabel(xlabel)

        if xaxis_ticks is not None:
            ax.set_xticks(xaxis_ticks)
            ax.set_xticklabels(xaxis_ticklabels)

        # Configure y-axis
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel(self.increment_label)

        # Configure z-axis
        ax.set_zlabel("$a$")

        if xaxis_unit != "osc_idx":
            self._set_3d_axes_xspine_and_lims(ax, merge_region_spans)
        else:
            ax.set_xlim(n_oscs - 1, 0)

        return fig, ax

    def plot_result(
        self,
        indices: Optional[Iterable[int]] = None,
        xaxis_unit: str = "hz",
        xaxis_ticks: Optional[Iterable[float]] = None,
        region_separation: float = 0.02,
        oscillator_colors: Any = None,
        elev: float = 45.,
        azim: float = 45.,
        spectrum_line_kwargs: Optional[Dict] = None,
        oscillator_line_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        """Generate a figure of the estimation result.

        A 3D plot is generated, showing the estimation result for each increment in
        the data.

        Parameters
        ----------
        indices
            See :ref:`INDICES`.

        xaxis_unit
            The unit to express chemical shifts in. Should be ``"hz"`` or ``"ppm"``.

        xaxis_ticks
            See :ref:`XAXIS_TICKS`.

        region_separation
            The extent by which adjacent regions are separated in the figure,
            in axes coordinates.

        oscillator_colors
            Describes how to color individual oscillators. See :ref:`COLOR_CYCLE`.

        elev
            Elevation angle for the plot view.

        azim
            Azimuthal angle for the plot view.

        spectrum_line_kwargs
            Keyword arguments for the spectrum lines. All keys should be valid
            arguments for
            `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.

        oscillator_line_kwargs
            Keyword arguments for the oscillator lines. All keys should be valid
            arguments for
            `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.
            If ``"color"`` is included, it is ignored (colors are processed based on
            the ``oscillator_colors`` argumnet.

        kwargs
            Keyword arguments for
            `matplotlib.pyplot.figure
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_
        """
        sanity_check(
            self._indices_check(indices),
            self._funit_check(xaxis_unit, "xaxis_unit"),
            (
                "oscillator_colors", oscillator_colors, sfuncs.check_oscillator_colors,
                (), {}, True,
            ),
            ("elev", elev, sfuncs.check_float),
            ("azim", azim, sfuncs.check_float),
        )

        spectrum_line_kwargs = proc_kwargs_dict(
            spectrum_line_kwargs,
            default={"linewidth": 0.8, "color": "k"},
        )
        oscillator_line_kwargs = proc_kwargs_dict(
            oscillator_line_kwargs,
            default={"linewidth": 0.5},
            to_pop=("color"),
        )

        indices = self._process_indices(indices)
        merge_indices, merge_regions = self._plot_regions(indices, xaxis_unit)

        if (n_regions := len(merge_regions)) > 1:
            max_rs = 1 / (n_regions - 1)
            sanity_check(
                (
                    "region_separation", region_separation, sfuncs.check_float,
                    (), {"min_value": 0., "max_value": max_rs},
                ),
            )
        else:
            region_separation = 0.

        sanity_check(
            (
                "xaxis_ticks", xaxis_ticks, sfuncs.check_xaxis_ticks,
                (merge_regions,), {}, True,
            ),
        )

        merge_region_spans = self._get_3d_xaxis_spans(
            merge_regions,
            region_separation,
        )

        xaxis_ticks, xaxis_ticklabels = self._get_3d_xticks_and_labels(
            xaxis_ticks,
            indices,
            xaxis_unit,
            merge_regions,
            merge_region_spans,
        )

        spectra = []
        for fid in self.data:
            fid[0] *= 0.5
            spectra.append(ne.sig.ft(fid).real)

        params = self.get_params(indices)

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(projection="3d", computed_zorder=False)

        colorcycle = make_color_cycle(oscillator_colors, params[0].shape[0])
        expinfo_direct = self.expinfo_direct
        for i, (spectrum, p, incr) in enumerate(
            zip(
                reversed(spectra), reversed(params), reversed(self.increments)
            )
        ):
            cc = copy.deepcopy(colorcycle)

            for (idx, region, span) in zip(
                merge_indices, merge_regions, merge_region_spans
            ):
                slice_ = slice(
                    *self.convert([region], f"{xaxis_unit}->idx")[0]
                )
                x = np.linspace(span[0], span[1], slice_.stop - slice_.start)
                y = np.full(x.shape, incr)
                spec = spectrum[slice_]
                ax.plot(x, y, spec, **spectrum_line_kwargs)
                for osc_params in p:
                    osc_params = np.expand_dims(osc_params, axis=0)
                    osc = expinfo_direct.make_fid(osc_params)
                    osc[0] *= 0.5
                    osc_spec = ne.sig.ft(osc).real[slice_]
                    ax.plot(x, y, osc_spec, color=next(cc), **oscillator_line_kwargs)

        ax.set_xticks(xaxis_ticks)
        ax.set_xticklabels(xaxis_ticklabels)
        ax.set_xlim(reversed(ax.get_xlim()))
        ax.set_xlabel(self._axis_freq_labels(xaxis_unit)[-1])
        ax.set_ylabel(self.increment_label)
        # ax.set_zticks([])
        # azim at 270 provies a face-on view of the spectra.
        ax.view_init(elev=elev, azim=270. + azim)

        # x-axis spine with breaks.
        self._set_3d_axes_xspine_and_lims(ax, merge_region_spans)

        return fig, ax

    def _plot_regions(
        self,
        indices: Iterable[int],
        xaxis_unit: str,
    ) -> Iterable[Tuple[float, float]]:
        merge_indices, merge_regions = (
            self._estimators[0]._plot_regions(indices, xaxis_unit)
        )

        return merge_indices, merge_regions

    def _get_3d_xticks_and_labels(
        self,
        xaxis_ticks: Optional[Iterable[int, Iterable[float]]],
        indices: Iterable[int],
        xaxis_unit: str,
        merge_regions: Iterable[Tuple[float, float]],
        merge_region_spans: Iterable[Tuple[float, float]],
    ) -> Tuple[Iterable[float], Iterable[str]]:
        # Get the default xticks from `Estimator1D.plot_result`.
        # Filter any away that are outside the region
        default_xaxis_ticks = []
        _, _axs, = (
            self._estimators[0].plot_result(
                indices=indices, xaxis_unit=xaxis_unit,
            )
        )
        for i, (_ax, region) in enumerate(zip(_axs[0], merge_regions)):
            mn, mx = min(region), max(region)
            default_xaxis_ticks.append(
                (i, [x for x in _ax.get_xticks() if mn <= x <= mx])
            )

        flatten = lambda lst: [x for sublist in lst for x in sublist]  # noqa: E731
        if xaxis_ticks is None:
            xaxis_ticks = default_xaxis_ticks
        else:
            for i, _ in enumerate(merge_regions):
                found = any([x[0] == i for x in xaxis_ticks])
                if not found:
                    xaxis_ticks.append(default_xaxis_ticks[i])

        xaxis_ticks = flatten([x[1] for x in xaxis_ticks])

        # Scale xaxis ticks so the lie in the range [0-1]
        xaxis_ticks_scaled = self._transform_freq_to_xaxis(
            [ticks for ticks in xaxis_ticks],
            merge_regions,
            merge_region_spans,
        )

        # TODO Better formatting of tick labels
        xaxis_ticklabels = [f"{xtick:.3g}" for xtick in xaxis_ticks]

        return xaxis_ticks_scaled, xaxis_ticklabels

    @staticmethod
    def _transform_freq_to_xaxis(
        values: Iterable[float],
        merge_regions: Iterable[Tuple[float, float]],
        merge_region_spans: Iterable[Tuple[float, float]],
    ) -> Iterable[float]:
        transformed_values = []
        coefficients = []
        for region, span in zip(merge_regions, merge_region_spans):
            min_region, max_region = min(region), max(region)
            m1 = 1 / (max_region - min_region)
            c1 = -min_region * m1
            m2 = max(span) - min(span)
            c2 = min(span)
            coefficients.append([m1, c1, m2, c2])

        scale = lambda x, coeffs: (x * coeffs[0] + coeffs[1]) * coeffs[2] + coeffs[3]  # noqa: E501, E731

        for value in values:
            # Determine which region the value lies in
            for region, coeffs in zip(merge_regions, coefficients):
                if min(region) <= value <= max(region):
                    transformed_values.append(scale(value, coeffs))
                    break

        return transformed_values

    @staticmethod
    def _get_3d_xaxis_spans(
        merge_regions: Iterable[float, float],
        region_separation: float,
    ) -> Iterable[Tuple[float, float]]:
        n_regions = len(merge_regions)
        # x-axis will span [0, 1].
        # Figure out sections of this range to assign each region
        plot_width = 1 - (n_regions - 1) * region_separation
        merge_region_widths = [max(mr) - min(mr) for mr in merge_regions]
        merge_region_sum = sum(merge_region_widths)
        merge_region_widths = [
            mrw / merge_region_sum * plot_width
            for mrw in merge_region_widths
        ]
        merge_region_lefts = [
            1 - (i * region_separation) - sum(merge_region_widths[:i])
            for i in range(n_regions)
        ]
        merge_region_spans = [
            (mrl, mrl - mrw)
            for mrl, mrw in zip(merge_region_lefts, merge_region_widths)
        ]
        return merge_region_spans

    def _set_3d_axes_xspine_and_lims(
        self,
        ax: mpl.axes.Axes,
        merge_region_spans: Iterable[Tuple[float, float]],
    ) -> None:
        xspine = ax.w_xaxis.line
        xspine_kwargs = {
            "lw": xspine.get_linewidth(),
            "color": xspine.get_color(),
        }
        ax.w_xaxis.line.set_visible(False)
        curr_zlim = ax.get_zlim()
        spine_y = 2 * [self.increments[0]]
        spine_z = 2 * [curr_zlim[0]]
        for span in merge_region_spans:
            ax.plot(span, spine_y, spine_z, **xspine_kwargs, zorder=10000)

        ax.set_xlim(1, 0)
        ax.set_ylim(self.increments[0], self.increments[-1])
        ax.set_zlim(curr_zlim)

    @logger
    def plot_result_increment(
        self,
        increment: int = 0,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, np.ndarray[mpl.axes.Axes]]:
        """Generate a figure of the estimation result for a particular increment.

        The figure created is of the same form as those generated by
        :py:meth:`nmrespy.Estimator1D.plot_result`.

        Parameters
        ----------
        increment
            The increment to view. By default, the first increment in used (``0``).

        kwargs
            All kwargs are accepted by :py:meth:`nmrespy.Estimator1D.plot_result`.

        See Also
        --------
        :py:meth:`nmrespy.Estimator1D.plot_result`.
        """
        sanity_check(
            (
                "increment", increment, sfuncs.check_int, (),
                {"min_value": 0, "max_value": self.n_increments - 1},
            ),
        )

        kwargs = proc_kwargs_dict(kwargs, default={"plot_model": False})
        fig, axs = self._estimators[increment].plot_result(**kwargs)

        return fig, axs

    def plot_dosy(
        self,
        y_range: Tuple[float, float],
        y_pts: int = 128,
        distribution_width: float = 1.,
        y_scale: str = "linear",
        indices: Optional[Iterable[int]] = None,
        region_separation: float = 0.02,
        oscillator_colors: Any = None,
        contour_base: Optional[float] = None,
        contour_factor: Optional[float] = 1.2,
        contour_nlevels: Optional[int] = 10,
        xaxis_unit: str = "hz",
        xaxis_ticks: Optional[Iterable[float]] = None,
        label_peaks: bool = True,
        gridspec_kwargs: Optional[Dict] = None,
        spectrum_line_kwargs: Optional[Dict] = None,
        oscillator_line_kwargs: Optional[Dict] = None,
        contour_kwargs: Optional[Dict] = None,
        label_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Generate a plot which maps oscillator locations to fit values.

        These plots effectively resemble DOSY spectra, in which chemical shift is
        along the x-axis, and the quantity of interest (:math:`D`, :math:`T_1`,
        :math:`T_2`, etc) is along the y-axis.

        Parameters
        ----------
        y_range
            Limits of the y-axis.

        y_pts
            The number of increments along the y-axis. **Be careful with this value.
            See the "Notes" section below**.

        distribution_width
            Affects the linewidth of the Gaussian distribution plotted
            along the y-axis for each oscillator. The standard deviation of the
            Guassian is given by ``distribution_width * error`` where ``error``
            is the error associated with the parameter. See Notes below for
            more information.

        y_scale
            Should be either ``"linear"`` or ``"log"``.

        indices
            See :ref:`INDICES`.

        region_separation
            The extent by which adjacent regions are separated in the figure,
            in axis coordinates.

        oscillator_colors
            Describes how to color individual oscillators. See :ref:`COLOR_CYCLE`
            for details.

        contour_base
            The lowest level for the contour levels.

        contour_factor
            The geometric scaling factor for adjacent contours.

        contour_nlevels
            The number of contour levels.

        xaxis_unit
            Should be one of ``"hz"`` or ``"ppm"``.

        xaxis_ticks
            See :ref:`XAXIS_TICKS`.

        label_peaks
            If ``True``, peaks are labelled with their oscillator index.

        gridspec_kwargs
            Keyword arguments given to
            `matplotlib.pyplot.subplots
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html>`_
            through the argument ``gridspec_kw``. All keys should be valid argumnets
            for `matplotlib.gridspec.GridSpec
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html>`_.
            If ``"hspace"`` is included, it is ignored (it is forced to be set
            to ``0``).

        spectrum_line_kwargs
            Keyword arguments for the spectrum line. All keys should be valid
            arguments for `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.

        oscillator_line_kwargs
            Keyword arguments for the oscillator lines. All keys should be valid
            arguments for `matplotlib.axes.Axes.plot
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.plot.html>`_.
            If ``"color"`` is included, it is ignored (colors are processed
            based on the ``oscillator_colors`` argument.

        contour_kwargs
            Keyword arguments for the contour plot. All keys should be valid
            arguments for `matplotlib.pyplot.contour
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contour.html>`_.
            All of the following are ignored: ``"colors"``, ``"cmap"``, ``"levels"``.

        label_kwargs
            Keyword arguments for osciilator labels (only relevent if
            ``label_peaks`` is ``True``). All keys should be valid
            arguments for `matplotlib.pyplot.text
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.text.html>`_.

        kwargs
            All extra kwargs are given to
            `matplotlib.pyplot.figure
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html>`_.

        Returns
        -------
        fig
            Figure

        axs
            Axes objects, of length 2.

        Notes
        -----
        You may find there are scenarios when the following warning appears:

        .. code::

            RuntimeWarning: invalid value encountered in divide
            gaussian /= np.amax(gaussian)

        This can occur when you have fits with low errors.
        You will not see certain peaks appear in the plot in these circumstances.
        To resolve this, you are advised to increase the value of
        ``distribution_width``, in order to increase the linewidth of the
        distribution plotted along the y-axis.
        """
        sanity_check(
            (
                "y_range", y_range, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True},
            ),
            ("y_pts", y_pts, sfuncs.check_int, (), {"min_value": 1}),
            (
                "distribution_width", distribution_width, sfuncs.check_float, (),
                {"greater_than_zero": True},
            ),
            ("y_scale", y_scale, sfuncs.check_one_of, ("linear", "log")),
            self._indices_check(indices),
            self._funit_check(xaxis_unit, "xaxis_unit"),
            (
                "oscillator_colors", oscillator_colors, sfuncs.check_oscillator_colors,
                (), {}, True,
            ),
            (
                "contour_base", contour_base, sfuncs.check_float, (),
                {"min_value": 0.}, True,
            ),
            (
                "contour_factor", contour_factor, sfuncs.check_float, (),
                {"min_value": 1.}, True,
            ),
            (
                "contour_nlevels", contour_nlevels, sfuncs.check_int, (),
                {"min_value": 1}, True,
            ),
            ("xaxis_unit", xaxis_unit, sfuncs.check_one_of, ("hz", "ppm")),
            ("label_peaks", label_peaks, sfuncs.check_bool),
        )

        if all(
            [
                x is not None
                for x in (contour_base, contour_nlevels, contour_factor)
            ]
        ):
            contour_levels = [
                contour_base * contour_factor ** i for i in range(contour_nlevels)
            ]
        else:
            contour_levels = None

        gridspec_kwargs = proc_kwargs_dict(
            gridspec_kwargs,
            to_pop=("hspace",)
        )
        gridspec_kwargs = proc_kwargs_dict(
            gridspec_kwargs,
            default={"hspace": 0.},
        )
        spectrum_line_kwargs = proc_kwargs_dict(
            spectrum_line_kwargs,
            default={"linewidth": 1.2, "color": "k"},
        )
        oscillator_line_kwargs = proc_kwargs_dict(
            oscillator_line_kwargs,
            default={"linewidth": 0.9},
            to_pop=("color",),
        )
        contour_kwargs = proc_kwargs_dict(
            contour_kwargs,
            default={"linewidths": 0.5},
            to_pop=("colors", "cmap", "levels"),
        )
        label_kwargs = proc_kwargs_dict(
            label_kwargs,
            default={"fontsize": 7},
            to_pop=("color",),
        )

        indices = self._process_indices(indices)
        merge_indices, merge_regions = self._plot_regions(indices, xaxis_unit)

        if (n_regions := len(merge_regions)) > 1:
            max_rs = 1 / (n_regions - 1)
            sanity_check(
                (
                    "region_separation", region_separation, sfuncs.check_float,
                    (), {"min_value": 0., "max_value": max_rs},
                ),
            )

        sanity_check(
            (
                "xaxis_ticks", xaxis_ticks, sfuncs.check_xaxis_ticks,
                (merge_regions,), {}, True,
            ),
        )

        merge_region_spans = self._get_3d_xaxis_spans(
            merge_regions,
            region_separation,
        )
        xaxis_ticks, xaxis_ticklabels = self._get_3d_xticks_and_labels(
            xaxis_ticks,
            indices,
            xaxis_unit,
            merge_regions,
            merge_region_spans,
        )

        y = np.linspace(y_range[0], y_range[1], y_pts)
        zss = []
        peakss = []
        for mi in merge_indices:
            thetas = self.get_params(indices=mi)[0]
            fits, errors = self.fit(indices=mi)
            fits = [x[1] for x in fits]
            errors = [x[1] for x in errors]

            zs = []
            peaks = []
            for theta, fit, error in zip(thetas, fits, errors):
                fid = self.make_fid(np.expand_dims(theta, axis=0))
                fid[0] *= 0.5
                spec = ne.sig.ft(fid).real
                if type(self).__name__ == "EstimatorInvRec":
                    spec *= -1
                peaks.append(spec)

                if y_range[0] <= fit <= y_range[1]:
                    sigma = distribution_width * error
                    gaussian = np.exp((-0.5 * (y - fit) ** 2) / (sigma ** 2))
                    gaussian /= np.amax(gaussian)
                    zs.append(np.outer(spec, gaussian))
                else:
                    zs.append(None)

            zss.append(zs)
            peakss.append(peaks)

        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            gridspec_kw=gridspec_kwargs,
            **kwargs,
        )
        colorcycle = make_color_cycle(oscillator_colors, sum([len(zs) for zs in zss]))

        fid = self.data[0]
        fid[0] *= 0.5
        spectrum = ne.sig.ft(fid).real
        if type(self).__name__ == "EstimatorInvRec":
            spectrum *= -1

        if label_peaks:
            peak_idx = 0
        for i, (zs, peaks, region, span) in enumerate(zip(
            reversed(zss),
            reversed(peakss),
            reversed(merge_regions),
            reversed(merge_region_spans),
        )):
            slice_ = slice(
                *self.convert([region], f"{xaxis_unit}->idx")[0]
            )
            x = np.linspace(span[0], span[1], slice_.stop - slice_.start)
            xx, yy = np.meshgrid(x, y, indexing="ij")
            zs_slice = [z[slice_] if z is not None else None for z in zs]
            spec = spectrum[slice_]
            axs[0].plot(x, spec, **spectrum_line_kwargs)

            peaks = [peak[slice_] for peak in peaks]
            for peak, z_slice in zip(peaks, zs_slice):
                color = next(colorcycle)
                axs[0].plot(x, peak, color=color, **oscillator_line_kwargs)
                if z_slice is not None:
                    axs[1].contour(
                        xx,
                        yy,
                        z_slice,
                        colors=color,
                        levels=contour_levels,
                        **contour_kwargs,
                    )

                if label_peaks:
                    x_argmax = np.argmax(peak)
                    x_label = x[x_argmax]
                    axs[0].text(
                        x_label,
                        peak[x_argmax],
                        str(peak_idx),
                        color=color,
                        **label_kwargs,
                    )
                    if z_slice is not None:
                        y_argmax = np.argmax(z_slice[x_argmax])
                        axs[1].text(
                            x_label,
                            y[y_argmax],
                            str(peak_idx),
                            color=color,
                            **label_kwargs,
                        )

                    peak_idx += 1

            del zs[-1]

        if y_scale == "linear":
            ylabel = f"{self.fit_labels[-1]} ({self.fit_units[-1]})"
        elif y_scale == "log":
            # [1:-1] is to slice away $s
            ylabel = f"$\\log({self.fit_labels[-1][1:-1]})$"
            axs[1].set_yscale("log")

        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_xticks(xaxis_ticks)
        axs[1].set_xticklabels(xaxis_ticklabels)
        axs[0].set_xlim(1, 0)
        axs[1].set_xlim(1, 0)
        axs[1].set_xlabel(self._axis_freq_labels(xaxis_unit)[-1])
        axs[1].set_ylabel(ylabel)

        # Configure spines
        spine_lw = axs[0].spines["top"].get_lw()
        spine_color = axs[0].spines["top"].get_edgecolor()
        break_kwargs = {
            "marker": [(-1, -3), (1, 3)],
            "markersize": 6,
            "linestyle": "none",
            "color": spine_color,
            "mec": spine_color,
            "mew": 1,
            "clip_on": False,
        }
        for ax in axs:
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            for i, mrs in enumerate(reversed(merge_region_spans)):
                if i != 0:
                    ax.plot([1 - mrs[1], 1 - mrs[1]], [0, 1], transform=ax.transAxes, **break_kwargs)
                if i != n_regions - 1:
                    ax.plot([1 - mrs[0], 1 - mrs[0]], [0, 1], transform=ax.transAxes, **break_kwargs)
                for y in ([0, 0], [1, 1]):
                    ax.plot(
                        [1 - mrs[0], 1 - mrs[1]],
                        y,
                        color=spine_color,
                        lw=spine_lw,
                        transform=ax.transAxes,
                        clip_on=False,
                    )

        return fig, axs

    def get_params(
        self,
        indices: Optional[Iterable[int]] = None,
        merge: bool = True,
        funit: str = "hz",
        sort_by: str = "f-1",
    ) -> Optional[Union[Iterable[np.ndarray], np.ndarray]]:
        params = []
        for estimator in self._estimators:
            params.append(estimator.get_params(indices, merge, funit, sort_by))

        return params

    def _check_results_exist(self) -> None:
        self._estimators[0]._check_results_exist()

    def _oscs_check(self, x: Any, n_oscs: int):
        return (
            "oscs", x, sfuncs.check_int_list, (),
            {
                "min_value": 0,
                "max_value": n_oscs - 1,
                "len_one_can_be_listless": True,
            }, True,
        )

    def _process_indices(self, indices: Optional[Iterable[int]]) -> Iterable[int]:
        return self._estimators[0]._process_indices(indices)

    def _process_oscs(
        self,
        oscs: Optional[Union[int, Iterable[int]]],
        n_oscs: int,
    ) -> Iterable[int]:
        if oscs is None:
            oscs = list(range(n_oscs))
        elif isinstance(oscs, int):
            oscs = [oscs]
        return oscs

    def _indices_check(self, x: Any):
        return self._estimators[0]._indices_check(x)

    def _index_check(self, x: Any):
        return self._estimators[0]._index_check(x)
