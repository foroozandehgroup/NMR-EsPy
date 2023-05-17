# twodim.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 16 May 2023 18:29:06 BST

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import nmrespy as ne
from nmrespy.load import load_bruker
from nmrespy._colors import RED, GRE, END, USE_COLORAMA
from nmrespy._files import check_existent_dir
from nmrespy._misc import boxed_text, proc_kwargs_dict
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)

from nmrespy.estimators import logger
from nmrespy.estimators import Estimator, Result
from nmrespy.estimators.jres import ContourApp

if USE_COLORAMA:
    import colorama
    colorama.init()


class Estimator2D(Estimator):
    """Estimator class for 2D data comprising a pair of States signals."""

    dim = 2
    proc_dims = [0, 1]
    # twodim_dtype is dictated by the type of data brought in
    # Can be "amp" or "phase" (see __init__)
    ft_dims = [0, 1]
    default_mpm_trim = (64, 128)
    default_nlp_trim = (128, 1024)
    default_max_iterations_exact_hessian = 40
    default_max_iterations_gn_hessian = 100

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        twodim_dtype: str,
        datapath: Optional[Path] = None,
    ) -> None:
        super().__init__(data, expinfo, datapath)
        sanity_check(("twodim_dtype", twodim_dtype, sfuncs.check_one_of, ("amp", "phase")))  # noqa: E501
        self.twodim_dtype = twodim_dtype

    @classmethod
    def new_bruker(
        cls,
        directory: Union[str, Path],
        convdta: bool = True,
    ) -> Estimator2D:
        sanity_check(
            ("directory", directory, check_existent_dir),
            ("convdta", convdta, sfuncs.check_bool),
        )
        directory = Path(directory).expanduser()
        data, expinfo = load_bruker(directory)
        pts_2 = data.shape[1] // 2
        data_shape = (2, data.shape[0], pts_2)

        if data.ndim != 2:
            raise ValueError(f"{RED}Data dimension should be 2.{END}")
        data_proc = np.zeros(data_shape, dtype="complex128")

        data_proc[0] = data[:, :pts_2]
        data_proc[1] = data[:, pts_2:]

        if convdta:
            grpdly = expinfo.parameters["acqus"]["GRPDLY"]
            data = ne.sig.convdta(data, grpdly)

        return cls(data_proc, expinfo, "phase", datapath=directory)

    @classmethod
    def new_spinach_hsqc(
        cls,
        shifts: Iterable[float],
        couplings: Optional[Iterable[Tuple(int, int, float)]],
        isotopes: Iterable[str],
        pts: Tuple[int, int],
        sw: Tuple[float, float],
        offset: Tuple[float, float],
        field: float = 11.74,
        nuclei: Tuple[str, str] = ("13C", "1H"),
        snr: Optional[float] = 30.,
        lb: float = 6.91,
    ) -> Estimator2D:
        r"""Create a new instance from a Spinach HSQC simulation.

        Parameters
        ----------
        shifts
            A list of tuple of chemical shift values for each spin.

        couplings
            The scalar couplings present in the spin system. Given ``shifts`` is of
            length ``n``, couplings should be an iterable with entries of the form
            ``(i1, i2, coupling)``, where ``1 <= i1, i2 <= n`` are the indices of
            the two spins involved in the coupling, and ``coupling`` is the value
            of the scalar coupling in Hz. ``None`` will set all spins to be
            uncoupled.

        isotopes
            A list of the identities of each nucleus. This should be a list with the
            same length as ``shifts``, with all elements being one of the
            elements in ``nuclei``.

        pts
            The number of points the signal comprises.

        sw
            The sweep width of the signal (Hz).

        offset
            The transmitter offset (Hz).

        field
            The magnetic field strength (T).

        nucleus
            The identity of the nuclei targeted in the pulse sequence.

        snr
            The signal-to-noise ratio of the resulting signal, in decibels. ``None``
            produces a noiseless signal.

        lb
            Line broadening (exponential damping) to apply to the signal.
            The first point will be unaffected by damping, and the final point will
            be multiplied by ``np.exp(-lb)``. The default results in the final
            point being decreased in value by a factor of roughly 1000.
        """

        sanity_check(
            ("shifts", shifts, sfuncs.check_float_list),
            ("pts", pts, sfuncs.check_int_list, (), {"length": 2, "min_value": 1}),
            ("sw", sw, sfuncs.check_float_list, (), {"length": 2, "must_be_positive": True}),  # noqa: E501
            ("offset", offset, sfuncs.check_float_list, (), {"length": 2}),
            ("field", field, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("nuclei", nuclei, sfuncs.check_nucleus_list, (), {"length": 2}),
            ("snr", snr, sfuncs.check_float, (), {}, True),
            ("lb", lb, sfuncs.check_float, (), {"greater_than_zero": True})
        )
        nspins = len(shifts)
        sanity_check(
            ("couplings", couplings, sfuncs.check_spinach_couplings, (nspins,), {}, True),  # noqa: E501
            ("isotopes", isotopes, sfuncs.check_nucleus_list, (), {"length": nspins}),
        )

        if couplings is None:
            couplings = []

        fid, sfo = cls._run_spinach(
            "hsqc_sim", shifts, couplings, isotopes, field, pts, sw, offset, nuclei,
            to_int=[4], to_double=[5, 6],
        )
        fid_pos = np.array(fid["pos"]).T
        fid_neg = np.array(fid["neg"]).T
        fid = np.zeros((2, *fid_pos.shape), dtype="complex128")
        fid[0] = fid_pos
        fid[1] = fid_neg
        sfo = tuple([x[0] for x in sfo])

        if snr is not None:
            fid = ne.sig.add_noise(fid, snr)

        fid = ne.sig.exp_apodisation(fid, lb, axes=(1, 2))

        expinfo = ne.ExpInfo(
            dim=2,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuclei=nuclei,
            default_pts=fid.shape[1:],
        )

        return cls(fid, expinfo, "phase")

    @classmethod
    def new_from_parameters(
        cls,
        params: np.ndarray,
        pts: Tuple[int, int],
        sw: Tuple[float, float],
        offset: Tuple[float, float],
        sfo: Tuple[Optional[float], Optional[float]] = (125., 500.),
        nucleus: str = ("13C", "1H"),
        snr: Optional[float] = 20.,
        twodim_dtype: str = "amp",
    ) -> None:
        sanity_check(
            ("params", params, sfuncs.check_parameter_array, (2,)),
            ("pts", pts, sfuncs.check_int_list, (), {"length": 2, "min_value": 1}),
            (
                "sw", sw, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True},
            ),
            (
                "offset", offset, sfuncs.check_float_list, (),
                {"length": 2}, True,
            ),
            ("nucleus", nucleus, sfuncs.check_nucleus_list, (), {"length": 2}),
            (
                "sfo", sfo, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True}, True,
            ),
            ("snr", snr, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            ("twodim_dtype", twodim_dtype, sfuncs.check_one_of, ("amp", "phase")),
        )

        expinfo = ne.ExpInfo(
            2,
            sw=sw,
            offset=offset,
            sfo=sfo,
            nuclei=nucleus,
            default_pts=pts,
        )

        data = expinfo.make_fid(params, snr=snr, indirect_modulation=twodim_dtype)
        return cls(data, expinfo, twodim_dtype)

    @property
    def spectrum(self) -> np.ndarray:
        if self.twodim_dtype == "amp":
            return ne.sig.proc_amp_modulated(self.data).real
        elif self.twodim_dtype == "phase":
            return ne.sig.proc_phase_modulated(self.data)[0]

    def get_data_1d(self, dim: int = 1) -> np.ndarray:
        if self.twodim_dtype == "amp":
            data_2d = self.data[0] + 1j * self.data[1]
        else:
            data_2d = self.data[0]

        if dim == 0:
            return data_2d[:, 0]
        else:
            return data_2d[0]

    def get_spectrum_1d(self, dim: int = 1) -> np.ndarray:
        return ne.sig.ft(self.get_data_1d(dim))

    def get_1d_expinfo(self, dim: int = 1) -> ne.ExpInfo:
        sanity_check(("dim", dim, sfuncs.check_one_of, (0, 1)))
        return ne.ExpInfo(
            dim=1,
            sw=self.sw()[dim],
            offset=self.offset()[dim],
            sfo=self.sfo[dim],
            nuclei=self.nuclei[dim],
            default_pts=self.data.shape[dim + 1],
        )

    def view_data_1d(
        self,
        dim: int = 1,
        **kwargs,
    ) -> None:
        sanity_check(
            ("dim", dim, sfuncs.check_one_of, (0, 1)),
        )
        estimator_1d = self.get_estimator_1d(dim)
        estimator_1d.view_data(**kwargs)

    def view_data(
        self,
        domain: str = "freq",
    ) -> None:
        spectrum = self.spectrum
        app = ContourApp(spectrum, self.expinfo)
        app.mainloop()

    def get_estimator_1d(self, dim: int = 1) -> ne.Estimator1D:
        data_1d = self.get_data_1d(dim)
        expinfo_1d = self.get_1d_expinfo(dim)
        estimator_1d = ne.Estimator1D(data_1d, expinfo_1d)

        if not self._results:
            return estimator_1d

        results_1d = []
        freq_idx = 2 + dim
        damp_idx = 4 + dim
        for result_2d in self.get_results():
            params_2d, errors_2d = result_2d.get_params(), result_2d.get_errors()
            params_1d, errors_1d = [
                np.zeros((params_2d.shape[0], 4), dtype="float64") for _ in range(2)
            ]
            params_1d[:, :2], errors_1d[:, :2] = params_2d[:, :2], errors_2d[:, :2]
            params_1d[:, 2], errors_1d[:, 2] = params_2d[:, freq_idx], errors_2d[:, freq_idx]  # noqa: E501
            params_1d[:, 3], errors_1d[:, 3] = params_2d[:, damp_idx], errors_2d[:, damp_idx]  # noqa: E501
            region_1d = (result_2d.get_region()[dim],)
            noise_region_1d = (result_2d.get_noise_region()[dim],)
            result_1d = Result(
                params_1d,
                errors_1d,
                region_1d,
                noise_region_1d,
                sfo=(self.sfo[dim],),
            )
            results_1d.append(result_1d)
        estimator_1d._results = results_1d

        return estimator_1d

    def plot_result(
        self,
        index: int = -1,
        region_unit: str = "hz",
        contour_base: Optional[float] = None,
        contour_nlevels: Optional[int] = None,
        contour_factor: Optional[float] = None,
        oscillator_colors: Any = None,
        contour_kwargs: Optional[Dict] = None,
        scatter_kwargs: Optional[Dict] = None,
    ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        sanity_check(
            self._index_check(index),
            self._funit_check(region_unit, "region_unit"),
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
            (
                "oscillator_colors", oscillator_colors, sfuncs.check_oscillator_colors,
                (), {}, True,
            ),
        )

        contour_kwargs = proc_kwargs_dict(
            contour_kwargs,
            default={"linewidths": 0.4, "colors": "k"},
            to_pop=["levels"],
        )
        scatter_kwargs = proc_kwargs_dict(
            scatter_kwargs,
            default={"marker": "o", "s": 8},
            to_pop=["color"],
        )

        index, = self._process_indices([index])
        result = self.get_results(indices=[index])[0]
        region = result.get_region(unit=region_unit)
        params = result.get_params(funit=region_unit)
        n_oscs = params.shape[0]
        full_shifts_2d_y, full_shifts_2d_x = self.get_shifts(unit=region_unit)
        full_spectrum = self.spectrum

        conv = f"{region_unit}->idx"
        slice_ = [slice(r[0], r[1], None) for r in self.expinfo.convert(region, conv)]
        spectrum = full_spectrum[slice_[0], slice_[1]]
        shifts_2d_x = full_shifts_2d_x[slice_[0], slice_[1]]
        shifts_2d_y = full_shifts_2d_y[slice_[0], slice_[1]]
        freqs = []
        for osc in params:
            freqs.append((osc[2], osc[3]))

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

        fig, ax = plt.subplots(nrows=1, ncols=1)

        ax.contour(
            shifts_2d_x,
            shifts_2d_y,
            spectrum,
            levels=contour_levels,
            **contour_kwargs,
        )

        colors = ne.plot.make_color_cycle(oscillator_colors, n_oscs)
        for (f1, f2) in freqs:
            ax.scatter(
                x=f2,
                y=f1,
                color=next(colors),
                zorder=100,
                **scatter_kwargs,
            )

        ylabel, xlabel = self._axis_freq_labels(unit=region_unit)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xlim(shifts_2d_x[0, 0], shifts_2d_x[0, -1])
        ax.set_ylim(shifts_2d_y[0, 0], shifts_2d_y[-1, 0])

        return fig, ax

    def plot_result_1d(
        self,
        dim: int = 1,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, np.ndarray[mpl.axes.Axes]]:
        """Generate a figure of the estimation result, looking at one dimension.

        Parameters
        ----------
        dim
            The dimension to consider. ``0`` corresponds to the indirect dimension,
            ``1`` corresponds to the direct dimension.

        kwargs
            For a description of all other arguments, see
            :py:meth:`Estimator1D.plot_result`.
        """
        sanity_check(
            ("dim", dim, sfuncs.check_one_of, (0, 1)),
        )
        estimator_1d = self.get_estimator_1d(dim)
        fig, axs = estimator_1d.plot_result(**kwargs)
        return fig, axs
