# jres.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 21 Jul 2023 12:36:09 BST

from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from nmrespy import ExpInfo, sig
from nmrespy.contour_app import ContourApp
from nmrespy.load import load_bruker
from nmrespy.estimators import logger
from nmrespy.estimators._proc_onedim import _Estimator1DProc
from nmrespy.plot import make_color_cycle
from nmrespy._colors import RED, GRE, END, USE_COLORAMA
from nmrespy._files import check_existent_dir, check_saveable_dir
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)


if USE_COLORAMA:
    import colorama
    colorama.init()


class Estimator2DJ(_Estimator1DProc):
    """Estimator class for J-Resolved (2DJ) datasets, enabling use of our CUPID
    method for Pure Shift spectra. For a tutorial on the basic functionailty
    this provides, see :ref:`ESTIMATOR2DJ`.

    .. note::

        To create an instance of ``Estimator2DJ``, you are advised to use one of
        the following methods if any are appropriate:

        * :py:meth:`new_bruker`
        * :py:meth:`increment=i, new_spinach`
        * :py:meth:`from_pickle` (re-loads a previously saved estimator).
    """

    dim = 2
    twodim_dtype = "hyper"
    proc_dims = [1]
    ft_dims = [0, 1]
    default_mpm_trim = [256]
    default_nlp_trim = [1024]
    default_max_iterations_exact_hessian = 40
    default_max_iterations_gn_hessian = 80

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
            If ``True``, removal of the FID's digital filter will be carried out,
            using the ``GRPDLY`` parameter.

        Notes
        -----
        There are certain file paths expected to be found relative to ``directory``
        which contain the data and parameter files:

        * ``directory/ser``
        * ``directory/acqus``
        * ``directory/acqu2s``

        See also
        --------
        :py:meth:`nmrespy.sig.convdta`
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
        couplings: Iterable[Tuple(int, int, float)],
        pts: Tuple[int, int],
        sw: Tuple[float, float],
        offset: float,
        field: float = 11.74,
        nucleus: str = "1H",
        snr: Optional[float] = 20.,
        lb: Optional[Tuple[float, float]] = (6.91, 6.91),
    ) -> None:
        r"""Create a new instance from a 2DJ Spinach simulation.

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

        pts
            The number of points the signal comprises.

        sw
            The sweep width of the signal (Hz).

        offset
            The transmitter offset (Hz).

        field
            The magnetic field strength (T).

        nucleus
            The identity of the nucleus targeted in the pulse sequence.

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
            (
                "sw", sw, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": 0.},
            ),
            ("offset", offset, sfuncs.check_float),
            ("field", field, sfuncs.check_float, (), {"greater_than_zero": True}),
            ("nucleus", nucleus, sfuncs.check_nucleus),
            ("snr", snr, sfuncs.check_float),
            (
                "lb", lb, sfuncs.check_float_list, (),
                {"length": 2, "must_be_positive": True},
            ),
        )
        nspins = len(shifts)
        sanity_check(
            ("couplings", couplings, sfuncs.check_spinach_couplings, (nspins,)),
        )

        if couplings is None:
            couplings = []

        fid, sfo = cls._run_spinach(
            "jres_sim", shifts, couplings, pts, sw, offset, field, nucleus,
            to_int=[2], to_double=[3, 4],
        )
        fid = np.array(fid)
        fid = sig.phase(fid, (0., np.pi / 2), (0., 0.))

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
            nuclei=(None, nucleus),
            default_pts=fid.shape,
        )

        return cls(fid, expinfo)

    def view_data(
        self,
        domain: str = "freq",
        abs_: bool = True,
    ) -> None:
        r"""View the data FID or the spectral data with an interactive matplotlib
        figure.

        Parameters
        ----------
        domain
            Must be ``"freq"`` or ``"time"``.

        abs\_
            Whether or not to display frequency-domain data in absolute-value mode,
            as is conventional with 2DJ data.
        """
        sanity_check(
            ("domain", domain, sfuncs.check_one_of, ("freq", "time")),
            ("abs_", abs_, sfuncs.check_bool),
        )

        if domain == "freq":
            spectrum = np.abs(self.spectrum_sinebell) if abs_ else self.spectrum
            app = ContourApp(spectrum, self.expinfo)
            app.mainloop()

        elif domain == "time":
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")

            x, y = self.get_timepoints()
            xlabel, ylabel = [f"$t_{i}$ (s)" for i in range(1, 3)]

            ax.plot_wireframe(x, y, self.data)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(reversed(ax.get_xlim()))
            ax.set_ylim(reversed(ax.get_ylim()))
            ax.set_zticks([])

            plt.show()

    @property
    def spectrum_tilt(self) -> np.ndarray:
        """Generate the spectrum of the data with a 45° tilt."""
        spectrum = np.abs(self.spectrum_sinebell).real
        sw1, sw2 = self.sw()
        n1, n2 = self.default_pts
        tilt_factor = (sw1 * n2) / (sw2 * n1)
        for i, row in enumerate(spectrum):
            spectrum[i] = np.roll(
                row,
                shift=int(tilt_factor * (n1 // 2 - i)),
            )

        return spectrum

    @property
    def spectrum_sinebell(self) -> np.ndarray:
        """Spectrum with sine-bell apodisation.

        Generated applying sine-bell apodisation to the FID, and applying FT.
        """
        data = copy.deepcopy(self.data)
        data[0, 0] *= 0.5
        data = sig.sinebell_apodisation(data)
        return sig.ft(data)

    @property
    def default_multiplet_thold(self) -> float:
        r"""The default margin for error when determining oscillators which belong to
        the same multiplet.

        Given by :math:`f_{\text{sw}}^{(1)} / 2 N^{(1)}` (i.e. half the
        spetral resolution in the indirect dimension).
        """
        return 0.5 * (self.sw()[0] / self.default_pts[0])

    @logger
    def cupid_signal(
        self,
        indices: Optional[Iterable[int]] = None,
        pts: Optional[int] = None,
        _log: bool = True,
    ) -> np.ndarray:
        r"""Generate the signal :math:`y_{-45^{\circ}}(t)`, where :math:`t \geq 0`:

        .. math::

            y_{-45^{\circ}}(t) = \sum_{m=1}^M a_m \exp\left( \mathrm{i} \phi_m \right)
            \exp\left( \left(2 \mathrm{i} \pi \left(f^{(2)}_m - f^{(1)}_m \right)
            - \eta^{(2)}_m \right) t \right)

        Producing this signal from parameters derived from estimation of a 2DJ dataset
        should generate an absorption-mode 1D homodecoupled spectrum.

        Parameters
        ----------
        indices
            See :ref:`INDICES`.

        pts
            The number of points to construct the signal from. If ``None``,
            ``self.default_pts`` will be used.
        """
        self._check_results_exist()
        sanity_check(
            self._indices_check(indices),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}, True),
        )

        params = self.get_params(indices)
        offset = self.offset()[1]
        if pts is None:
            pts = self.default_pts[1]
        tp = self.get_timepoints(pts=(1, pts), meshgrid=False)[1]
        f1 = params[:, 2]
        f2 = params[:, 3]
        signal = np.einsum(
            "ij,j->i",
            np.exp(
                np.outer(
                    tp,
                    2j * np.pi * (f2 - f1 - offset) - params[:, 5],
                )
            ),
            params[:, 0] * np.exp(1j * params[:, 1])
        )

        return signal

    @logger
    def cupid_spectrum(
        self,
        indices: Optional[Iterable[int]] = None,
        pts: Optional[int] = None,
        _log: bool = True,
    ) -> np.ndarray:
        """Generate a homodecoupled spectrum according to the CUPID method.

        This generates an FID using :py:meth:`cupid_signal`, halves the first point,
        and applies FT.

        Parameters
        ----------
        indices
            See :ref:`INDICES`.

        pts
            The number of points to construct the signal from. If ``None``,
            ``self.default_pts`` will be used.

        _log
            Ignore this!
        """
        self._check_results_exist()
        sanity_check(
            self._indices_check(indices),
            ("pts", pts, sfuncs.check_int, (), {"min_value": 1}, True),
        )

        fid = self.cupid_signal(indices=indices, pts=pts)
        fid[0] *= 0.5
        return sig.ft(fid)

    @logger
    def predict_multiplets(
        self,
        indices: Optional[Iterable[int]] = None,
        thold: Optional[float] = None,
        freq_unit: str = "hz",
        rm_spurious: bool = False,
        _log: bool = True,
        **estimate_kwargs,
    ) -> Dict[float, Iterable[int]]:
        r"""Predict the estimated oscillators which correspond to each multiplet
        in the signal.

        Parameters
        ----------
        indices
            See :ref:`INDICES`.

        thold
            Frequency threshold for multiplet prediction. All oscillators that make
            up a multiplet are assumed to obey the following expression:

            .. math::
                f_c - f_t < f^{(2)} - f^{(1)} < f_c + f_t

            where :math:`f_c` is the central frequency of the multiplet, and `f_t` is
            the threshold.

        freq_unit
            Must be ``"hz"`` or ``"ppm"``.

        rm_spurious
            If set to ``True``, all oscillators which fall into the following criteria
            will be purged:

            * The oscillator is the only member in a multiplet set.
            * The oscillator's frequency in F1 has a magnitude greater than
              ``thold`` (i.e. the indirect-dimension frequency is sufficiently
              far from 0Hz)

        _log
            Ignore me!

        estimate_kwargs
            If ``rm_suprious`` is ``True``, and oscillators are purged,
            optimisation isrun. Kwargs are given to :py:meth:estimate:\.

        Returns
        -------
        Dict[float, Iterable[int]]
            A dictionary with keys as the multiplet's central frequency, and values
            as a list of oscillator indices which make up the multiplet.
        """
        self._check_results_exist()
        sanity_check(
            self._indices_check(indices),
            ("thold", thold, sfuncs.check_float, (), {"greater_than_zero": True}, True),
            ("freq_unit", freq_unit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
            ("rm_spurious", rm_spurious, sfuncs.check_bool),
        )
        if thold is None:
            thold = self.default_multiplet_thold

        params = self.get_params(indices)
        multiplets = {}
        in_range = lambda f, g: (g - thold < f < g + thold)
        for i, osc in enumerate(params):
            centre_freq = osc[3] - osc[2]
            assigned = False
            for freq in multiplets:
                if in_range(centre_freq, freq):
                    multiplets[freq].append(i)
                    assigned = True
                    break
            if not assigned:
                multiplets[centre_freq] = [i]

        # Set center freqs to average f2 - f1 in the multiplet
        for old_freq, mp_indices in list(multiplets.items()):
            new_freq = np.mean(params[mp_indices, 3] - params[mp_indices, 2])
            multiplets[new_freq] = multiplets.pop(old_freq)

        # Remove spurious opscillators, if requested
        if rm_spurious:
            spurious = {}
            for oscs in multiplets.values():
                if len(oscs) == 1:
                    osc = oscs[0]
                    f1 = params[osc, 2]
                    if abs(f1) > thold:
                        # osc_loc is a tuple of the form (result_index, osc_index)
                        osc_loc = self.find_osc(params[osc])
                        if osc_loc[0] in spurious:
                            spurious[osc_loc[0]].append(osc_loc[1])
                        else:
                            spurious[osc_loc[0]] = [osc_loc[1]]

            for res_idx, osc_idx in spurious.items():
                self.edit_result(index=res_idx, rm_oscs=osc_idx, **estimate_kwargs)

        factor = 1. if freq_unit == "hz" else self.sfo[-1]
        multiplets = {
            freq / factor: indices
            for freq, indices in sorted(multiplets.items(), key=lambda item: item[0])
        }

        return multiplets

    def get_multiplet_integrals(
        self,
        scale: bool = True,
        **kwargs,
    ) -> Dict[float, float]:
        """Get integrals of multiplets assigned using :py:meth:`predict_multiplets`.

        Parameters
        ----------
        scale
            If ``True``, the integrals are scaled so that the smallest integral is 1.

        kwargs
            Keyword arguments for :py:meth:`predict_multiplet_integrals`.
        """
        self._check_results_exist()
        sanity_check(("scale", scale, sfuncs.check_bool))

        multiplets = self.predict_multiplets(**kwargs)
        indices = self._process_indices(kwargs.get("indices", None))
        params = self.get_params(indices)
        integrals = {
            freq: sum(self.oscillator_integrals(params[mp]))
            for freq, mp in list(multiplets.items())
        }

        if scale:
            min_integral = min(list(integrals.values()))
            integrals = {
                freq: integral / min_integral
                for freq, integral in list(integrals.items())
            }

        return integrals

    @logger
    def sheared_signal(
        self,
        indices: Optional[Iterable[int]] = None,
        pts: Optional[Tuple[int, int]] = None,
        indirect_modulation: Optional[str] = None,
    ) -> np.ndarray:
        r"""Return an FID where direct dimension frequencies are perturbed such that:

        .. math::

            f^{(2)}_m = f^{(2)}_m - f^{(1)}_m

        This should yeild a signal where all components in a multiplet are centered
        at the spin's chemical shift in the direct dimenion, akin to performing
        a 45° tilt.

        Parameters
        ----------
        indices
            See :ref:`INDICES`.

        pts
            The number of points to construct the signal from. If ``None``,
            ``self.default_pts`` will be used.

        indirect_modulation
            Acquisition mode in the indirect dimension.

            * ``None`` - hypercomplex dataset:

              .. math::

                  y \left(t_1, t_2\right) = \sum_{m} a_m e^{\mathrm{i} \phi_m}
                  e^{\left(2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
                  e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}

            * ``"amp"`` - amplitude modulated pair:

              .. math::

                  y_{\mathrm{cos}} \left(t_1, t_2\right) = \sum_{m} a_m
                  e^{\mathrm{i} \phi_m} \cos\left(\left(2 \pi \mathrm{i} f_{1,
                  m} - \eta_{1, m}\right) t_1\right) e^{\left(2 \pi \mathrm{i}
                  f_{2, m} - \eta_{2, m}\right) t_2}

              .. math::

                  y_{\mathrm{sin}} \left(t_1, t_2\right) = \sum_{m} a_m
                  e^{\mathrm{i} \phi_m} \sin\left(\left(2 \pi \mathrm{i} f_{1,
                  m} - \eta_{1, m}\right) t_1\right) e^{\left(2 \pi \mathrm{i}
                  f_{2, m} - \eta_{2, m}\right) t_2}

            * ``"phase"`` - phase-modulated pair:

              .. math::

                  y_{\mathrm{P}} \left(t_1, t_2\right) = \sum_{m} a_m
                  e^{\mathrm{i} \phi_m} e^{\left(2 \pi \mathrm{i} f_{1, m} -
                  \eta_{1, m}\right) t_1} e^{\left(2 \pi \mathrm{i} f_{2, m} -
                  \eta_{2, m}\right) t_2}

              .. math::

                  y_{\mathrm{N}} \left(t_1, t_2\right) = \sum_{m} a_m
                  e^{\mathrm{i} \phi_m}
                  e^{\left(-2 \pi \mathrm{i} f_{1, m} - \eta_{1, m}\right) t_1}
                  e^{\left(2 \pi \mathrm{i} f_{2, m} - \eta_{2, m}\right) t_2}

            ``None`` will lead to an array of shape ``(n1, n2)``. ``amp`` and ``phase``
            will lead to an array of shape ``(2, n1, n2)``, with ``fid[0]`` and
            ``fid[1]`` being the two components of the pair.

        See also
        --------
        * For converting amplitude-modulated data to spectral data, see
          :py:func:`nmrespy.sig.proc_amp_modulated`
        * For converting phase-modulated data to spectral data, see
          :py:func:`nmrespy.sig.proc_phase_modulated`
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

        return self.make_fid(
            edited_params, pts=pts, indirect_modulation=indirect_modulation,
        )

    def construct_multiplet_fids(
        self,
        indices: Optional[Iterable[int]] = None,
        pts: Optional[int] = None,
        thold: Optional[float] = None,
        freq_unit: str = "hz",
    ) -> Iterable[np.ndarray]:
        """Generate a list of FIDs corresponding to each multiplet structure.

        Parameters
        ----------
        indices
            See :ref:`INDICES`.

        pts
            The number of points to construct the mutliplets from.

        thold
            Frequency threshold for multiplet prediction. All oscillators that make
            up a multiplet are assumed to obey the following expression:

            .. math::
                f_c - f_t < f^{(2)} - f^{(1)} < f_c + f_t

            where :math:`f_c` is the central frequency of the multiplet, and `f_t` is
            ``thold``

        freq_unit
            Must be ``"hz"`` or ``"ppm"``.

        Returns
        -------
        List of numpy arrays with each FID.
        """
        # TODO: CHECKING
        multiplets = self.predict_multiplets(
            indices=indices,
            thold=thold,
            freq_unit=freq_unit,
        )

        # Sort by frequency
        multiplets = sorted(list(multiplets.items()), key=lambda item: item[0])
        full_params = self.get_params(indices=indices)[:, [0, 1, 3, 5]]
        expinfo_direct = self.expinfo_direct

        fids = []
        for (_, idx) in multiplets:
            params = full_params[idx]
            fids.append(
                expinfo_direct.make_fid(
                    params=params,
                    pts=pts,
                )
            )

        return fids

    def write_multiplets_to_bruker(
        self,
        path: Union[Path, str],
        expno_prefix: Optional[int] = None,
        indices: Optional[Iterable[int]] = None,
        pts: Optional[int] = None,
        thold: Optional[float] = None,
        force_overwrite: bool = False,
    ) -> None:
        """Write each individual multiplet structure to a Bruker data directory.

        Each multiplet is saved to a directory of the form
        ``<path>/<expno_prefix><x>/pdata/1`` where ``<x>`` is iterated from 1 onwards.

        Parameters
        ----------
        path
            The path to the root directory to store the data in.

        expinfo_prefix
            Prefix to the experiment numbers for storing the multiplets to. If
            ``None``, experiments will be numbered ``1``, ``2``, etc.

        indices
            See :ref:`INDICES`.

        pts
            The number of points to construct the mutliplets from.

        thold
            Frequency threshold for multiplet prediction. All oscillators that make
            up a multiplet are assumed to obey the following expression:

            .. math::
                f_c - f_t < f^{(2)} - f^{(1)} < f_c + f_t

            where :math:`f_c` is the central frequency of the multiplet, and `f_t` is
            ``thold``

        force_overwite
            If ``False``, if any directories that will be written to already exist,
            you will be promted if you are happy to overwrite. If ``True``,
            overwriting will take place without asking.
        """
        self._check_results_exist()
        sanity_check(
            ("path", path, check_saveable_dir, (True,)),
            (
                "expno_prefix", expno_prefix, sfuncs.check_int, (), {"min_value": 1},
                True,
            ),
            self._indices_check(indices),
            # Not a "normal" pts check as checking for valid 1D value rather than 2D
            (
                "pts", pts, sfuncs.check_int, (), {"min_value": 1},
                self.default_pts is not None,
            ),
            ("thold", thold, sfuncs.check_float, (), {"greater_than_zero": True}, True),
        )

        path = Path(path).expanduser()
        fids = self.construct_multiplet_fids(
            indices=indices,
            pts=pts,
            thold=thold,
        )

        # Establish list of expno names
        n = len(fids)
        if expno_prefix is None:
            first = 1
        else:
            ndigits = int(np.log10(n)) + 1
            first = expno_prefix * (10 ** ndigits) + 1
        expnos = list(range(first, first + n))

        if not force_overwrite:
            for expno in expnos:
                sanity_check(
                    (
                        "expno_prefix", path / str(expno), check_saveable_dir, (False,),
                    ),
                )

        expinfo_1d = self.expinfo_direct
        for (fid, expno) in zip(fids, expnos):
            expinfo_1d.write_to_bruker(
                fid, path, expno=expno, procno=1, force_overwrite=True,
            )

        print(
            f"{GRE}Saved multiplets to folders {path}/[{expnos[0]}-{expnos[-1]}]/"
            f"{END}"
        )

    def write_cupid_to_bruker(
        self,
        path: Union[Path, str],
        expno: Optional[int] = None,
        indices: Optional[Iterable[int]] = None,
        pts: Optional[int] = None,
        force_overwrite: bool = False,
    ) -> None:
        """Write the signal generated by :py:meth:`cupid_signal` to a Bruker dataset.

        The dataset is saved to a directory of the form ``<path>/<expno>``

        Parameters
        ----------
        path
            The path to the root directory to store the data in. This must already
            exist.

        expno
            The experiment number. If ``None``, the first directory number
            ``<x>`` for which ``<path>/<x>/`` isn;t currently a directory will be
            used.

        indices
            See :ref:`INDICES`.

        pts
            The number of points to construct the dataset from.

        force_overwite
            If ``False``, and ``<path>/<expno>/`` already exists, the user will
            be asked if they are happy to overwrite. If ``True``, overwriting
            will take place without asking.
        """
        self._check_results_exist()
        sanity_check(
            ("path", path, check_saveable_dir, (True,)),
            (
                "expno", expno, sfuncs.check_int, (), {"min_value": 1}, True,
            ),
            self._indices_check(indices),
            # Not a "normal" pts check as checking for valid 1D value rather than 2D
            (
                "pts", pts, sfuncs.check_int, (), {"min_value": 1},
                self.default_pts is not None,
            ),
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
        )

        fid = self.cupid_signal(indices=indices, pts=pts, _log=False)
        expinfo_1d = self.expinfo_direct
        expinfo_1d.write_to_bruker(
            fid, path, expno=expno, procno=1, force_overwrite=force_overwrite,
        )
        print(f"{GRE}Saved CUPID signal to {path}/{expno}/{END}")

    @logger
    def plot_result(
        self,
        indices: Optional[Iterable[int]] = None,
        multiplet_thold: Optional[float] = None,
        high_resolution_pts: Optional[int] = None,
        ratio_1d_2d: Tuple[float, float] = (2., 1.),
        region_unit: str = "hz",
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
        jres_sinebell: bool = True,
        multiplet_colors: Any = None,
        multiplet_lw: float = 1.,
        multiplet_vertical_shift: float = 0.,
        multiplet_show_center_freq: bool = True,
        multiplet_show_45: bool = True,
        marker_size: float = 3.,
        marker_shape: str = "o",
        label_peaks: bool = False,
        denote_regions: bool = False,
        **kwargs,
    ) -> Tuple[mpl.figure.Figure, np.ndarray[mpl.axes.Axes]]:
        r"""Generate a figure of the estimation result.

        The figure includes a contour plot of the 2DJ spectrum, a 1D plot of the
        first slice through the indirect dimension, plots of estimated multiplets,
        and a plot of :py:meth:`cupid_spectrum`.

        Parameters
        ----------
        indices
            See :ref:`INDICES`.

        multiplet_thold
            Frequency threshold for multiplet prediction. All oscillators that make
            up a multiplet are assumed to obey the following expression:

            .. math::
                f_c - f_t < f^{(2)} - f^{(1)} < f_c + f_t

            where :math:`f_c` is the central frequency of the multiplet, and `f_t` is
            the threshold.

        high_resolution_pts
            Indicates the number of points used to generate the multiplet structures
            and :py:meth:`cupid_spectrum`. Should be greater than or equal to
            ``self.default_pts[1]``.

        ratio_1d_2d
            The relative heights of the regions containing the 1D spectra and the
            2DJ spectrum.

        axes_left
            The position of the left edge of the axes, in `figure coordinates
            <https://matplotlib.org/stable/tutorials/advanced/\
            transforms_tutorial.html>`_\. Should be between ``0.`` and ``1.``.

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
            only slightly larger than ``0.``.

        xaxis_ticks
            See :ref:`XAXIS_TICKS`.

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

        jres_sinebell
            If ``True``, applies sine-bell apodisation to the 2DJ spectrum.

        multiplet_colors
            Describes how to color multiplets. See :ref:`COLOR_CYCLE` for options.

        multiplet_lw
            Line width of multiplet spectra

        multiplet_vertical_shift
            The vertical displacement of adjacent mutliplets, as a multiple of
            ``mutliplet_lw``. Set to ``0.`` if you want all mutliplets to lie on the
            same line.

        multiplet_show_center_freq
            If ``True``, lines are plotted on the 2DJ spectrum indicating the central
            frequency of each mutliplet.

        multiplet_show_45
            If ``True``, lines are plotted on the 2DJ spectrum indicating the 45° line
            along which peaks lie in each multiplet.

        marker_size
            The size of markers indicating positions of peaks on the 2DJ contour plot.

        marker_shape
            The `shape of markers <https://matplotlib.org/stable/api/markers_api.html>`_
            indicating positions of peaks on the 2DJ contour plot.

        kwargs
            Keyword arguments provided to `matplotlib.pyplot.figure
            <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html\
            #matplotlib.pyplot.figure>`_\. Allowed arguments include
            ``figsize``, ``facecolor``, ``edgecolor``, etc.

        Returns
        -------
        fig
            The result figure. This can be saved to various formats using the
            `savefig <https://matplotlib.org/stable/api/figure_api.html\
            #matplotlib.figure.Figure.savefig>`_ method.

        axs
            A ``(2, N)`` NumPy array of the axes used for plotting. The first row
            of axes contain the 1D plots. The second row contain the 2DJ
            contour plots.
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
            self._funit_check(region_unit, "region_unit"),
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
            ("jres_sinebell", jres_sinebell, sfuncs.check_bool),
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

        indices = self._process_indices(indices)
        regions = sorted(
            [
                (i, result.get_region(unit=region_unit)[1])
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
            **kwargs,
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

        expinfo_1d = self.expinfo_direct
        expinfo_1d_highres = copy.deepcopy(expinfo_1d)
        expinfo_1d_highres.default_pts = (high_resolution_pts,)
        full_shifts_1d, = expinfo_1d.get_shifts(unit=region_unit)
        full_shifts_1d_highres, = expinfo_1d_highres.get_shifts(unit=region_unit)
        full_shifts_2d_y, full_shifts_2d_x = self.get_shifts(unit=region_unit)
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

        conv = f"{region_unit}->idx"
        full_spectrum = np.abs(
            self.spectrum_sinebell if jres_sinebell else self.spectrum
        ).real

        for idx, region in zip(merge_indices, merge_regions):
            slice_ = slice(*expinfo_1d.convert([region], conv)[0])
            highres_slice = slice(*expinfo_1d_highres.convert([region], conv)[0])

            shifts_2d.append(
                (full_shifts_2d_x[:, slice_], full_shifts_2d_y[:, slice_])
            )
            shifts_1d.append(full_shifts_1d[slice_])
            shifts_1d_highres.append(full_shifts_1d_highres[highres_slice])

            spectra_2d.append(np.abs(full_spectrum).real[:, slice_])
            spectra_1d.append(self.spectrum_first_direct.real[slice_])
            neg_45_spectra.append(
                self.cupid_spectrum(
                    indices=idx, pts=high_resolution_pts, _log=False,
                ).real[highres_slice]
            )

            params = self.get_params(indices=idx)
            multiplet_indices.append(
                list(
                    reversed(
                        self.predict_multiplets(
                            indices=idx, thold=multiplet_thold, _log=False,
                        ).values()
                    )
                )
            )

            multiplet_params = [params[i] for i in multiplet_indices[-1]]
            f1_f2_region = []
            center_freq = []
            for multiplet_param in multiplet_params:
                f1, f2 = multiplet_param[:, [2, 3]].T
                cf = np.mean(f2 - f1)
                f2 = f2 / sfo if region_unit == "ppm" else f2
                cf = cf / sfo if region_unit == "ppm" else cf
                center_freq.append(cf)
                f1_f2_region.append((f1, f2))

                multiplet = expinfo_1d.make_fid(
                    multiplet_param[:, [0, 1, 3, 5]],
                    pts=high_resolution_pts,
                )
                multiplet[0] *= 0.5
                multiplet_spectra.append(sig.ft(multiplet).real)

            f1_f2.append(f1_f2_region)
            center_freqs.append(center_freq)

        print(center_freqs)
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
                    edgecolor="none",
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
                            clip_on=True,
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
                        [cf + lim / (sfo if region_unit == "ppm" else 1.)
                         for lim in ylim1],
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

        self._configure_axes(
            fig,
            axs,
            merge_regions,
            xaxis_ticks,
            axes_left,
            axes_right,
            xaxis_label_height,
            region_unit,
        )
        for ax in axs[0]:
            ax.set_ylim(ylim0)
        for ax in axs[1]:
            ax.set_ylim(ylim1)
        axs[1, 0].set_ylabel("Hz")

        return fig, axs

    def edit_result(
        self,
        index: int = -1,
        add_oscs: Optional[np.ndarray] = None,
        rm_oscs: Optional[Iterable[int]] = None,
        merge_oscs: Optional[Iterable[Iterable[int]]] = None,
        split_oscs: Optional[Dict[int, Optional[Dict]]] = None,
        mirror_oscs: Optional[Iterable[int]] = None,
        **estimate_kwargs,
    ) -> None:
        r"""Manipulate an estimation result. After the result has been changed,
        it is subjected to optimisation.

        There are five types of edit that you can make:

        * *Add* new oscillators with defined parameters.
        * *Remove* oscillators.
        * *Merge* multiple oscillators into a single oscillator.
        * *Split* an oscillator into many oscillators.
        * **Unique to 2DJ**: *Mirror* an oscillator. This allows you add a new
          oscillator with the same parameters as an osciallator in the result,
          except with the following frequencies:

            .. math::

                f^{(1)}_{\text{new}} = -f^{(1)}_{\text{old}}

            .. math::

                f^{(2)}_{\text{new}} = f^{(2)}_{\text{old}} - f^{(1)}_{\text{old}}

        Parameters
        ----------
        index
            See :ref:`INDEX`.

        add_oscs
            The parameters of new oscillators to be added. Should be of shape
            ``(n, 2 * (1 + self.dim))``, where ``n`` is the number of new
            oscillators to add. Even when one oscillator is being added this
            should be a 2D array, i.e.

            * 1D data:

                .. code::

                    params = np.array([[a, φ, f, η]])

            * 2D data:

                .. code::

                    params = np.array([[a, φ, f₁, f₂, η₁, η₂]])

        rm_oscs
            An iterable of ints for the indices of oscillators to remove from
            the result.

        merge_oscs
            An iterable of iterables. Each sub-iterable denotes the indices of
            oscillators to merge together. For example, ``[[0, 2], [6, 7]]``
            would mean that oscillators 0 and 2 are merged, and oscillators 6
            and 7 are merged. A merge involves removing all the oscillators,
            and creating a new oscillator with the sum of amplitudes, and the
            average of phases, freqeuncies and damping factors.

        split_oscs
            A dictionary with ints as keys, denoting the oscillators to split.
            The values should themselves be dicts, with the following permitted
            key/value pairs:

            * ``"separation"`` - An list of length equal to ``self.dim``.
              Indicates the frequency separation of the split oscillators in Hz.
              If not specified, this will be the spectral resolution in each
              dimension.
            * ``"number"`` - An int indicating how many oscillators to split
              into. If not specified, this will be ``2``.
            * ``"amp_ratio"`` A list of floats with length equal to the number of
              oscillators to be split into (see ``"number"``). Specifies the
              relative amplitudes of the oscillators. If not specified, the amplitudes
              will be equal.

            As an example for a 1D estimator:

            .. code::

                split_oscs = {
                    2: {
                        "separation": 1.,  # if 1D, don't need a list
                    },
                    5: {
                        "number": 3,
                        "amp_ratio": [1., 2., 1.],
                    },
                }

            Here, 2 oscillators will be split.

            * Oscillator 2 will be split into 2 (default) oscillators with
              equal amplitude (default). These will be separated by 1Hz.
            * Oscillator 5 will be split into 3 oscillators with relative
              amplitudes 1:2:1. These will be separated by ``self.sw()[0] /
              self.default_pts()[0]`` Hz (default).

        mirror_oscs
            An interable of oscillators to mirror (see the description above).

        estimate_kwargs
            Keyword arguments to provide to the call to :py:meth:`estimate`. Note
            that ``"initial_guess"`` and ``"region_unit"`` are set internally and
            will be ignored if given.
        """
        sanity_check(self._index_check(index))
        index, = self._process_indices([index])
        result, = self.get_results(indices=[index])
        params = result.get_params()
        max_osc_idx = len(params) - 1
        sanity_check(
            (
                "mirror_oscs", mirror_oscs, sfuncs.check_int_list, (),
                {"min_value": 0, "max_value": max_osc_idx}, True,
            ),
        )
        if mirror_oscs is not None:
            to_mirror = params[mirror_oscs]
            mirrored = copy.deepcopy(to_mirror)
            mirrored[:, 2] = -mirrored[:, 2]
            mirrored[:, 3] += mirrored[:, 2]

            if isinstance(add_oscs, np.ndarray):
                add_oscs = np.vstack((add_oscs, mirrored))
            else:
                add_oscs = mirrored

        super().edit_result(
            index, add_oscs, rm_oscs, merge_oscs, split_oscs, **estimate_kwargs,
        )
