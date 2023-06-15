# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Sun 11 Jun 2023 19:20:33 BST

from __future__ import annotations
import copy
import datetime
import functools
import io
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

import nmrespy as ne
from nmrespy.freqfilter import Filter
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import nonlinear_programming
from nmrespy._colors import RED, GRE, END, USE_COLORAMA
from nmrespy._files import (
    cd,
    check_saveable_path,
    check_existent_path,
    configure_path,
    open_file,
    save_file,
)
from nmrespy._paths_and_links import SPINACHPATH
from nmrespy._result_fetcher import ResultFetcher
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy.write import ResultWriter
from nmrespy.write.textfile import experiment_info, titled_table

if USE_COLORAMA:
    import colorama
    colorama.init()

if ne.MATLAB_AVAILABLE:
    import matlab
    import matlab.engine


def logger(f: callable):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        class_instance = args[0]
        if "_log" in kwargs:
            if not kwargs["_log"]:
                return f(*args, **kwargs)
            else:
                del kwargs["_log"]
        class_instance._log += f"--> `{f.__name__}` {args[1:]} {kwargs}\n"
        return f(*args, **kwargs)
    return inner


class Estimator(ne.ExpInfo):
    """Base estimation class."""

    # Each child of Estimator should have the following class attributes:
    # dim
    # twodim_dtype
    # proc_dims
    # ft_dims
    # default_mpm_trim
    # default_nlp_trim
    # default_max_iterations_exact_hessian
    # default_max_iterations_gn_hessian

    def __init__(
        self,
        data: np.ndarray,
        expinfo: ne.ExpInfo,
        datapath: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        data
            The data associated with the binary file in `path`.

        datapath
            The path to the directory containing the NMR data.

        expinfo
            Experiment information.
        """
        self._data = data
        self._datapath = datapath
        if hasattr(expinfo, "parameters"):
            self._bruker_params = expinfo.parameters

        # Deal with 2D amp- and phase modulated pairs
        shape = (
            self._data.shape if self._data.ndim == expinfo.dim
            else self._data.shape[1:]
        )
        super().__init__(
            dim=expinfo.dim,
            sw=expinfo.sw(),
            offset=expinfo.offset(),
            sfo=expinfo.sfo,
            nuclei=expinfo.nuclei,
            default_pts=shape,
            fn_mode=expinfo.fn_mode,
        )

        self._results = []
        now = datetime.datetime.now().strftime('%d-%m-%y %H:%M:%S')
        self._log = (
            "=====================\n"
            "Logfile for Estimator\n"
            "=====================\n"
            f"--> Created @ {now}\n"
        )

    def __str__(self) -> str:
        writer = ResultWriter(
            self.expinfo,
            [params for params in self.get_params(merge=False)]
            if self._results else None,
            [errors for errors in self.get_errors(merge=False)]
            if self._results else None,
            None,
        )
        acqu_table = experiment_info(writer._construct_experiment_info(sig_figs=5))
        if self._results:
            titles = [
                f"{r.region[0][0]:.2f} - {r.region[0][1]:.2f}Hz"
                if r.region is not None else "Full signal"
                for r in self.get_results()
            ]
            param_tables = "\n\n" + "\n\n".join([
                titled_table(title, params) for title, params in zip(
                    titles,
                    writer._construct_parameters(
                        sig_figs=5, sci_lims=(-2, 3), integral_mode="relative",
                    )
                )
            ])
        else:
            param_tables = "\n\nNo estimation performed yet."

        return (
            f"<{self.__class__.__name__} object at {hex(id(self))}>\n\n"
            f"{acqu_table}{param_tables}"
        )

    def _check_results_exist(self) -> None:
        if not self._results:
            raise ValueError(f"{RED}No estimation has been carried out yet!{END}")

    @staticmethod
    def _run_spinach(
        func: str,
        *args,
        to_int: Optional[Iterable[int]] = None,
        to_double: Optional[Iterable[int]] = None,
    ) -> np.ndarray:
        if not ne.MATLAB_AVAILABLE:
            raise NotImplementedError(
                f"{RED}MATLAB isn't accessible to Python. To get up and running, "
                "take at look here:\n"
                "https://www.mathworks.com/help/matlab/matlab_external/"
                f"install-the-matlab-engine-for-python.html{END}"
            )

        with cd(SPINACHPATH):
            devnull = io.StringIO(str(os.devnull))
            try:
                eng = matlab.engine.start_matlab()
                args = list(args)
                to_double = [] if to_double is None else to_double
                to_int = [] if to_int is None else to_int
                for i in to_double:
                    args[i] = matlab.double([args[i]])
                for i in to_int:
                    args[i] = matlab.int32([args[i]])
                fid, sfo = eng.__getattr__(func).__call__(
                    *args, nargout=2, stdout=devnull, stderr=devnull,
                )
            except matlab.engine.MatlabExecutionError:
                raise ValueError(
                    f"{RED}Something went wrong in trying to run Spinach.\n"
                    "Read what is stated below the line "
                    "\"matlab.engine.MatlabExecutionError:\" "
                    f"for more details on the error raised.{END}"
                )

        return fid, sfo

    @property
    def bruker_params(self) -> Optional[dict]:
        """Return a dictionary of Bruker parameters.

        If the class instance was generated by :py:meth:`new_bruker`, a
        dictionary of experiment parameters will be returned. Otherwise,
        ``None`` will be returned.
        """
        if hasattr(self, "_bruker_params"):
            return self._bruker_params
        else:
            return None

    @property
    def expinfo(self) -> ne.ExpInfo:
        return ne.ExpInfo(
            self.dim,
            self.sw(),
            self.offset(),
            self.sfo,
            self.nuclei,
            self.default_pts,
            self.fn_mode,
        )

    @property
    def expinfo_direct(self) -> ne.ExpInfo:
        """Generate a 1D :py:meth:`~nmrespy.ExpInfo` object with parameters
        related to the direct dimension.
        """
        return ne.ExpInfo(
            dim=1,
            sw=self.sw()[-1],
            offset=self.offset()[-1],
            sfo=self.sfo[-1],
            nuclei=self.nuclei[-1],
            default_pts=self.default_pts[-1],
        )

    # DATA RETREVAL METHODS
    @property
    def data(self) -> np.ndarray:
        """Return the data associated with the estimator."""
        return self._data

    @property
    def spectrum(self) -> np.ndarray:
        """Return the spectrum associated with the estimator."""
        data = copy.deepcopy(self.data)
        slice_ = tuple([slice(0, 1, None) for _ in range(self.dim)])
        data[slice_] *= 0.5
        return ne.sig.ft(data, axes=self.ft_dims)

    @property
    def data_direct(self) -> np.ndarray:
        """Generate a 1D FID of the first signal in the direct dimension."""
        slice_ = [slice(0, 1, None) for _ in range(self.dim - 1)]
        slice_ += [slice(None, None, None)]
        slice_ = tuple(slice_)
        return self.data[slice_].flatten()

    @property
    def spectrum_direct(self) -> np.ndarray:
        """Generate a 1D spectrum of the first signal in the direct dimension."""
        data_direct = self.data_direct
        data_direct[0] *= 0.5
        return ne.sig.ft(data_direct)

    def get_log(self) -> str:
        """Get the log for the estimator instance."""
        return self._log

    def save_log(
        self,
        path: Union[str, Path] = "./espy_logfile",
        force_overwrite: bool = False,
        fprint: bool = True,
    ) -> None:
        """Save the estimator's log.

        Parameters
        ----------
        path
            The path to save the log to.

        force_overwrite
            If ``path`` already exists and ``force_overwrite`` is set to ``False``,
            the user will be asked to confirm whether they are happy to
            overwrite the file. If ``True``, the file will be overwritten
            without prompt.

        fprint
            Specifies whether or not to print infomation to the terminal.
        """
        sanity_check(
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
            ("fprint", fprint, sfuncs.check_bool),
        )
        sanity_check(
            ("path", path, check_saveable_path, ("log", force_overwrite)),
        )

        path = configure_path(path, "log")
        save_file(self.get_log(), path, fprint=fprint)

    # LOADING/SAVING ESTIMATORS
    @logger
    def to_pickle(
        self,
        path: Optional[Union[Path, str]] = None,
        force_overwrite: bool = False,
        fprint: bool = True,
    ) -> None:
        """Save the estimator to a byte stream using Python's pickling protocol.

        Parameters
        ----------
        path
            Path of file to save the byte stream to. Do not include the
            ``'".pkl"`` suffix. If ``None``, ``./estimator_<x>.pkl`` will be
            used, where ``<x>`` is the first number that doesn't cause a clash
            with an already existent file.

        force_overwrite
            Defines behaviour if the specified path already exists:

            * If ``False``, the user will be prompted if they are happy
              overwriting the current file.
            * If ``True``, the current file will be overwritten without prompt.

        fprint
            Specifies whether or not to print infomation to the terminal.

        See Also
        --------
        :py:meth:`from_pickle`
        """
        sanity_check(
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
            ("fprint", fprint, sfuncs.check_bool),
        )
        sanity_check(
            ("path", path, check_saveable_path, ("pkl", force_overwrite), {}, True),
        )

        if path is None:
            x = 1
            while True:
                path = Path(f"estimator_{x}.pkl").resolve()
                if path.is_file():
                    x += 1
                else:
                    break

        path = configure_path(path, "pkl")
        save_file(self, path, binary=True, fprint=fprint)

    @classmethod
    def from_pickle(
        cls,
        path: Union[str, Path],
    ) -> Estimator:
        """Load a pickled estimator instance.

        .. warning::
           `From the Python docs:`

           *"The pickle module is not secure. Only unpickle data you trust.
           It is possible to construct malicious pickle data which will
           execute arbitrary code during unpickling. Never unpickle data
           that could have come from an untrusted source, or that could have
           been tampered with."*

           You should only use ``from_pickle`` on files that you are 100%
           certain were generated using :py:meth:`to_pickle`. If you load
           pickled data from a .pkl file, and the resulting output is not an
           estimator object, an error will be raised.

        Parameters
        ----------
        path
            The path to the pickle file. Do not include the ``.pkl`` suffix.
        """
        sanity_check(("path", path, check_existent_path, ("pkl",)))
        path = configure_path(path, "pkl")
        obj = open_file(path, binary=True)

        if isinstance(obj, __class__):
            return obj
        else:
            raise TypeError(
                f"{RED}It is expected that the object loaded by"
                " `from_pickle` is an instance of"
                f" {__class__.__module__}.{__class__.__qualname__}."
                f" What was loaded didn't satisfy this!{END}"
            )

    # PRE-PROCESSING METHODS
    def exp_apodisation(self, k: Iterable[float]):
        """Apply an exponential window function to the direct dimnsion of the data.

        Parameters
        ----------
        k
            Line-broadening factor for each dimension.
        """
        sanity_check(
            (
                "k", k, sfuncs.check_float_list, (),
                {"length": len(self.proc_dims), "len_one_can_be_listless": True}
            )
        )
        if isinstance(k, float):
            k = [k]

        for i, lb in zip(self.proc_dims, k):
            self._data = ne.sig.exp_apodisation(self._data, lb, axes=[i])

    def phase_data(self, p0: float = 0., p1: float = 0., pivot: int = 0) -> None:
        """Apply a first-order phase correction in the direct dimension.

        Parameters
        ----------
        p0
            Zero-order phase correction, in radians.

        p1
            First-order phase correction, in radians.

        pivot
            Index of the pivot. ``0`` corresponds to the leftmost point in the
            spectrum.

        See also
        --------
        :py:meth:`manual_phase_data`
        """
        if len(self.proc_dims) > 1:
            raise NotImplementedError(
                f"{RED}Not implemented for this data type yet.{END}"
            )

        # TODO: Enable 2D dataset phase correction
        sanity_check(
            ("p0", p0, sfuncs.check_float),
            ("p1", p1, sfuncs.check_float),
            (
                "pivot", pivot, sfuncs.check_int, (),
                {"min_value": 0, "max_value": self.data.shape[self.proc_dims[0]] - 1},
            ),
        )

        p0s = [0. for _ in range(self.dim)]
        p1s = [0. for _ in range(self.dim)]
        pivots = [0. for _ in range(self.dim)]

        p0s[self.proc_dims[0]] = p0
        p1s[self.proc_dims[0]] = p1
        pivots[self.proc_dims[0]] = pivot

        spec = copy.deepcopy(self.data)
        spec[self._first_point_slice] *= 0.5
        spec = ne.sig.ft(spec, axes=self.proc_dims)

        self._data = ne.sig.ift(
            ne.sig.phase(spec, p0=p0s, p1=p1s, pivot=pivots),
            axes=self.proc_dims,
        )

    def manual_phase_data(
        self,
        max_p1: float = 10 * np.pi,
    ) -> Tuple[float, float]:
        """Manually phase the data using a Graphical User Interface.

        Parameters
        ----------
        max_p1
            The largest permitted first order correction (rad). Set this to a larger
            value than the default (10π) if you anticipate having to apply a
            very large first order correction.

        Returns
        -------
        p0
            Zero order phase (rad)

        p1
            First prder phase (rad)

        See also
        --------
        :py:meth:`phase_data`
        """
        if len(self.proc_dims) > 1:
            raise NotImplementedError(
                f"{RED}Not implemented for this data type yet.{END}"
            )

        sanity_check(
            ("max_p1", max_p1, sfuncs.check_float, (), {"greater_than_zero": True}),
        )
        if self.data.ndim == 1:
            spectrum = self.spectrum
        else:
            spectrum = self.spectrum_direct

        p0, p1 = ne.sig.manual_phase_data(spectrum, max_p1=[max_p1])
        p0, p1 = p0[0], p1[0]
        self.phase_data(p0=p0, p1=p1)
        return p0, p1

    def baseline_correction(self, min_length: int = 50) -> None:
        """Apply baseline correction to the estimator's data.

        The algorithm applied is desribed in [#]_. This uses an implementation
        provided by `pybaselines
        <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/api/index.html#pybaselines.api.Baseline.fabc>`_.

        Parameters
        ----------
        min_length
            *From the pybaseline docs:* Any region of consecutive baseline
            points less than ``min_length`` is considered to be a false
            positive and all points in the region are converted to peak points.
            A higher ``min_length`` ensures less points are falsely assigned as
            baseline points.

        References
        ----------
        .. [#] Cobas, J., et al. A new general-purpose fully automatic
           baseline-correction procedure for 1D and 2D NMR data. Journal of
           Magnetic Resonance, 2006, 183(1), 145-151.
        """
        if len(self.proc_dims) > 1:
            raise NotImplementedError(
                f"{RED}Not implemented for this data type yet.{END}"
            )

        # TODO: Change for case when proc_dims > 1
        shape = self.data.shape
        # Fix
        sanity_check(
            (
                "min_length", min_length, sfuncs.check_int, (),
                {"min_value": 1, "max_value": shape[self.proc_dims[0]]},
            ),
        )

        new_data = np.zeros(shape, dtype="complex128")

        if self.data.ndim == 1:
            data = np.expand_dims(self.data, axis=0)
            new_data = np.expand_dims(new_data, axis=0)
        else:
            data = self.data

        for i, fid in enumerate(data):
            spectrum = ne.sig.ft(ne.sig.make_virtual_echo(fid)).real
            spectrum, _ = ne.sig.baseline_correction(spectrum, min_length=min_length)
            # Fix
            new_data[i] = ne.sig.ift(spectrum)[:shape[self.proc_dims[0]]]

        self._data = new_data[0] if self.data.ndim == 1 else new_data

    @logger
    def estimate(
        self,
        region: Optional[Iterable[Tuple[float, float]]] = None,
        noise_region: Optional[Iterable[Tuple[float, float]]] = None,
        region_unit: str = "hz",
        initial_guess: Optional[Union[np.ndarray, int]] = None,
        mode: str = "apfd",
        amp_thold: Optional[float] = None,
        phase_variance: bool = True,
        cut_ratio: Optional[float] = 1.1,
        mpm_trim: Optional[Iterable[int]] = None,
        nlp_trim: Optional[Iterable[int]] = None,
        hessian: str = "gauss-newton",
        max_iterations: Optional[int] = None,
        negative_amps: str = "remove",
        output_mode: Optional[int] = 10,
        save_trajectory: bool = False,
        epsilon: float = 1.e-8,
        eta: float = 0.15,
        initial_trust_radius: float = 1.0,
        max_trust_radius: float = 4.0,
        check_neg_amps_every: int = 10,
        _log: bool = True,
        **optimiser_kwargs,
    ):
        r"""Estimate a specified region of the signal.

        The basic steps that this method carries out are:

        * (Optional, but highly advised) Generate a frequency-filtered "sub-FID"
          corresponding to a specified region of interest.
        * (Optional) Generate an initial guess using the Minimum Description
          Length (MDL) [#]_ and Matrix Pencil Method (MPM) [#]_ [#]_ [#]_ [#]_
        * Apply numerical optimisation to determine a final estimate of the signal
          parameters. The optimisation routine employed is the Trust Newton Conjugate
          Gradient (NCG) algorithm ([#]_ , Algorithm 7.2).

        Parameters
        ----------
        region
            The frequency range of interest. Should be of the form ``[left, right]``
            where ``left`` and ``right`` are the left and right bounds of the region
            of interest in Hz or ppm (see ``region_unit``). If ``None``, the
            full signal will be considered, though for sufficently large and
            complex signals it is probable that poor and slow performance will
            be realised.

        noise_region
            If ``region`` is not ``None``, this must be of the form ``[left, right]``
            too. This should specify a frequency range where no noticeable signals
            reside, i.e. only noise exists.

        region_unit
            One of ``"hz"`` or ``"ppm"`` Specifies the units that ``region``
            and ``noise_region`` have been given as.

        initial_guess
            * If ``None``, an initial guess will be generated using the MPM
              with the MDL being used to estimate the number of oscillators
              present.
            * If an int, the MPM will be used to compute the initial guess with
              the value given being the number of oscillators.
            * If a NumPy array, this array will be used as the initial guess.

        hessian
            Specifies how to construct the Hessian matrix.

            * If ``"exact"``, the exact Hessian will be used.
            * If ``"gauss-newton"``, the Hessian will be approximated as is
              done with the Gauss-Newton method. See the *"Derivation from
              Newton's method"* section of `this article
              <https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm>`_.

        mode
            A string containing a subset of the characters ``"a"`` (amplitudes),
            ``"p"`` (phases), ``"f"`` (frequencies), and ``"d"`` (damping factors).
            Specifies which types of parameters should be considered for optimisation.
            In most scenarios, you are likely to want the default value, ``"apfd"``.

        amp_thold
            A value that imposes a threshold for deleting oscillators of
            negligible ampltiude.

            * If ``None``, does nothing.
            * If a float, oscillators with amplitudes satisfying :math:`a_m <
              a_{\mathrm{thold}} \lVert \boldsymbol{a} \rVert_2` will be
              removed from the parameter array, where :math:`\lVert
              \boldsymbol{a} \rVert_2` is the Euclidian norm of the vector of
              all the oscillator amplitudes. It is advised to set ``amp_thold``
              at least a couple of orders of magnitude below 1.

        phase_variance
            Whether or not to include the variance of oscillator phases in the cost
            function. This should be set to ``True`` in cases where the signal being
            considered is derived from well-phased data.

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

        max_iterations
            A value specifiying the number of iterations the routine may run
            through before it is terminated. If ``None``, a default number
            of maximum iterations is set, based on the the data dimension and
            the value of ``hessian``.

        negative_amps
            Indicates how to treat oscillators which have gained negative
            amplitudes during the optimisation.

            * ``"remove"`` will result in such oscillators being purged from
              the parameter estimate. The optimisation routine will the be
              re-run recursively until no oscillators have a negative
              amplitude.
            * ``"flip_phase"`` will retain oscillators with negative
              amplitudes, but the the amplitudes will be multiplied by -1,
              and a π radians phase shift will be applied.
            * ``"ignore"`` will do nothing (negative amplitude oscillators will remain).

        output_mode
            Dictates what information is sent to stdout.

            * If ``None``, nothing will be sent.
            * If ``0``, only a message on the outcome of the optimisation will
              be sent.
            * If a positive int ``k``, information on the cost function,
              gradient norm, and trust region radius is sent every kth
              iteration.

        save_trajectory
            If ``True``, a list of parameters at each iteration will be saved, and
            accessible via the ``trajectory`` attribute.

            .. warning:: Not implemented yet!

        epsilon
            Sets the convergence criterion. Convergence will occur when
            :math:`\lVert \boldsymbol{g}_k \rVert_2 < \epsilon`.

        eta
            Criterion for accepting an update. An update will be accepted if
            the ratio of the actual reduction and the predicted reduction is
            greater than ``eta``:

            .. math ::

                \rho_k = \frac{f(x_k) - f(x_k - p_k)}{m_k(0) - m_k(p_k)} > \eta

        initial_trust_radius
            The initial value of the radius of the trust region.

        max_trust_radius
            The largest permitted radius for the trust region.

        check_neg_amps_every
            For every iteration that is a multiple of this, negative amplitudes
            will be checked for and dealt with if found.

        _log
            Ignore this!

        References
        ----------
        .. [#] Yingbo Hua and Tapan K Sarkar. “Matrix pencil method for estimating
           parameters of exponentially damped/undamped sinusoids in noise”. In:
           IEEE Trans. Acoust., Speech, Signal Process. 38.5 (1990), pp. 814–824.

        .. [#] Yung-Ya Lin et al. “A novel detection–estimation scheme for noisy
           NMR signals: applications to delayed acquisition data”. In: J. Magn.
           Reson. 128.1 (1997), pp. 30–41.

        .. [#] Yingbo Hua. “Estimating two-dimensional frequencies by matrix
           enhancement and matrix pencil”. In: [Proceedings] ICASSP 91: 1991
           International Conference on Acoustics, Speech, and Signal Processing.
           IEEE. 1991, pp. 3073–3076.

        .. [#] Fang-Jiong Chen et al. “Estimation of two-dimensional frequencies
           using modified matrix pencil method”. In: IEEE Trans. Signal Process.
           55.2 (2007), pp. 718–724.

        .. [#] M. Wax, T. Kailath, Detection of signals by information theoretic
           criteria, IEEE Transactions on Acoustics, Speech, and Signal Processing
           33 (2) (1985) 387–392.

        .. [#] Jorge Nocedal and Stephen J. Wright. Numerical optimization. 2nd
               ed. Springer series in operations research. New York: Springer,
               2006.
        """
        sanity_check(
            (
                "region_unit", region_unit, sfuncs.check_frequency_unit,
                (self.hz_ppm_valid,),
            ),
            (
                "initial_guess", initial_guess, sfuncs.check_initial_guess,
                (len(self.ft_dims),), {}, True,
            ),
            ("hessian", hessian, sfuncs.check_one_of, ("gauss-newton", "exact")),
            ("phase_variance", phase_variance, sfuncs.check_bool),
            ("mode", mode, sfuncs.check_optimiser_mode),
            (
                "amp_thold", amp_thold, sfuncs.check_float, (),
                {"greater_than_zero": True}, True,
            ),
            (
                "mpm_trim", mpm_trim, sfuncs.check_int_list, (),
                {"min_value": 1, "len_one_can_be_listless": True}, True,
            ),
            (
                "nlp_trim", nlp_trim, sfuncs.check_int_list, (),
                {"min_value": 1, "len_one_can_be_listless": True}, True,
            ),
            (
                "cut_ratio", cut_ratio, sfuncs.check_float, (),
                {"min_value": 1.}, True,
            ),
            (
                "max_iterations", max_iterations, sfuncs.check_int, (),
                {"min_value": 1}, True,
            ),
            (
                "negative_amps", negative_amps, sfuncs.check_one_of,
                ("remove", "flip_phase", "ignore"),
            ),
            ("output_mode", output_mode, sfuncs.check_int, (), {"min_value": 0}, True),
            ("save_trajectory", save_trajectory, sfuncs.check_bool),
            (
                "epsilon", epsilon, sfuncs.check_float, (),
                {"min_value": np.finfo(float).eps},
            ),
            ("eta", eta, sfuncs.check_float, (), {"min_value": 0.0, "max_value": 1.0}),
            (
                "initial_trust_radius", initial_trust_radius, sfuncs.check_float, (),
                {"greater_than_zero": True},
            ),
        )
        sanity_check(
            self._region_check(region, region_unit, "region"),
            self._region_check(noise_region, region_unit, "noise_region"),
            (
                "max_trust_radius", max_trust_radius, sfuncs.check_float, (),
                {"min_value": initial_trust_radius},
            ),
            (
                "check_neg_amps_every", check_neg_amps_every, sfuncs.check_int, (),
                {"min_value": 1, "max_value": max_iterations},
            ),
        )

        region = self._process_region(region)
        noise_region = self._process_region(noise_region)

        if output_mode != 0:
            print(self._estimate_banner(region, region_unit))

        (
            region, noise_region, mpm_expinfo, nlp_expinfo, mpm_signal, nlp_signal,
        ) = self._filter_signal(
            region, noise_region, region_unit, mpm_trim, nlp_trim, cut_ratio,
        )

        if isinstance(initial_guess, np.ndarray):
            x0 = initial_guess
        else:
            oscillators = initial_guess if isinstance(initial_guess, int) else 0

            x0 = MatrixPencil(
                mpm_expinfo,
                mpm_signal,
                oscillators=oscillators,
                output_mode=isinstance(output_mode, int),
            ).get_params()

            if x0 is None:
                self._results.append(
                    Result(
                        np.array([[]]),
                        np.array([[]]),
                        self._reduce_region(region),
                        self._reduce_region(noise_region),
                        self.sfo,
                    )
                )
                return

        if max_iterations is None:
            if hessian == "exact":
                max_iterations = self.default_max_iterations_exact_hessian
            elif hessian == "gauss-newton":
                max_iterations = self.default_max_iterations_gn_hessian

        optimiser_kwargs = {
            "phase_variance": phase_variance,
            "hessian": hessian,
            "mode": mode,
            "amp_thold": amp_thold,
            "max_iterations": max_iterations,
            "negative_amps": negative_amps,
            "output_mode": output_mode,
            "save_trajectory": save_trajectory,
            "epsilon": epsilon,
            "eta": eta,
            "initial_trust_radius": initial_trust_radius,
            "max_trust_radius": max_trust_radius,
            "check_neg_amps_every": check_neg_amps_every,
        }

        self._run_optimisation(
            nlp_expinfo,
            nlp_signal,
            x0,
            region,
            noise_region,
            **optimiser_kwargs,
        )

    def _filter_signal(
        self,
        region: Optional[Iterable[Tuple[float, float]]],
        noise_region: Optional[Iterable[Tuple[float, float]]],
        region_unit: str,
        mpm_trim: Optional[Iterable[int]],
        nlp_trim: Optional[Iterable[int]],
        cut_ratio: Optional[float],
    ):
        # This method is uused by `Estimator1D` and `Estimator2DJ`.
        # It is overwritten by `EstimatorSeq1D`.
        if region is None:
            region_unit = "hz"
            region = self._full_region
            noise_region = None
            mpm_signal = nlp_signal = self.data
            mpm_expinfo = nlp_expinfo = self.expinfo

        else:
            filter_ = Filter(
                self.data,
                self.expinfo,
                region,
                noise_region,
                region_unit=region_unit,
                twodim_dtype=self.twodim_dtype,
            )

            spec, expinfo = filter_.get_filtered_spectrum(cut_ratio=cut_ratio)

            region = filter_.get_region()
            noise_region = filter_.get_noise_region()
            mpm_signal, mpm_expinfo = filter_.get_filtered_fid(cut_ratio=cut_ratio)
            nlp_signal, nlp_expinfo = filter_.get_filtered_fid(cut_ratio=None)

        mpm_slice = self._get_slice("mpm", mpm_trim, mpm_signal.shape)
        mpm_signal = mpm_signal[mpm_slice]
        nlp_slice = self._get_slice("nlp", nlp_trim, nlp_signal.shape)
        nlp_signal = nlp_signal[nlp_slice]

        return (
            region,
            noise_region,
            mpm_expinfo,
            nlp_expinfo,
            mpm_signal,
            nlp_signal,
        )

    def _run_optimisation(
        self,
        nlp_expinfo,
        nlp_signal,
        x0,
        region,
        noise_region,
        **optimiser_kwargs,
    ) -> None:
        # This is called by `Estimator1D` and `Estimator2DJ`, `Estimator2D`.
        # It is overwritten by `EstimatorSeq1D`.
        result = nonlinear_programming(
            nlp_expinfo,
            nlp_signal,
            x0,
            **optimiser_kwargs,
        )

        self._results.append(
            Result(
                result.x,
                result.errors,
                region,
                noise_region,
                self.sfo,
                result.trajectory,
            )
        )

    def make_fid_from_result(
        self,
        indices: Optional[Iterable[int]] = None,
        osc_indices: Optional[Iterable[Iterable[int]]] = None,
        pts: Optional[Iterable[int]] = None,
        indirect_modulation: Optional[str] = None,
    ) -> np.ndarray:
        sanity_check(
            self._indices_check(indices),
            self._pts_check(pts),
        )

        indices = self._process_indices(indices)

        full_params = self.get_params(indices)
        sanity_check(
            (
                "osc_indices", osc_indices, sfuncs.check_int_list, (),
                {
                    "len_one_can_be_listless": True,
                    "min_value": 0,
                    "max_value": full_params.shape[0] - 1,
                },
                True,
            ),
        )

        if osc_indices is None:
            osc_indices = list(range(full_params.shape[0]))
        elif isinstance(osc_indices, int):
            osc_indices = [osc_indices]
        else:
            osc_indices = list(osc_indices)

        if self.dim > 1:
            sanity_check(
                (
                    "indirect_modulation", indirect_modulation,
                    sfuncs.check_one_of, ("amp", "phase"), {}, True
                ),
            )

        params = full_params[osc_indices]
        return self.make_fid(params, pts, indirect_modulation=indirect_modulation)

    def get_results(self, indices: Optional[Iterable[int]] = None) -> Iterable[Result]:
        """Obtain a subset of the estimation results obtained.

        By default, all results are returned, in the order in which they are obtained.

        Parameters
        ----------
        indices
            see :ref:`INDICES`
        """
        self._check_results_exist()
        sanity_check(
            self._indices_check(indices),
        )
        indices = self._process_indices(indices)
        return [self._results[i] for i in indices]

    def get_params(
        self,
        indices: Optional[Iterable[int]] = None,
        merge: bool = True,
        funit: str = "hz",
        sort_by: str = "f-1",
    ) -> Optional[Union[Iterable[np.ndarray], np.ndarray]]:
        """Return estimation result parameters.

        Parameters
        ----------
        indices
            see :ref:`INDICES`

        merge
            * If ``True``, a single array of all parameters will be returned.
            * If ``False``, an iterable of each individual estimation result's
              parameters will be returned.

        funit
            The unit to express frequencies in. Must be one of ``"hz"`` and ``"ppm"``.

        sort_by
            Specifies the parameter by which the oscillators are ordered by.

            Should be one of

            * ``"a"`` for amplitude
            * ``"p"`` for phase
            * ``"f<n>"`` for frequency in the ``<n>``-th dimension
            * ``"d<n>"`` for the damping factor in the ``<n>``-th dimension.

            By setting ``<n>`` to ``-1``, the final (direct) dimension will be
            used. For 1D data, ``"f"`` and ``"d"`` can be used to specify the
            frequency or damping factor.
        """
        self._check_results_exist()
        sanity_check(
            self._indices_check(indices),
            ("merge", merge, sfuncs.check_bool),
            ("funit", funit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
            ("sort_by", sort_by, sfuncs.check_sort_by, (self.dim,)),
        )

        return self._get_arrays("params", indices, funit, sort_by, merge)

    def get_errors(
        self,
        indices: Optional[Iterable[int]] = None,
        merge: bool = True,
        funit: str = "hz",
        sort_by: str = "f-1",
    ) -> Optional[Union[Iterable[np.ndarray], np.ndarray]]:
        """Return estimation result errors.

        Parameters
        ----------
        indices
            see :ref:`INDICES`

        merge
            * If ``True``, a single array of all parameters will be returned.
            * If ``False``, an iterable of each individual estimation result's
              parameters will be returned.

        funit
            The unit to express frequencies in. Must be one of ``"hz"`` and ``"ppm"``.

        sort_by
            Specifies the parameter by which the oscillators are ordered by.

            Note the errors are re-ordered such that they would agree with the
            parameters from :py:meth:`get_params` when given the same ``sort_by``
            argument.

            Should be one of

            * ``"a"`` for amplitude
            * ``"p"`` for phase
            * ``"f<n>"`` for frequency in the ``<n>``-th dimension
            * ``"d<n>"`` for the damping factor in the ``<n>``-th dimension.

            By setting ``<n>`` to ``-1``, the final (direct) dimension will be
            used. For 1D data, ``"f"`` and ``"d"`` can be used to specify the
            frequency or damping factor.
        """
        self._check_results_exist()
        sanity_check(
            self._indices_check(indices),
            ("merge", merge, sfuncs.check_bool),
            ("funit", funit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
            ("sort_by", sort_by, sfuncs.check_sort_by, (self.dim,)),
        )

        return self._get_arrays("errors", indices, funit, sort_by, merge)

    def find_osc(self, params: np.ndarray) -> Tuple[int, int]:
        for i, result in enumerate(self._results):
            result_params = result.get_params()
            try:
                j = int(np.where((result_params == params).all(axis=-1))[0][0])
                return (i, j)
            except IndexError:
                pass
        return None

    def _get_arrays(
        self,
        name: str,
        indices: Optional[Iterable[int]],
        funit: str,
        sort_by: str,
        merge: bool,
    ) -> Optional[np.ndarray]:
        results = self.get_results(indices)
        arrays = [result._get_array(name, funit, sort_by) for result in results]

        if merge:
            array = np.vstack(arrays)
            sort_idx = results[0]._process_sort_by(sort_by, self.dim)

            param_array = np.vstack(
                [
                    result._get_array("params", funit, sort_by)
                    for result in results
                ]
            )

            array = array[np.argsort(param_array[:, sort_idx])]
            return array

        else:
            return arrays

    @logger
    def edit_result(
        self,
        index: int = -1,
        add_oscs: Optional[np.ndarray] = None,
        rm_oscs: Optional[Iterable[int]] = None,
        merge_oscs: Optional[Iterable[Iterable[int]]] = None,
        split_oscs: Optional[Dict[int, Optional[Dict]]] = None,
        **estimate_kwargs,
    ) -> None:
        """Manipulate an estimation result. After the result has been changed,
        it is subjected to optimisation.

        There are four types of edit that you can make:

        * *Add* new oscillators with defined parameters.
        * *Remove* oscillators.
        * *Merge* multiple oscillators into a single oscillator.
        * *Split* an oscillator into many oscillators.

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

        estimate_kwargs
            Keyword arguments to provide to the call to :py:meth:`estimate`. Note
            that ``"initial_guess"`` and ``"region_unit"`` are set internally and
            will be ignored if given.
        """
        self._check_results_exist()
        sanity_check(self._index_check(index))
        index, = self._process_indices([index])
        result, = self.get_results(indices=[index])
        params = result.get_params()
        max_osc_idx = len(params) - 1
        sanity_check(
            (
                "add_oscs", add_oscs, sfuncs.check_parameter_array, (self.dim,), {},
                True,
            ),
            (
                "rm_oscs", rm_oscs, sfuncs.check_int_list, (),
                {"min_value": 0, "max_value": max_osc_idx}, True,
            ),
            (
                "merge_oscs", merge_oscs, sfuncs.check_int_list_list,
                (), {"min_value": 0, "max_value": max_osc_idx}, True,
            ),
            (
                "split_oscs", split_oscs, sfuncs.check_split_oscs,
                (self.dim, max_osc_idx), {}, True,
            ),
        )

        idx_to_remove = []
        oscs_to_add = add_oscs

        if rm_oscs is not None:
            idx_to_remove.extend(rm_oscs)

        if merge_oscs is not None:
            for oscs in merge_oscs:
                new_osc = np.sum(params[oscs], axis=0, keepdims=True)
                new_osc[:, 1:] = new_osc[:, 1:] / float(len(oscs))
                new_osc[:, 1] = (new_osc[:, 1] + np.pi) % (2 * np.pi) - np.pi
                if oscs_to_add is None:
                    oscs_to_add = new_osc
                else:
                    oscs_to_add = np.vstack((oscs_to_add, new_osc))

                idx_to_remove.extend(oscs)

        if split_oscs is not None:
            def_sep = lambda x: self.sw()[x] / self.default_pts[x]
            def_n = 2
            def_amp_ratio = np.array([1, 1])
            def_split_dim = self.dim - 1
            for osc, split_info in split_oscs.items():
                to_split = params[osc]
                if split_info is None:
                    n, amp_ratio, split_dim = \
                        def_n, def_amp_ratio, def_split_dim
                    sep = def_sep(split_dim)
                else:
                    split_dim = self.dim - 1
                    if "separation" in split_info:
                        sep = split_info["separation"]
                    else:
                        sep = def_sep(split_dim)

                    if ("number" not in split_info and "amp_ratio" not in split_info):
                        n = def_n
                        amp_ratio = def_n
                    elif ("number" in split_info and "amp_ratio" not in split_info):
                        n = split_info["number"]
                        amp_ratio = np.ones((n,))
                    elif ("number" not in split_info and "amp_ratio" in split_info):
                        amp_ratio = np.array(split_info["amp_ratio"])
                        n = amp_ratio.size
                    else:
                        n = split_info["number"]
                        amp_ratio = np.array(split_info["amp_ratio"])

                amps = to_split[0] * amp_ratio / amp_ratio.sum()
                # Highest frequency of all the new oscillators
                max_freq = to_split[split_dim + 2] + 0.5 * (n - 1) * sep
                # Array of all frequencies (lowest to highest)
                freqs = np.array(
                    [max_freq - i * sep for i in range(n)],
                    dtype="float64",
                )
                new_oscs = np.zeros((n, 2 * (1 + self.dim)), dtype="float64")
                new_oscs[:, 0] = amps
                new_oscs[:, 1] = to_split[1]
                for i in range(self.dim):
                    if i == split_dim:
                        new_oscs[:, 2 + i] = freqs
                    else:
                        new_oscs[:, 2 + i] = to_split[2 + i]

                new_oscs[:, 2 + self.dim :] = to_split[2 + self.dim :] / len(amps)

                if oscs_to_add is None:
                    oscs_to_add = new_oscs
                else:
                    oscs_to_add = np.vstack((oscs_to_add, new_oscs))

                idx_to_remove.append(osc)

        if idx_to_remove:
            params = np.delete(params, idx_to_remove, axis=0)
        if oscs_to_add is not None:
            params = np.vstack((params, oscs_to_add))

        print(f"Editing result {index}")
        self._optimise_after_edit(params, result, index, **estimate_kwargs)

    def _optimise_after_edit(
        self,
        x0: np.ndarray,
        result: Result,
        index: int,
        **estimate_kwargs,
    ) -> None:
        for key in list(estimate_kwargs.keys()):
            if key in ("region", "noise_region", "region_unit", "initial_guess"):
                del estimate_kwargs[key]

        if result.get_noise_region() is None:
            region, noise_region = None, None
        else:
            region, noise_region = result.get_region()[-1], result.get_noise_region()[-1]
        self.estimate(
            region,
            noise_region,
            region_unit="hz",
            initial_guess=x0,
            _log=False,
            **estimate_kwargs,
        )

        del self._results[index]
        self._results.insert(index, self._results.pop(-1))

    def _get_slice(
        self,
        purpose: str,
        trim: Optional[Iterable[int]],
        shape: Iterable[int],
    ) -> Iterable[int]:
        default = iter(
            self.default_mpm_trim if purpose == "mpm"
            else self.default_nlp_trim
        )

        if trim is None:
            trim = copy.deepcopy(default)

        if len(self.proc_dims) == 1 and isinstance(trim, int):
            trim = [trim]

        trim_iter = iter(trim)
        shape_iter = iter(shape)
        slice_ = []
        for i in range(self.dim):
            shape_dim = next(shape_iter)
            if i not in self.proc_dims:
                slice_.append(slice(None, None, None))
            else:
                trim_dim = next(trim_iter)
                default_dim = next(default)
                if trim_dim is None:
                    if default_dim is None:
                        slice_.append(slice(0, shape_dim))
                    else:
                        slice_.append(slice(0, min(default_dim, shape_dim)))
                else:
                    slice_.append(slice(0, min(trim_dim, shape_dim)))

        return tuple(slice_)

    @property
    def _full_region(self) -> Iterable[Tuple[float, float]]:
        return self.convert(
            self._process_region(
                [
                    (0, self.data.shape[i] - 1)
                    for i in self.proc_dims
                ],
            ),
            "idx->hz",
        )

    def _process_indices(self, indices: Optional[Iterable[int]]) -> Iterable[int]:
        nres = len(self._results)
        if indices is None:
            return list(range(nres))
        return [idx % nres for idx in indices]

    def _reduce_region(
        self,
        region: Optional[Iterable[Tuple[float, float]]],
    ) -> Iterable[Tuple[int, int]]:
        return tuple(
            [
                r[i] for i, r in enumerate(region)
                if i in self.proc_dims
            ]
        )

    def _process_region(
        self,
        region: Optional[Iterable[Tuple[float, float]]],
    ) -> Iterable[Optional[Tuple[float, float]]]:
        if region is None:
            return None

        if len(self.proc_dims) == 1 and len(region) == 2:
            region = [region]
        region_iter = iter(region)
        return tuple(
            [
                tuple(next(region_iter)) if i in self.proc_dims
                else None
                for i in range(self.dim)
            ]
        )

    def _region_check(self, region: Any, region_unit: str, name: str):
        sws = self.sw(region_unit)
        offsets = self.offset(region_unit)
        return (
            name, region, sfuncs.check_region,
            (
                [sws[i] for i in self.proc_dims],
                [offsets[i] for i in self.proc_dims],
            ),
            {}, True,
        )

    def _estimate_banner(
        self,
        region: Optional[Iterable[Tuple[float, float]]],
        region_unit: str,
    ) -> str:
        if region is None:
            txt = "ESTIMATING ENTIRE SIGNAL"
        else:
            txt = "ESTIMATING REGION: "
            unit = region_unit.replace("h", "H")
            dim_strs = [
                f"{r[0]} - {r[1]} {unit} (F{i + 1})"
                for i, r in enumerate(region)
                if i in self.proc_dims
            ]
            txt += ", ".join(dim_strs)

        return ne._misc.boxed_text(txt, GRE)

    @property
    def _first_point_slice(self) -> Iterable[slice]:
        return tuple([
            slice(0, 1, None) if i in self.proc_dims
            else slice(None, None, None)
            for i in range(self.dim)
        ])

    # Commonly used sanity checks
    def _index_check(self, x: Any):
        return (
            "index", x, sfuncs.check_int, (),
            {"min_value": -len(self._results), "max_value": len(self._results) - 1},
        )

    def _indices_check(self, x: Any):
        return (
            "indices", x, sfuncs.check_int_list, (),
            {"min_value": -len(self._results), "max_value": len(self._results) - 1},
            True,
        )


class Result(ResultFetcher):

    def __init__(
        self,
        params: np.ndarray,
        errors: np.ndarray,
        region: Iterable[Tuple[float, float]],
        noise_region: Iterable[Tuple[float, float]],
        sfo: Iterable[float],
        trajectory: Optional[Iterable[np.ndarray]] = None,
    ) -> None:
        self.params = params
        self.errors = errors
        self.region = region
        self.noise_region = noise_region
        self.trajectory = trajectory
        super().__init__(sfo)

    def get_region(self, unit: str = "hz"):
        sanity_check(
            ("unit", unit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
        )
        return self.convert(self.region, f"hz->{unit}")

    def get_noise_region(self, unit: str = "hz"):
        sanity_check(
            ("unit", unit, sfuncs.check_frequency_unit, (self.hz_ppm_valid,)),
        )
        return self.convert(self.noise_region, f"hz->{unit}")
