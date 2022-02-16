# core.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Mon 14 Feb 2022 10:15:05 GMT

from __future__ import annotations
from dataclasses import dataclass
import datetime
import functools
from pathlib import Path
import pickle
from typing import Iterable, Optional, Union

import numpy as np

from nmrespy import ExpInfo, GRE, ORA, RED, END, USE_COLORAMA
from nmrespy._sanity import sanity_check, funcs as sfuncs
from nmrespy.freqfilter import RegionIntFloatType, filter_spectrum
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import NonlinearProgramming
from nmrespy.sig import ft, make_fid, make_virtual_echo

if USE_COLORAMA:
    import colorama
    colorama.init()


class Estimator:
    """Estimation class.

    .. note::
       The methods :py:meth:`new_bruker`, :py:meth:`new_synthetic_from_data`
       and :py:meth:`new_synthetic_from_parameters` generate instances
       of the class. The method :py:meth:`from_pickle` loads an estimator
       instance that was previously saved using :py:meth:`to_pickle`.
       While you can manually input the listed parameters
       as arguments to initialise the class, it is more straightforward
       to use one of these.

    Parameters
    ----------
    data
        The data associated with the binary file in `path`.

    datapath
        The path to the directory containing the NMR data.

    expinfo
        Experiment information.
    """

    def __init__(
        self, data: np.ndarray, datapath: Optional[Path], expinfo: ExpInfo,
    ) -> None:
        self._data = data
        self._datapath = datapath
        self._expinfo = expinfo
        self._dim = self._expinfo.unpack("dim")
        self._results = []
        now = datetime.datetime.now()
        self._log = (
            "==============================\n"
            "Logfile for Estimator instance\n"
            "==============================\n"
            f"--> Instance created @ {now.strftime('%d-%m-%y %H:%M:%S')}\n"
        )

    def logger(f: callable) -> callable:
        """Decorator for logging method calls."""
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # The first arg is the class instance.
            # Append to the log text.
            args[0]._log += f"--> `{f.__name__}` {args[1:]} {kwargs}\n"
            return f(*args, **kwargs)
        return wrapper

    @property
    def dim(self):
        return self._dim

    @classmethod
    def new_synthetic_from_parameters(
        cls, parameters: np.ndarray, expinfo: ExpInfo, pts: Iterable[int], *,
        snr: float = 30.0,
    ) -> Estimator:
        """Generate an instance of :py:class:`Estimator` from an array of oscillator
        parameters.

        Parameters
        ----------
        parameters
            Parameter array with the following structure:

            * **1-dimensional data:**

              .. code:: python

                 parameters = numpy.array([
                    [a_1, φ_1, f_1, η_1],
                    [a_2, φ_2, f_2, η_2],
                    ...,
                    [a_m, φ_m, f_m, η_m],
                 ])

            * **2-dimensional data:**

              .. code:: python

                 params = numpy.array([
                    [a_1, φ_1, f1_1, f2_1, η1_1, η2_1],
                    [a_2, φ_2, f1_2, f2_2, η1_2, η2_2],
                    ...,
                    [a_m, φ_m, f1_m, f2_m, η1_m, η2_m],
                 ])

        expinfo
            Experiment information

        pts
            The number of points the signal comprises in each dimension.

        snr
            The signal-to-noise ratio. If ``None`` then no noise will be added
            to the FID.

        Returns
        -------
        estimator: :py:class:`Estimator`"""
        func_name = "Estimator.new_synthetic_from_parameters",
        sanity_check(func_name, ("expinfo", expinfo, sfuncs.check_expinfo))

        dim = expinfo.unpack("dim")
        sanity_check(
            func_name,
            ("parameters", parameters, sfuncs.check_parameter_array, (dim,)),
            ("pts", pts, sfuncs.check_points, (dim,)),
            ("snr", snr, sfuncs.check_positive_float, (), True),
        )

        data = make_fid(parameters, expinfo, pts, snr=snr)[0]
        return cls(data, None, expinfo)

    @logger
    def to_pickle(
        self, path: Optional[Union[Path, str]] = None, force_overwrite: bool = False,
        fprint: bool = True
    ) -> None:
        """Save the estimator to a byte stream using Python's pickling protocol.

        Parameters
        ----------
        path
            Path of file to save the byte stream to. `'.pkl'` is added to the end of
            the path if this is not given by the user. If ``None``,
            ``./estimator_<x>.pkl`` will be used, where ``<x>`` is the first number
            that doesn't cause a clash with an already existent file.

        force_overwrite
            Defines behaviour if the specified path already exists:

            * If ``force_overwrite`` is set to ``False``, the user will be prompted
              if they are happy overwriting the current file.
            * If ``force_overwrite`` is set to ``True``, the current file will be
              overwritten without prompt.

        fprint
            Specifies whether or not to print infomation to the terminal.

        See Also
        --------
        :py:meth:`from_pickle`
        """
        sanity_check(
            "Estimator.to_pickle",
            ("path", path, sfuncs.check_path, (), True),
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
            ("fprint", fprint, sfuncs.check_bool),
        )

        if path is None:
            x = 1
            while True:
                path = Path(f"estimator_{x}.pkl").resolve()
                if path.is_file():
                    x += 1
                else:
                    break

        path = Path(path).resolve()
        # Append extension to file path
        if path.suffix != ".pkl":
            path = path.with_suffix(".pkl")
        if path.is_file() and not force_overwrite:
            print(
                f"{ORA}to_pickle: `path` {path} already exists, and you have not "
                f"given permission to overwrite with `force_overwrite`. Skipping{END}."
            )
            return

        with open(path, "wb") as fh:
            pickle.dump(self, fh, pickle.HIGHEST_PROTOCOL)

        if fprint:
            print(f"{GRE}Saved estimator to {path}{END}")

    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> Estimator:
        """Load a pickled estimator instance.

        Parameters
        ----------
        path
            The path to the pickle file.

        Returns
        -------
        estimator : :py:class:`Estimator`

        Notes
        -----
        .. warning::
           `From the Python docs:`

           *"The pickle module is not secure. Only unpickle data you trust.
           It is possible to construct malicious pickle data which will
           execute arbitrary code during unpickling. Never unpickle data
           that could have come from an untrusted source, or that could have
           been tampered with."*

           You should only use :py:meth:`from_pickle` on files that
           you are 100% certain were generated using
           :py:meth:`to_pickle`. If you load pickled data from a .pkl file,
           and the resulting output is not an instance of
           :py:class:`Estimator`, an error will be raised.

        See Also
        --------
        py:meth:`Estimator.to_pickle`
        """
        path = Path(path).resolve()
        if path.suffix != ".pkl":
            path = path.with_suffix(".pkl")
        if not path.is_file():
            raise ValueError(f"{RED}Invalid path specified.{END}")

        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if isinstance(obj, __class__):
            return obj
        else:
            raise TypeError(
                f"{RED}It is expected that the object loaded by"
                " `from_pickle` is an instance of"
                f" {__class__.__module__}.{__class__.__qualname__}."
                f" What was loaded didn't satisfy this!{END}"
            )

    @logger
    def estimate(
        self,
        region: RegionIntFloatType,
        noise_region: RegionIntFloatType,
        *,
        region_unit: str = "ppm",
        initial_guess: Optional[np.ndarray, int] = None,
        hessian: str = "gauss-newton",
        phase_variance: bool = False,
        max_iterations: Optional[int] = None,
    ):
        """Estimate a specified region of the signal.

        The basic steps that this method carries out are:

        * Generate a frequency-filtered signal corresponding to the specified region.
        * (Optional) Generate an inital guess using the Matrix Pencil Method (MPM).
        * Apply numerical optimisation to determine a final estimate of the signal
          parameters

        Parameters
        ----------
        region
            The frequency range of interest.

        noise_region
            A frequency range where no noticeable signals reside, i.e. only noise
            exists.

        region_unit
            One of ``"hz"``, ``"ppm"``, ``"idx"``, corresponding to Hertz, parts per
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
            Specifies how to compute the Hessian.

            * ``"exact"`` - the exact analytical Hessian will be computed.
            * ``"gauss-newton"`` - the Hessian will be approximated as per the
              Guass-Newton method.

        phase_variance
            Whether or not to include the variance of oscillator phases in the cost
            function. This should be included in cases where the signal being
            considered is derived from phased data.

        max_iterations
            The greatest number of iterations to allow the optimiser to run before
            terminating. If ``None``, this number will be set to a default, depending
            on the identity of ``hessian``.
        """
        func_name = "Estimator.estimate"
        sanity_check(
            func_name,
            ("region_unit", region_unit, sfuncs.check_one_of, ("hz", "ppm", "idx")),
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
            if self._expinfo.unpack("sfo") is None:
                raise ValueError(
                    f"{RED}Cannot specify region in ppm. No information on "
                    f"transmitter frequency exists.{END}"
                )
            region_check = sfuncs.check_region_ppm
        else:
            region_check = sfuncs.check_region_idx

        sanity_check(
            func_name,
            ("region", region, region_check, (self._expinfo,)),
            ("noise_region", noise_region, region_check, (self._expinfo,)),
        )

        timestamp = datetime.datetime.now()
        filter_info = filter_spectrum(
            ft(make_virtual_echo([self._data])),
            self._expinfo,
            region,
            noise_region,
            region_unit=region_unit,
        )
        signal, expinfo = filter_info.get_filtered_fid()
        if isinstance(initial_guess, np.ndarray):
            x0 = initial_guess
        else:
            M = initial_guess if isinstance(initial_guess, int) else 0
            x0 = MatrixPencil(signal, expinfo, M=M).get_result()

        nlp_result = NonlinearProgramming(
            signal, x0, expinfo, phase_variance=phase_variance, hessian=hessian,
            max_iterations=max_iterations,
        )
        result, errors = nlp_result.get_result(), nlp_result.get_errors()
        self._results.append(
            Result(timestamp, signal, expinfo, result, errors)
        )


@dataclass
class Result:
    timestamp: datetime.datetime
    signal: np.ndarray
    expinfo: ExpInfo
    result: np.ndarray
    error: np.ndarray
