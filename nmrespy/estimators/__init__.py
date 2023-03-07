# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 07 Mar 2023 12:12:08 GMT

from __future__ import annotations
import copy
import datetime
import functools
import io
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import matplotlib as mpl
from matplotlib import cm, pyplot as plt
import numpy as np
from pybaselines.classification import dietrich

import nmrespy as ne
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
from nmrespy import sig
from nmrespy.freqfilter import Filter
from nmrespy.mpm import MatrixPencil
from nmrespy.nlp import nonlinear_programming
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

        super().__init__(
            dim=expinfo.dim,
            sw=expinfo.sw(),
            offset=expinfo.offset(),
            sfo=expinfo.sfo,
            nuclei=expinfo.nuclei,
            default_pts=data.shape,
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
                fid = np.array(
                    eng.__getattr__(func).__call__(
                        *args, stdout=devnull, stderr=devnull,
                    )
                ).flatten()
            except matlab.engine.MatlabExecutionError:
                raise ValueError(
                    f"{RED}Something went wrong in trying to run Spinach.\n"
                    "Read what is stated below the line "
                    "\"matlab.engine.MatlabExecutionError:\" "
                    f"for more details on the error raised.{END}"
                )

        return fid

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
    def data(self) -> np.ndarray:
        """Return the data associated with the estimator."""
        return self._data

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
            The indices of results to return. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be returned.
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
            The indices of results to extract parameters from. Index ``0``
            corresponds to the first result obtained using the estimator, ``1``
            corresponds to the next, etc.  If ``None``, all results will be
            used.

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
            The indices of results to extract errors from. Index ``0`` corresponds to
            the first result obtained using the estimator, ``1`` corresponds to
            the next, etc.  If ``None``, all results will be used.

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
            The index of the result to edit. Index ``0`` corresponds to the
            first result obtained using the estimator, ``1`` corresponds to the
            next, etc. You can also use ``-1`` for the most recent result,
            ``-2`` for the second most recent, etc. By default, the most
            recently obtained result will be edited.

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

                new_oscs[:, 2 + self.dim :] = to_split[2 + self.dim :]

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

        self.estimate(
            result.get_region()[-1],
            result.get_noise_region()[-1],
            region_unit="hz",
            initial_guess=x0,
            _log=False,
            **estimate_kwargs,
        )

        del self._results[index]
        self._results.insert(index, self._results.pop(-1))

    def _process_indices(self, indices: Optional[Iterable[int]]) -> Iterable[int]:
        nres = len(self._results)
        if indices is None:
            return list(range(nres))
        return [idx % nres for idx in indices]

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


class _Estimator1DProc(Estimator):
    """Parent class for estimators which require processing/filtering in a single
    dimension. i.e. 1D, 2DJ."""

    @property
    def spectrum_first_direct(self) -> np.ndarray:
        """Generate a 1D spectrum of the first signal in the direct dimension.

        Generated by taking the first direct-dimension slice (``self.data[0]``),
        halving the initial point, and applying FT.
        """
        data = copy.deepcopy(self.data[0])
        data[0] *= 0.5
        return sig.ft(data)

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
        sanity_check(
            ("p0", p0, sfuncs.check_float),
            ("p1", p1, sfuncs.check_float),
            (
                "pivot", pivot, sfuncs.check_int, (),
                {"min_value": 0, "max_value": self.data.shape[-1] - 1},
            ),
        )

        ndim = self.data.ndim
        p0 = (ndim - 1) * [0.] + [p0]
        p1 = (ndim - 1) * [0.] + [p1]
        pivot = (ndim - 1) * [0.] + [pivot]

        spec = self.data
        spec[0] *= 0.5

        self._data = sig.ift(
            sig.phase(
                sig.ft(
                    spec,
                    axes=[-1],
                ),
                p0=p0,
                p1=p1,
                pivot=pivot,
            ),
            axes=[-1],
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
        sanity_check(
            ("max_p1", max_p1, sfuncs.check_float, (), {"greater_than_zero": True}),
        )
        if self.data.ndim == 1:
            spectrum = self.spectrum
        else:
            spectrum = self.spectrum_first_direct

        p0, p1 = sig.manual_phase_data(spectrum, max_p1=[max_p1])
        p0, p1 = p0[0], p1[0]
        self.phase_data(p0=p0, p1=p1)
        return p0, p1

    def exp_apodisation(self, k: float):
        """Apply an exponential window function to the direct dimnsion of the data.

        The window function is computed as ``np.exp(-k * np.linspace(0, 1, n))``,
        where ``n`` is the number of points in the direct dimension.
        """
        sanity_check(("k", k, sfuncs.check_float, (), {"greater_than_zero": True}))
        self._data = sig.exp_apodisation(self._data, k, axes=[-1])

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
        shape = self.data.shape
        direct_size = shape[-1]
        sanity_check(
            (
                "min_length", min_length, sfuncs.check_int, (),
                {"min_value": 1, "max_value": direct_size},
            ),
        )
        new_size = (2 * direct_size - 1) // 2
        new_shape = (*shape[:-1], new_size)
        new_data = np.zeros(new_shape, dtype="complex128")

        if self.data.ndim == 1:
            data = np.expand_dims(self.data, axis=0)
            new_data = np.expand_dims(new_data, axis=0)
        else:
            data = self.data

        for i, fid in enumerate(data):
            spectrum = sig.ft(sig.make_virtual_echo(fid)).real
            spectrum, _ = sig.baseline_correction(spectrum, min_length=min_length)
            new_data[i] = sig.ift(spectrum)[:new_size]

        self._data = new_data[0] if self.data.ndim == 1 else new_data
        new_default_pts = list(self.default_pts)
        new_default_pts[-1] = self._data.shape[-1]
        self._default_pts = tuple(new_default_pts)

    @logger
    def estimate(
        self,
        region: Optional[Tuple[float, float]] = None,
        noise_region: Optional[Tuple[float, float]] = None,
        region_unit: str = "hz",
        initial_guess: Optional[Union[np.ndarray, int]] = None,
        mode: str = "apfd",
        amp_thold: Optional[float] = None,
        phase_variance: bool = True,
        cut_ratio: Optional[float] = 1.1,
        mpm_trim: Optional[int] = None,
        nlp_trim: Optional[int] = None,
        hessian: str = "gauss-newton",
        max_iterations: Optional[int] = None,
        negative_amps: str = "remove",
        output_mode: Optional[int] = 10,
        save_trajectory: bool = False,
        epsilon: float = 1.0e-8,
        eta: float = 0.15,
        initial_trust_radius: float = 1.0,
        max_trust_radius: float = 4.0,
        _log: bool = True,
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
                (self.dim,), {}, True,
            ),
            ("hessian", hessian, sfuncs.check_one_of, ("gauss-newton", "exact")),
            ("phase_variance", phase_variance, sfuncs.check_bool),
            ("mode", mode, sfuncs.check_optimiser_mode),
            (
                "amp_thold", amp_thold, sfuncs.check_float, (),
                {"greater_than_zero": True}, True,
            ),
            (
                "mpm_trim", mpm_trim, sfuncs.check_int, (),
                {"min_value": 1}, True,
            ),
            (
                "nlp_trim", nlp_trim, sfuncs.check_int, (),
                {"min_value": 1}, True,
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
        )

        if region is None:
            region_unit = "hz"
            region = self._full_region
            noise_region = None
            mpm_signal = nlp_signal = self._data
            mpm_expinfo = nlp_expinfo = self.expinfo

        else:
            region = self._process_region(region)
            noise_region = self._process_region(noise_region)
            filter_ = Filter(
                self.data,
                self.expinfo,
                region,
                noise_region,
                region_unit=region_unit,
                twodim_dtype=None if self.dim == 1 else "hyper",
            )

            region = filter_.get_region()
            noise_region = filter_.get_noise_region()
            mpm_signal, mpm_expinfo = filter_.get_filtered_fid(cut_ratio=cut_ratio)
            nlp_signal, nlp_expinfo = filter_.get_filtered_fid(cut_ratio=None)

        mpm_trim = self._get_trim("mpm", mpm_trim, mpm_signal.shape[-1])
        nlp_trim = self._get_trim("nlp", nlp_trim, nlp_signal.shape[-1])

        if isinstance(initial_guess, np.ndarray):
            x0 = initial_guess
        else:
            oscillators = initial_guess if isinstance(initial_guess, int) else 0

            x0 = MatrixPencil(
                mpm_expinfo,
                mpm_signal[..., :mpm_trim],
                oscillators=oscillators,
                fprint=isinstance(output_mode, int),
            ).get_params()

            if x0 is None:
                self._results.append(
                    Result(
                        np.array([[]]),
                        np.array([[]]),
                        region[-1],
                        noise_region[-1],
                        self.sfo,
                    )
                )
                return

        if max_iterations is None:
            if hessian == "exact":
                max_iterations = self.default_max_iterations_exact_hessian
            elif hessian == "gauss-newton":
                max_iterations = self.default_max_iterations_gn_hessian

        result = nonlinear_programming(
            nlp_expinfo,
            nlp_signal[..., :nlp_trim],
            x0,
            phase_variance=phase_variance,
            hessian=hessian,
            mode=mode,
            amp_thold=amp_thold,
            max_iterations=max_iterations,
            negative_amps=negative_amps,
            output_mode=output_mode,
            save_trajectory=save_trajectory,
            tolerance=epsilon,
            eta=eta,
            initial_trust_radius=initial_trust_radius,
            max_trust_radius=max_trust_radius,
        )

        self._results.append(
            Result(
                result.x,
                result.errors,
                region,
                noise_region,
                self.sfo,
            )
        )

    def _get_trim(self, type_: str, trim: Optional[float], size: float):
        default = (
            self.default_mpm_trim if type_ == "mpm" else
            self.default_nlp_trim
        )

        if trim is None:
            if default is None:
                trim = size
            else:
                trim = min(default, size)
        else:
            trim = min(trim, size)

        return trim

    def predict_regions(
        self,
        unit: str = "hz",
        min_baseline_length: int = 15,
        true_region_filter: int = 200,
    ) -> Iterable[Tuple[float, float]]:
        # TODO: docstring
        # TODO: sanity checking
        if self.data.ndim == 1:
            data = copy.deepcopy(self.data)
        else:
            data = self.data[0]
        data[0] *= 0.5
        spectrum = ne.sig.ft(data).real
        shifts = self.get_shifts(unit=unit)[-1]
        mask = dietrich(
            spectrum,
            x_data=shifts,
            min_length=min_baseline_length,
        )[1]["mask"].astype(int)
        mask_diff = np.abs(np.diff(mask))
        flip_idxs, = np.nonzero(mask_diff == 1)
        flip_iter = iter(flip_idxs)

        region_idxs = []
        if mask[0] == 0:
            region_idxs.append(0)
        else:
            region_idxs.append(int(next(flip_iter)))
        region_idxs.extend([int(idx) for idx in flip_iter])
        if mask[-1] == 0:
            region_idxs.append(mask.size - 1)

        region_freqs = iter(self.convert([region_idxs], f"idx->{unit}")[0])
        regions = [(l, r) for l, r in zip(region_freqs, region_freqs)]  # noqa: E741

        # Filter away sufficiently small regions
        width_threshold = (
            self.convert([0], f"idx->{unit}")[0] -
            self.convert([true_region_filter], f"idx->{unit}")[0]
        )
        regions = list(filter(lambda r: r[0] - r[1] >= width_threshold, regions))

        return regions

    def view_proposed_regions(
        self,
        regions: Iterable[Tuple[float, float]],
        unit: str = "hz",
    ) -> None:
        # TODO docstring
        # TODO sanity checking
        if self.data.ndim == 1:
            data = copy.deepcopy(self.data)
        else:
            data = self.data[0]
        data[0] *= 0.5
        spectrum = ne.sig.ft(data).real
        shifts = self.get_shifts(unit=unit)[-1]

        fig, ax = plt.subplots()
        ax.set_xlim(shifts[0], shifts[-1])
        ax.plot(shifts, spectrum, color="k")
        colors = cm.viridis(np.linspace(0, 1, len(regions)))
        for region, color in zip(regions, colors):
            ax.axvspan(
                region[0],
                region[1],
                facecolor=color,
                alpha=0.6,
                edgecolor=None,
            )

        ax.set_yticks([])
        ax.set_xlabel(self._axis_freq_labels(unit)[-1])

        plt.show()

    @logger
    def subband_estimate(
        self,
        noise_region: Tuple[float, float],
        noise_region_unit: str = "hz",
        nsubbands: Optional[int] = None,
        **estimate_kwargs,
    ) -> None:
        r"""Perform estiamtion on the entire signal via estimation of
        frequency-filtered sub-bands.

        This method splits the signal up into ``nsubbands`` equally-sized region
        and extracts parameters from each region before finally concatenating all
        the results together.

        .. warning::

            This method is a work-in-progress. It is unlikely to produce decent
            results at the moment! I aim to improve the way that regions are
            created in the future.

        Parameters
        ----------
        noise_region
            Specifies a frequency range where no noticeable signals reside, i.e. only
            noise exists.

        noise_region_unit
            One of ``"hz"`` or ``"ppm"``. Specifies the units that ``noise_region``
            have been given in.

        nsubbands
            The number of sub-bands to break the signal into. If ``None``, the number
            will be set as the nearest integer to the data size divided by 500.

        estimate_kwargs
            Keyword arguments to give to :py:meth:`estimate`. Note that ``region``
            and ``initial_guess`` will be ignored.
        """
        sanity_check(
            self._funit_check(noise_region_unit, "noise_region_unit"),
            ("nsubbands", nsubbands, sfuncs.check_int, (), {"min_value": 1}, True),
        )
        sanity_check(
            self._region_check(noise_region, noise_region_unit, "noise_region"),
        )

        regions, mid_regions = self._get_subbands(nsubbands)
        nsubbands = len(regions)
        noise_region = self.convert(
            (self.dim - 1) * [None] + [noise_region],
            f"{noise_region_unit}->hz",
        )[-1]

        fprint = "fprint" not in estimate_kwargs or estimate_kwargs["fprint"]
        if fprint:
            print(f"Starting sub-band estimation using {nsubbands} sub-bands:")

        for string in ("region", "region_unit", "initial_guess", "_log"):
            if string in estimate_kwargs:
                del estimate_kwargs[string]

        params, errors = None, None
        for i, (region, mid_region) in enumerate(zip(regions, mid_regions), start=1):
            if fprint:
                msg = (
                    f"--> Estimating region #{i}: "
                    f"{mid_region[0]:.2f} - {mid_region[1]:.2f}Hz"
                )
                if self.hz_ppm_valid:
                    mid_region_ppm = self.convert(
                        (self.dim - 1) * [None] + [mid_region],
                        "hz->ppm"
                    )[-1]
                    msg += f" ({mid_region_ppm[0]:.3f} - {mid_region_ppm[1]:.3f}ppm)"
                print(msg)

            self.estimate(
                region, noise_region, region_unit="hz", _log=False,
                **estimate_kwargs,
            )
            p, e = self._keep_middle_freqs(self._results.pop(), mid_region)

            if p is None:
                continue
            if params is None:
                params = p
                errors = e
            else:
                params = np.vstack((params, p))
                errors = np.vstack((errors, e))

        # Sort in order of direct-dimension freqs.
        sort_idx = np.argsort(params[:, self.dim + 1])
        params = params[sort_idx]
        errors = errors[sort_idx]

        if fprint:
            print(f"{GRE}Sub-band estimation complete.{END}")

        self._results.append(
            Result(
                params,
                errors,
                region=self._full_region,
                noise_region=None,
                sfo=self.sfo,
            )
        )

    def _get_subbands(self, nsubbands: Optional[int]):
        # N.B. This is only appropriate for Estimator1D and Estimator2DJ
        if nsubbands is None:
            nsubbands = int(np.ceil(self.data.shape[-1] / 500))

        idxs, mid_idxs = self._get_subband_indices(nsubbands)
        shifts = self.get_shifts(meshgrid=False)[-1]
        regions = [(shifts[idx[0]], shifts[idx[1]]) for idx in idxs]
        mid_regions = [(shifts[mid_idx[0]], shifts[mid_idx[1]]) for mid_idx in mid_idxs]

        return regions, mid_regions

    def _get_subband_indices(
        self,
        nsubbands: int,
    ) -> Tuple[Iterable[Tuple[int, int]], Iterable[Tuple[int, int]]]:
        # (nsubbands - 2) full-size regions plus 2 half-size regions on each end.
        size = self.data.shape[-1]
        width = int(np.ceil(2 * size / (nsubbands - 1)))
        mid_width = int(np.ceil(width / 2))
        start_factor = int(np.ceil(size / (nsubbands - 1)))
        idxs = []
        mid_idxs = []
        for i in range(0, nsubbands - 2):
            start = i * start_factor
            mid_start = int(np.ceil((i + 0.5) * start_factor))
            if i == nsubbands - 3:
                idxs.append((start, size - 1))
            else:
                idxs.append((start, start + width))
            mid_idxs.append((mid_start, mid_start + mid_width))

        idxs.insert(0, (0, start_factor))
        idxs.append(((nsubbands - 2) * start_factor, size - 1))
        mid_idxs.insert(0, (0, mid_idxs[0][0]))
        mid_idxs.append((mid_idxs[-1][-1], size - 1))

        return idxs, mid_idxs

    def _keep_middle_freqs(
        self,
        result: Result,
        mid_region: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if result.params.size == 0:
            return None, None
        to_remove = (
            list(np.nonzero(result.params[:, 1 + self.dim] >= mid_region[0])[0]) +
            list(np.nonzero(result.params[:, 1 + self.dim] < mid_region[1])[0])
        )
        return (
            np.delete(result.params, to_remove, axis=0),
            np.delete(result.errors, to_remove, axis=0),
        )

    @logger
    def write_result(
        self,
        path: Union[Path, str] = "./nmrespy_result",
        indices: Optional[Iterable[int]] = None,
        fmt: str = "txt",
        description: Optional[str] = None,
        sig_figs: Optional[int] = 5,
        sci_lims: Optional[Tuple[int, int]] = (-2, 3),
        integral_mode: str = "relative",
        force_overwrite: bool = False,
        fprint: bool = True,
        pdflatex_exe: Optional[Union[str, Path]] = None,
    ) -> None:
        r"""Write estimation result tables to a text/PDF file.

        Parameters
        ----------
        path
            Path to save the result file to.

        indices
            The indices of results to include. Index ``0`` corresponds to the first
            result obtained using the estimator, ``1`` corresponds to the next, etc.
            If ``None``, all results will be included.

        fmt
            Must be one of ``"txt"`` or ``"pdf"``. If you wish to generate a PDF, you
            must have a LaTeX installation. See :ref:`LATEX_INSTALL`\.

        description
            Descriptive text to add to the top of the file.

        sig_figs
            The number of significant figures to give to parameters. If
            ``None``, the full value will be used. By default this is set to ``5``.

        sci_lims
            Given a value ``(-x, y)`` with ints ``x`` and ``y``, any parameter ``p``
            with a value which satisfies ``p < 10 ** -x`` or ``p >= 10 ** y`` will be
            expressed in scientific notation. If ``None``, scientific notation
            will never be used.

        integral_mode
            One of ``"relative"`` or ``"absolute"``.

            * If ``"relative"``, the smallest integral will be set to ``1``,
              and all other integrals will be scaled accordingly.
            * If ``"absolute"``, the absolute integral will be computed. This
              should be used if you wish to directly compare different datasets.

        force_overwrite
            Defines behaviour if the specified path already exists:

            * If ``False``, the user will be prompted if they are happy
              overwriting the current file.
            * If ``True``, the current file will be overwritten without prompt.

        fprint
            Specifies whether or not to print information to the terminal.

        pdflatex_exe
            The path to the system's ``pdflatex`` executable.

            .. note::

               You are unlikely to need to set this manually. It is primarily
               present to specify the path to ``pdflatex.exe`` on Windows when
               the NMR-EsPy GUI has been loaded from TopSpin.
        """
        self._check_results_exist()
        sanity_check(
            self._indices_check(indices),
            ("fmt", fmt, sfuncs.check_one_of, ("txt", "pdf")),
            ("description", description, sfuncs.check_str, (), {}, True),
            ("sig_figs", sig_figs, sfuncs.check_int, (), {"min_value": 1}, True),
            ("sci_lims", sci_lims, sfuncs.check_sci_lims, (), {}, True),
            (
                "integral_mode", integral_mode, sfuncs.check_one_of,
                ("relative", "absolute"),
            ),
            ("force_overwrite", force_overwrite, sfuncs.check_bool),
            ("fprint", fprint, sfuncs.check_bool),
        )
        sanity_check(("path", path, check_saveable_path, (fmt, force_overwrite)))

        indices = self._process_indices(indices)
        results = [self._results[i] for i in indices]
        writer = ResultWriter(
            self.expinfo,
            [result.get_params() for result in results],
            [result.get_errors() for result in results],
            description,
        )
        region_unit = "ppm" if self.hz_ppm_valid else "hz"

        titles = []
        for result in results:
            if result.get_region() is None:
                titles.append("Full signal")
            else:
                left, right = result.get_region(region_unit)[-1]
                titles.append(
                    f"{left:.3f} - {right:.3f} {region_unit}".replace("h", "H")
                )

        writer.write(
            path=path,
            fmt=fmt,
            titles=titles,
            parameters_sig_figs=sig_figs,
            parameters_sci_lims=sci_lims,
            integral_mode=integral_mode,
            force_overwrite=True,
            fprint=fprint,
            pdflatex_exe=pdflatex_exe,
        )

    def _plot_regions(
        self,
        indices: Iterable[int],
        region_unit: str,
    ) -> Tuple[Iterable[Iterable[int]], Iterable[Tuple[float, float]]]:
        regions = sorted(
            [
                (i, result.get_region(unit=region_unit)[-1])
                for i, result in enumerate(self.get_results())
                if i in indices
            ],
            key=lambda x: x[1][0],
            reverse=True,
        )

        # Merge overlapping/bordering regions
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

        return merge_indices, merge_regions

    def _configure_axes(
        self,
        fig: mpl.figure.Figure,
        axs: np.ndarray[mpl.Axes.axes],
        regions: Iterable[Tuple[float, float]],
        xaxis_ticks: Iterable[Tuple[int, Iterable[float]]],
        axes_left: float,
        axes_right: float,
        xaxis_label_height: float,
        region_unit: str,
    ) -> None:
        if axs.shape[0] > 1:
            for ax in axs[0]:
                ax.spines["bottom"].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in axs[1]:
                ax.spines["top"].set_visible(False)
        for region, ax_col in zip(regions, axs.T):
            for ax in ax_col:
                ax.set_xlim(*region)

        if len(regions) > 1:
            for axs_col in axs[:, :-1]:
                for ax in axs_col:
                    ax.spines["right"].set_visible(False)
            for axs_col in axs[:, 1:]:
                for ax in axs_col:
                    ax.spines["left"].set_visible(False)

            break_kwargs = {
                "marker": [(-1, -3), (1, 3)],
                "markersize": 6,
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
            for ax in axs[-1, :-1]:
                ax.plot([1], [0], transform=ax.transAxes, **break_kwargs)
            for ax in axs[-1, 1:]:
                ax.plot([0], [0], transform=ax.transAxes, **break_kwargs)
                ax.set_yticks([])

        if xaxis_ticks is not None:
            for i, ticks in xaxis_ticks:
                axs[-1, i].set_xticks(ticks)

        label = self._axis_freq_labels(region_unit)[-1]

        fig.text(
            x=(axes_left + axes_right) / 2,
            y=xaxis_label_height,
            s=label,
            horizontalalignment="center",
        )

    def _region_check(self, region: Any, region_unit: str, name: str):
        return (
            name, region, sfuncs.check_region,
            (
                (self.sw(region_unit)[-1],),
                (self.offset(region_unit)[-1],)
            ),
            {}, True,
        )

    def _process_region(self, direct_region: Tuple[float, float]):
        return tuple(
            [
                tuple(direct_region) if i == self.dim - 1
                else None
                for i in range(self.dim)
            ]
        )

    @property
    def _full_region(self) -> Tuple[float, float]:
        return self.convert(
            self._process_region((0, self.data.shape[-1] - 1)),
            "idx->hz",
        )


class Result(ResultFetcher):

    def __init__(
        self,
        params: np.ndarray,
        errors: np.ndarray,
        region: Iterable[Tuple[float, float]],
        noise_region: Iterable[Tuple[float, float]],
        sfo: Iterable[float],
    ) -> None:
        self.params = params
        self.errors = errors
        self.region = region
        self.noise_region = noise_region
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
