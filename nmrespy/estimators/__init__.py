# __init__.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 12 May 2023 12:04:47 BST

from __future__ import annotations
import datetime
import functools
import io
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np

import nmrespy as ne
from nmrespy._colors import RED, END, USE_COLORAMA
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
