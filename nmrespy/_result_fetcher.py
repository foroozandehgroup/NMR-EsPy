# _result_fetcher.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Tue 26 Apr 2022 16:10:37 BST

import copy
import re
from typing import Any, Iterable, Optional

import numpy as np

from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._freqconverter import FrequencyConverter
from nmrespy._sanity import sanity_check
from nmrespy._sanity.funcs import check_frequency_unit

if USE_COLORAMA:
    import colorama
    colorama.init()


def check_sort_by(obj: Any, dim: int) -> Optional[str]:
    if not isinstance(obj, str):
        return "Should be a str."

    valids = (
        ["a", "p", "f-1", "d-1"] +
        ([f"f{i}" for i in range(1, dim + 1)] if dim > 1 else ["f", "f1"]) +
        ([f"d{i}" for i in range(1, dim + 1)] if dim > 1 else ["d", "d1"])
    )

    if obj not in valids:
        valid_list = ", ".join(valids)
        return f"Invalid value. Should be one of: {valid_list}."


class ResultFetcher(FrequencyConverter):

    def __init__(
        self,
        sfo: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__(sfo=sfo)

    def get_result(
        self,
        funit: str = "hz",
        sort_by: str = "f-1",
    ) -> np.ndarray:
        """Returns the estimation result

        Parameters
        ----------
        funit
            The unit to express the frequencies in. Must be ``"hz"`` or ``"ppm"``.

        sort_by
            Specifies the parameter by which the oscillators are ordered by.
            Should be one of ``"a"`` for amplitudes ``"p"`` for phase, ``"f<n>"``
            for frequency in the ``<n>``-th dimension, ``"d<n>"`` for the damping
            factor in the ``<n>``-th dimension. By setting ``<n>`` to ``-1``, the
            final (direct) dimension will be used. For 1D data, ``"f"`` and ``"d"``
            can be used to specify the frequency or damping factor.
        """
        dim = (self.result.shape[1] - 2) // 2
        sanity_check(
            ("funit", funit, check_frequency_unit, (self.hz_ppm_valid,)),
            ("sort_by", sort_by, check_sort_by, (dim,)),
        )
        return self._get_array("result", funit, sort_by)

    def get_errors(
        self,
        funit: str = "hz",
        sort_by: str = "f-1",
    ) -> np.ndarray:
        """Returns the errors of the estimation result.

        Parameters
        ----------
        funit
            The unit to express the frequencies in. Must be ``"hz"`` or ``"ppm"``.

        sort_by
            Specifies the parameter by which the oscillators are ordered by.
            Should be one of ``"a"`` for amplitudes ``"p"`` for phase, ``"f<n>"``
            for frequency in the ``<n>``-th dimension, ``"d<n>"`` for the damping
            factor in the ``<n>``-th dimension. By setting ``<n>`` to ``-1``, the
            final (direct) dimension will be used. For 1D data, ``"f"`` and ``"d"``
            can be used to specify the frequency or damping factor.
        """
        dim = (self.result.shape[1] - 2) // 2
        sanity_check(
            ("funit", funit, check_frequency_unit, (self.hz_ppm_valid,)),
            ("sort_by", sort_by, check_sort_by, (dim,)),
        )
        return self._get_array("errors", funit, sort_by)

    def _get_array(
        self,
        name: str,
        funit: str,
        sort_by: str,
    ) -> np.ndarray:
        array = copy.deepcopy(getattr(self, name, None))
        if array is None:
            raise AttributeError(
                f"{RED}This class does not possess the attribute `{name}`.{END}"
            )

        dim = ((array.shape[1] - 2) // 2)
        if funit == "ppm":
            f_idx = list(range(2, 2 + dim))
            array[:, f_idx] = np.array(
                self.convert(
                    [a for a in array[:, f_idx].T], conversion="hz->ppm",
                )
            ).T

        sort_idx = self._process_sort_by(sort_by, dim)
        return array[np.argsort(array[:, sort_idx])]

    @staticmethod
    def _process_sort_by(sort_by: str, ndim: int) -> int:
        if sort_by == "a":
            return 0
        elif sort_by == "p":
            return 1
        elif sort_by == "f":
            return 2
        elif sort_by == "d":
            return 3

        f_or_d, dim = re.fullmatch(r"(f|d)(-?\d)", sort_by).groups()

        if f_or_d == "f":
            pad = 1
        elif f_or_d == "d":
            pad = 1 + ndim

        if dim == "-1":
            return pad + ndim
        else:
            return pad + int(dim)
