# _result_fetcher.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 30 Mar 2022 17:10:20 BST

import copy
from typing import Iterable, Optional

import numpy as np

from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._freqconverter import FrequencyConverter
from nmrespy._sanity import sanity_check
from nmrespy._sanity.funcs import check_frequency_unit

if USE_COLORAMA:
    import colorama
    colorama.init()


class ResultFetcher(FrequencyConverter):

    def __init__(
        self,
        sfo: Optional[Iterable[float]] = None,
    ) -> None:
        super().__init__(sfo=sfo)

    def get_result(self, funit: str = "hz"):
        """Returns the estimation result

        Parameters
        ----------
        funit
            The unit to express the frequencies in. Must be ``"hz"`` or ``"ppm"``.
        """
        sanity_check(("funit", funit, check_frequency_unit, (self.hz_ppm_valid,)))
        return self._get_array("result", funit)

    def get_errors(self, funit: str = "hz"):
        """Returns the errors of the estimation result.

        Parameters
        ----------
        funit
            The unit to express the frequencies in. Must be ``"hz"`` or ``"ppm"``.
        """
        sanity_check(("funit", funit, check_frequency_unit, (self.hz_ppm_valid,)))
        return self._get_array("errors", funit)

    def _get_array(self, name: str, funit: str) -> np.ndarray:
        array = copy.deepcopy(getattr(self, name, None))
        if array is None:
            raise AttributeError(
                f"{RED}This class does not possess the attribute `{name}`.{END}"
            )

        if funit == "ppm":
            dim = 2 + ((array.shape[1] - 2) // 2)
            f_idx = list(range(2, dim))
            array[:, f_idx] = np.array(
                self.convert(
                    [a for a in array[:, f_idx].T], conversion="hz->ppm",
                )
            ).T

        return array
