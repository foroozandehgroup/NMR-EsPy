# _result.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 25 Mar 2022 11:47:26 GMT

import copy
from typing import Iterable, Optional

import numpy as np

from nmrespy import ExpInfo
from nmrespy._colors import RED, END, USE_COLORAMA
from nmrespy._freqconverter import FrequencyConverter
from nmrespy._sanity import (
    sanity_check,
    funcs as sfuncs,
)

if USE_COLORAMA:
    import colorama
    colorama.init()


class ResultFetcher(FrequencyConverter):

    def __init__(self, expinfo: ExpInfo, pts: Optional[Iterable[int]] = None) -> None:
        super().__init__(expinfo, pts)

    def get_result(self, funit: str = "hz"):
        """Returns the estimation result

        Parameters
        ----------
        funit
            The unit to express the frequencies in. Should be ``"hz"`` or ``"ppm"``.
        """
        sanity_check(("funit", funit, sfuncs.check_frequency_unit, (self.expinfo,)))
        return self._get_array("result", funit)

    def get_errors(self, funit: str = "hz"):
        """Returns the errors of the estimation result.

        Parameters
        ----------
        funit
            The unit to express the frequencies in. Should be ``"hz"`` or ``"ppm"``.
        """
        sanity_check(("funit", funit, sfuncs.check_frequency_unit, (self.expinfo,)))
        return self._get_array("errors", funit)

    def _get_array(self, name, funit):
        array = copy.deepcopy(getattr(self, name, None))
        if array is None:
            raise AttributeError(
                f"{RED}This class does not possess the attribute `{name}`{END}."
            )

        if funit == "ppm":
            f_idx = [2 + i for i in range(self.dim)]
            array[:, f_idx] = np.array(
                self.convert(
                    [a for a in array[:, f_idx].T], conversion="hz->ppm",
                )
            ).T

        return array
