# _freqconverter.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Fri 25 Mar 2022 11:40:22 GMT

import re
from typing import Any, Iterable, Optional, Union

from nmrespy._sanity import sanity_check, funcs as sfuncs


def check_frequency_conversion(
    obj: Any,
    ppm_valid: bool,
    idx_valid: bool,
) -> Optional[str]:
    pattern = r"^(idx|ppm|hz)->(idx|ppm|hz)$"
    if not bool(re.match(pattern, obj)):
        return (
            "Should be a str of the form \"{from}->{to}\", "
            "where {from} and {to} are each one of \"idx\", \"hz\", or \"ppm\""
        )
    if (not ppm_valid) and ("ppm" in obj):
        return "Cannot convert to/from ppm when sfo has not been specified."
    if (not idx_valid) and ("idx" in obj):
        return "Cannot convert to/from array indices when pts has not been specified."


class FrequencyConverter:
    """Handle frequency unit conversion."""

    def __init__(
        self,
        sw: Iterable[float],
        offset: Iterable[float],
        sfo: Optional[Iterable[float]] = None,
        pts: Optional[Iterable[int]] = None,
    ) -> None:
        """Initialise the converter.

        Parameters
        ----------
        sw
            The sweep width (spectral window) (Hz).

        offset
            The transmitter offset (Hz).

        sfo
            The transmitter frequency (MHz). If ``None``, conversion to and
            from ppm will not be possible.

        pts
            The points associated with the signal.
            If ``None``, conversion to and from array indices will not be
            possible.
        """
        # TODO: sanity check
        self._sw = sw
        self._offset = offset
        self._sfo = sfo
        self._pts = pts
        self.ppm_valid = self._sfo is not None
        self.idx_valid = self._pts is not None

    def __len__(self) -> int:
        return len(self._sw)

    def convert(
        self,
        lst: Iterable[Optional[Union[Iterable[Union[float, int]], float, int]]],
        conversion: str,
    ) -> Iterable[Union[int, float]]:
        """Convert quantities contained within an iterable.

        Parameters
        ----------
        lst
            An iterable of numerical values, with the same length as
            ``len(self)``.

        conversion
            A string denoting the coversion to be applied. Its form should be
            ``f"{from}->{to}"``, where ``"from"`` and ``"to"`` are each one of the
            following:

            * ``"idx"``: array index
            * ``"hz"``: Hertz
            * ``"ppm"``: parts per million

        Returns
        -------
        converted_lst
            An iterable of the same structure as ``lst``, with converted values.
        """
        sanity_check(
            ("lst", lst, sfuncs.check_convertible_list, (len(self),)),
            (
                "conversion", conversion, check_frequency_conversion,
                (self.ppm_valid, self.idx_valid),
            )
        )

        converted_lst = []
        for dim, elem in enumerate(lst):
            if elem is None:
                converted_lst.append(None)
            elif isinstance(elem, (tuple, list)):
                converted_lst.append(
                    type(elem)([
                        self._convert_value(x, dim, conversion) for x in elem
                    ])
                )
            else:
                # elem is a float/int...
                converted_lst.append(self._convert_value(elem, dim, conversion))

        return type(lst)(converted_lst)

    def _convert_value(
        self, value: Union[int, float], dim: int, conversion: str
    ) -> Union[int, float]:
        """Convert frequency."""
        sw = self._sw[dim]
        off = self._offset[dim]
        if "ppm" in conversion:
            sfo = self._sfo[dim]
            if sfo is None:
                return value
        if "idx" in conversion:
            pts = self._pts[dim]

        elif conversion == "idx->hz":
            return off + sw * (0.5 - (float(value) / (pts - 1)))

        elif conversion == "idx->ppm":
            return (off + sw * (0.5 - (float(value) / (pts - 1)))) / sfo

        elif conversion == "ppm->idx":
            return int(round(((pts - 1)) / (2 * sw) * (sw + 2 * (off - (value * sfo)))))

        elif conversion == "ppm->hz":
            return value * sfo

        elif conversion == "hz->idx":
            return int(round((pts - 1) / (2 * sw) * (sw + 2 * (off - value))))

        elif conversion == "hz->ppm":
            return value / sfo

        else:
            # conversion will be one of 'hz->hz', 'ppm->ppm', 'idx->idx'
            return value
